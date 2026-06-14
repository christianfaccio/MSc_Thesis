"""
Oceananigans HYDROSTATIC free-surface model of a 5×5 km × 40 m patch of Abu
Dhabi coastal waters (Arabian Gulf shelf), driven by NW shamal wind stress +
Coriolis, with a cross-shore thermal FRONT (the source of spatial structure),
stratified T, buoyancy-active S ≈ 40 PSU, and randomized coastal pollution
sources.

MULTI-RUN DATASET GENERATION:
the script runs N independent simulations in one Julia session (amortizing the
package compile time), each with per-run randomized — but Abu-Dhabi-plausible —
parameters: sources (number, position, depth, Q), wind speed/direction around
the seasonal shamal, thermal-front geometry (position, width, ΔT, guarded by a
Richardson-number stability check), and a seeded initial-temperature
perturbation so identical parameters still produce different eddy fields.

    julia --project=oceananigans oceananigans/hydrostatic.jl \\
        --n-runs 8 --season winter --seed 1337

Options:
    --n-runs N           number of simulations (default 1)
    --season S           winter | summer (default winter)
    --seed N             base seed; run k is seeded by hash((seed, k)), so
                         re-running with --start-index k reproduces run k exactly
    --start-index N      index of the first run (default 1) — use to extend an
                         existing dataset without collisions
    --output-dir DIR     output directory (default data/oceananigans)
    --warmup-days D      spin-up before recording (default 1; keep ≳1 day — the
                         baroclinic eddies need ~15 h e-folding to develop)
    --recording-days D   recorded span per run (default 2, ≈192 snapshots/run;
                         cross-run randomization supplies most of the dataset
                         variability, so long single-run records add little)
    --debug              short run (2 h warmup + 6 h recording) for smoke tests

Output → <output-dir>/hydrostatic_<season>_run<NNN>.nc
        (u, v, w, T, S snapshots every 15 min of simulated time) plus a sidecar
        hydrostatic_<season>_run<NNN>.json recording every sampled parameter
        (sources in the config/sources.json schema, depth positive-down). The
        sidecar is written only after a successful run, so it doubles as a
        completion marker: runs whose sidecar exists are skipped.

WHY A FRONT (and why 5 km): a uniform wind on a periodic box stays horizontally
flat forever (u, v, T identical at every (x,y) ⇒ w ≡ 0). To get spatially
varying currents we add a cross-shore temperature front in thermal-wind balance.
At 5 km the domain spans ~2.5 deformation radii (Rd = N·H/f ≈ 2 km), so the
front goes baroclinically unstable and sheds resolved mesoscale eddies. A 1 km
box is < Rd and CANNOT host this — it would only overturn at the non-hydrostatic
(Kelvin-Helmholtz) scale and blow up.

WHY HYDROSTATIC (vs. the non-hydrostatic LES in non_hydrostatic.jl):
at km horizontal scale over tens of metres depth the flow is hydrostatic
(horizontal ≫ vertical scales), so a HydrostaticFreeSurfaceModel is both the
physically correct regime and far cheaper — it skips the 3-D pressure Poisson
solve and tolerates the large-aspect-ratio cells (≈39 m horizontal / ≈0.85 m
vertical) that would wreck a non-hydrostatic LES. Use this script for the
1-10 km training domain; keep non_hydrostatic.jl for the ~100 m turbulence-
resolving patch.

ARCHITECTURE (CPU vs GPU):
choose at run time with the OCEAN_ARCH environment variable —
    Mac (CPU):      OCEAN_ARCH=CPU julia --project=. hydrostatic.jl
    Jetson (GPU):   OCEAN_ARCH=GPU julia --project=. hydrostatic.jl
On the Jetson Orin Nano (ARM64, CUDA sm_87) CUDA.jl must be installed and
functional — verify first with `julia -e 'using CUDA; CUDA.functional()'`.
The CPU path is the guaranteed fallback.
"""

using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: HorizontalFormulation
using NCDatasets
using Printf
using Random

# ─────────────────────────────────────────────────────────────────────────────
# Architecture switch (CPU on the Mac, GPU/CUDA on the Jetson)
# ─────────────────────────────────────────────────────────────────────────────
const USE_GPU = uppercase(get(ENV, "OCEAN_ARCH", "CPU")) == "GPU"
if USE_GPU
	using CUDA	# loads OceananigansCUDAExt, which defines zero-arg GPU()
end
const ARCH = USE_GPU ? GPU(CUDABackend()) : CPU()
@info "Oceananigans architecture: $(ARCH)  (set OCEAN_ARCH=GPU|CPU to change)"

# ─────────────────────────────────────────────────────────────────────────────
# Domain size
# ─────────────────────────────────────────────────────────────────────────────
const LX    = 5000.0   # along-shore extent [m]
const LY    = 5000.0   # cross-shore extent [m]
const DEPTH = 40.0     # water column depth [m]

# ─────────────────────────────────────────────────────────────────────────────
# Abu Dhabi season presets (same physics baselines as non_hydrostatic.jl).
# wind_speed is the seasonal MEAN; each run samples around it and derives the
# kinematic stress via the bulk drag law τ = ρ_air · C_d · U² / ρ_water
# (3.87e-5 at the 5 m/s winter mean — the value the fixed-parameter script used).
# ─────────────────────────────────────────────────────────────────────────────
const SEASON_PARAMS = Dict(
    :winter => (T_surface = 22.0, T_bottom = 20.0, S_baseline = 40.0,
                N2 = 1.0e-5, wind_speed = 5.0),
    :summer => (T_surface = 33.0, T_bottom = 22.0, S_baseline = 42.0,
                N2 = 1.0e-4, wind_speed = 8.0),
)

const RHO_AIR   = 1.225    # air density [kg/m³]
const C_DRAG    = 1.3e-3   # 10-m neutral drag coefficient
const RHO_WATER = 1027.0   # seawater reference density [kg/m³]

# Abu Dhabi reference latitude 24.5°N → f ≈ 6.06e-5 s⁻¹.
const LATITUDE_DEG = 24.5
const Ω_EARTH      = 7.292e-5
const f_coriolis   = 2 * Ω_EARTH * sind(LATITUDE_DEG)

# Linear equation of state coefficients (set explicitly so the thermal-wind jet
# below uses the SAME α the model does).
const G_GRAV  = 9.81
const ALPHA_T = 1.67e-4    # thermal expansion  [1/°C]
const BETA_S  = 7.8e-4     # haline contraction [1/PSU]

# Salinity plume shape and strength (fixed across runs; sources vary instead).
# Each source is a Gaussian brine injection balanced by a linear relaxation back
# to baseline (timescale S_DECAY_TIME); the two balance at a steady core excess
# of S_SOURCE_ANOMALY PSU at the strongest source. Bounded brine ⇒ gentle
# sinking ⇒ no vertical-CFL blow-up.
const SIGMA_H          = 400.0    # plume horizontal std [m]
const SIGMA_V          = 12.0     # plume vertical std [m]
const S_SOURCE_ANOMALY = 10.0     # steady-state core salinity excess [PSU]
const S_DECAY_TIME     = 6hours   # relaxation timescale back to baseline
const γ_S              = 1 / S_DECAY_TIME

# Durations: warmup should stay ~1 day — the baroclinic eddies grow on an
# e-folding time of ~15 h (Eady: σ ≈ 0.3·f/√Ri) and the inertial period at
# 24.5°N is ~29 h, so a shorter spin-up records a still-laminar front.
# Recording defaults to 2 days: consecutive 15-min snapshots are correlated
# (eddy turnover ~hours), and dataset variability now comes from the
# randomized parameters across runs, so long single-run records add little.
const DEFAULT_WARMUP_DAYS    = 1.0
const DEFAULT_RECORDING_DAYS = 2.0
const OUTPUT_INTERVAL        = 15minutes
const DEBUG_DURATIONS        = (2hours, 6hours)   # --debug smoke-test override

# ─────────────────────────────────────────────────────────────────────────────
# Per-run randomized parameters.
#
# Front stability guard: the thermal-wind shear peaks at ∂u/∂z = gα·ΔT/(2f·W),
# so Ri = N²/(∂u/∂z)² ∝ W²·N²/ΔT². The known-stable baseline (ΔT=0.3, W=500,
# N²=1e-5) defines ri_rel = 1; samples with ri_rel < RI_REL_MIN risk a
# Kelvin-Helmholtz blow-up and are resampled (width clamped up as a last resort).
# ─────────────────────────────────────────────────────────────────────────────
const RI_REL_MIN = 0.31

const SourceSpec = NamedTuple{(:Q, :x0, :y0, :z0), NTuple{4, Float64}}

struct RunParams
    season       :: Symbol
    run_index    :: Int
    run_seed     :: UInt64
    wind_speed   :: Float64   # [m/s]
    wind_dir_deg :: Float64   # direction the wind blows FROM
    tau_mag      :: Float64   # kinematic stress [m²/s²], derived from wind_speed
    front_y      :: Float64   # front centre [m]
    front_width  :: Float64   # tanh half-width [m]
    delta_T      :: Float64   # cross-front temperature contrast [°C]
    ri_rel       :: Float64   # stability margin relative to the baseline front
    t_noise_amp  :: Float64   # IC temperature perturbation amplitude [°C]
    sources      :: Vector{SourceSpec}
end

unif(rng, lo, hi) = lo + (hi - lo) * rand(rng)

ri_relative(width, dT, N2) = (width / 500.0)^2 * (0.3 / dT)^2 * (N2 / 1.0e-5)

function sample_params(season::Symbol, run_index::Int, base_seed::Int)
    run_seed = hash((base_seed, run_index))   # resume-stable: depends only on (seed, index)
    rng = Xoshiro(run_seed)
    sp = SEASON_PARAMS[season]

    # Wind around the seasonal shamal (NW = 315°)
    wind_speed   = unif(rng, sp.wind_speed - 2.5, sp.wind_speed + 2.5)
    wind_dir_deg = unif(rng, 275.0, 355.0)
    tau_mag      = RHO_AIR * C_DRAG * wind_speed^2 / RHO_WATER

    # Front geometry, rejection-sampled against the Ri guard
    front_y     = unif(rng, 1500.0, 3500.0)
    front_width = 500.0
    delta_T     = 0.3
    ri_rel      = 1.0
    for _ in 1:100
        front_width = unif(rng, 300.0, 800.0)
        delta_T     = unif(rng, 0.2, 0.5)
        ri_rel      = ri_relative(front_width, delta_T, sp.N2)
        ri_rel >= RI_REL_MIN && break
    end
    if ri_rel < RI_REL_MIN   # clamp width up to the stability boundary
        front_width = 500.0 * (delta_T / 0.3) * sqrt(RI_REL_MIN / (sp.N2 / 1.0e-5))
        ri_rel      = ri_relative(front_width, delta_T, sp.N2)
    end

    # Sources: border-anchored like src/utils/sources.py random_sources —
    # half on the y=0 coast (the only true coast), half on the periodic x edge.
    n_sources = rand(rng, 3:6)
    sources = SourceSpec[]
    for _ in 1:n_sources
        Q     = unif(rng, 2.0, 10.0)
        depth = unif(rng, 0.0, 15.0)
        if rand(rng) < 0.5
            x0, y0 = unif(rng, 0.0, LX), 0.0
        else
            x0, y0 = 0.0, unif(rng, 0.0, LY)
        end
        push!(sources, (Q = Q, x0 = x0, y0 = y0, z0 = -depth))
    end

    t_noise_amp = unif(rng, 0.01, 0.02)

    return RunParams(season, run_index, run_seed, wind_speed, wind_dir_deg,
                     tau_mag, front_y, front_width, delta_T, ri_rel,
                     t_noise_amp, sources)
end

# ─────────────────────────────────────────────────────────────────────────────
# Progress callback
# ─────────────────────────────────────────────────────────────────────────────
function progress(simulation)
    u, v, w = simulation.model.velocities
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) m/s, wall: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time))
    @info msg
    flush(stderr)
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# One full simulation: build grid + model from the run's parameters, warm up,
# record. All parameter-dependent functions are closures over plain local
# bindings that are NEVER reassigned (reassignment creates a Core.Box, which
# breaks GPU kernel capture and CPU type stability). Sources are converted to a
# Tuple so the captured value is isbits — a Vector is not GPU-capturable.
# ─────────────────────────────────────────────────────────────────────────────
function build_and_run(arch, params::RunParams, output_path::AbstractString; warmup, recording)
    sp = SEASON_PARAMS[params.season]

    # Wind stress: a negative kinematic top-flux drives the matching velocity
    # component in the +axis direction (same convention as before).
    tau_x_top = -params.tau_mag * cosd(params.wind_dir_deg - 270)
    tau_y_top = +params.tau_mag * sind(params.wind_dir_deg - 270)

    # Locals captured by the closures (never reassigned)
    fy    = params.front_y
    fw    = params.front_width
    dT    = params.delta_T
    sbase = sp.S_baseline
    Tsurf = sp.T_surface
    Tbot  = sp.T_bottom
    srcs  = Tuple(params.sources)
    qmax  = maximum(s -> s.Q, srcs)

    # Cross-shore temperature front (warm inshore, cooler offshore) and its
    # thermal-wind-balanced along-shore jet.
    T_front(y)    = (dT / 2) * tanh((y - fy) / fw)
    dTdy_front(y) = (dT / (2 * fw)) * sech((y - fy) / fw)^2

    # Continuous injection balanced by a γ_S sink is exactly RELAXATION toward
    # `S_baseline + S_SOURCE_ANOMALY·(normalised plume)` at rate γ_S. We give
    # that target to Oceananigans' built-in `Relaxation`, which is type-stable —
    # a hand-rolled field-dependent `Forcing` boxed per cell and GC-thrashed the
    # step (Δt pinned, run stalled). The plume is the Q-weighted sum of source
    # Gaussians, normalised so the strongest source's core reaches S_SOURCE_ANOMALY.
    function S_target(x, y, z, t)
        excess = 0.0
        for src in srcs
            excess += (src.Q / qmax) * exp(-((x - src.x0)^2 + (y - src.y0)^2) / (2 * SIGMA_H^2)
                                           - (z - src.z0)^2 / (2 * SIGMA_V^2))
        end
        return sbase + S_SOURCE_ANOMALY * excess
    end

    T_init(x, y, z) = Tbot + (Tsurf - Tbot) * (1 + z / DEPTH) + T_front(y)
    S_init(x, y, z) = sbase
    U_geo(x, y, z)  = -(G_GRAV * ALPHA_T / f_coriolis) * dTdy_front(y) * (z + DEPTH)

    # Fail fast (on CPU too) if any closure captured a non-isbits value — on GPU
    # that would surface as a cryptic kernel-compilation error.
    @assert isbits(S_target) && isbits(T_init) && isbits(U_geo) "closures must capture only isbits values"

    # Grid: Periodic along-shore (x), Bounded cross-shore (y, coast at y=0) so
    # the front can live across y; Bounded in z. ≈39 m × 0.85 m cells.
    grid = RectilinearGrid(arch;
        size     = (128, 128, 48),
        x        = (0, LX),
        y        = (0, LY),
        z        = (-DEPTH, 0),              # z negative downward, surface at 0
        topology = (Periodic, Bounded, Bounded),
    )

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_x_top))
    v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_y_top))

    # Model: hydrostatic free-surface with buoyancy-active T and S.
    # closure: (1) vertical eddy diffusivity with VerticallyImplicitTimeDiscretization
    #   to remove the hidden explicit-diffusion step limit (Δt < dz²/2ν ≈ 35 s);
    #   (2) horizontal biharmonic diffusivity ν₄ = 200 m⁴/s to scrub the 2Δx grid
    #   noise the front/eddies generate (explicit, stable while Δt·ν₄ ≲ 4e4).
    # free_surface: ImplicitFreeSurface avoids the split-explicit barotropic
    #   instability (InexactError: Int64(NaN) from the substep-count calc).
    model = HydrostaticFreeSurfaceModel(grid;
        momentum_advection  = WENO(),
        tracer_advection    = WENO(),
        coriolis            = FPlane(f = f_coriolis),
        buoyancy            = SeawaterBuoyancy(equation_of_state =
                                LinearEquationOfState(thermal_expansion = ALPHA_T,
                                                      haline_contraction = BETA_S)),
        tracers             = (:T, :S),
        closure             = (VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization();
                                                         ν = 1e-2, κ = 1e-3),
                               ScalarBiharmonicDiffusivity(HorizontalFormulation();
                                                           ν = 200.0, κ = 200.0)),
        free_surface        = ImplicitFreeSurface(),
        boundary_conditions = (u = u_bcs, v = v_bcs),
        forcing             = (S = Relaxation(rate = γ_S, target = S_target),),
    )

    set!(model, T = T_init, S = S_init, u = U_geo)

    # Seeded IC temperature noise so identical parameters still diverge.
    # Generated on a CPU array (randn() inside a set! function would be neither
    # per-run seedable nor GPU-safe), then pushed back through set!.
    noise_rng = Xoshiro(params.run_seed + 1)
    T_cpu = Array(interior(model.tracers.T))
    T_cpu .+= params.t_noise_amp .* randn(noise_rng, size(T_cpu))
    set!(model, T = T_cpu)

    # Adaptive Δt. The explicit biharmonic term caps max_Δt (Δt·ν₄ ≲ 4e4 ⇒
    # Δt ≲ 200 s at ν₄=200, so 60 s leaves a 3× margin).
    simulation = Simulation(model, Δt = 5.0, stop_time = warmup)
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))
    wizard = TimeStepWizard(cfl = 0.7, max_change = 1.1, max_Δt = 60.0)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    @info "Run $(params.run_index) warmup ($(params.season), $(arch)): $(prettytime(warmup)) with no output..."
    run!(simulation)
    @info "Warmup complete at t = $(prettytime(time(simulation))); attaching output writer."

    simulation.output_writers[:fields] = NetCDFWriter(
        model,
        (u = model.velocities.u,
         v = model.velocities.v,
         w = model.velocities.w,
         T = model.tracers.T,
         S = model.tracers.S),
        filename           = output_path,
        schedule           = TimeInterval(OUTPUT_INTERVAL),
        overwrite_existing = true,
    )

    simulation.stop_time = warmup + recording

    @info "Run $(params.run_index) recording ($(prettytime(recording))) → $output_path"
    run!(simulation)
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Metadata sidecar: every sampled parameter, with sources in the
# config/sources.json schema ({name, x, y, depth, Q}, depth positive-down) so
# src/utils/sources.load_sources() reads it directly. Hand-rolled JSON to keep
# the Julia environment dependency-free.
# ─────────────────────────────────────────────────────────────────────────────
function write_metadata(path::AbstractString, params::RunParams, base_seed::Int;
                        warmup, recording, debug::Bool=false)
    src_lines = join(["""    { "name": "$(k)", "x": $(s.x0), "y": $(s.y0), "depth": $(-s.z0), "Q": $(s.Q) }"""
                      for (k, s) in enumerate(params.sources)], ",\n")
    open(path, "w") do io
        print(io, """
        {
          "season": "$(params.season)",
          "run_index": $(params.run_index),
          "base_seed": $(base_seed),
          "run_seed": $(params.run_seed),
          "debug": $(debug),
          "wind_speed": $(params.wind_speed),
          "wind_dir_deg": $(params.wind_dir_deg),
          "tau_mag": $(params.tau_mag),
          "front_y": $(params.front_y),
          "front_width": $(params.front_width),
          "delta_T_front": $(params.delta_T),
          "ri_rel": $(params.ri_rel),
          "t_noise_amp": $(params.t_noise_amp),
          "warmup_seconds": $(Float64(warmup)),
          "recording_seconds": $(Float64(recording)),
          "output_interval_seconds": $(Float64(OUTPUT_INTERVAL)),
          "sigma_h": $(SIGMA_H),
          "sigma_v": $(SIGMA_V),
          "s_source_anomaly": $(S_SOURCE_ANOMALY),
          "sources": [
        $(src_lines)
          ]
        }
        """)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# CLI parsing + main loop
# ─────────────────────────────────────────────────────────────────────────────
const USAGE = """
Usage: julia --project=oceananigans hydrostatic.jl [options]
  --n-runs N           number of simulations to run (default 1)
  --season S           winter | summer (default winter)
  --seed N             base seed; run k is seeded by hash((seed, k)) (default 1337)
  --start-index N      index of the first run (default 1)
  --output-dir DIR     output directory (default data/oceananigans)
  --warmup-days D      spin-up before recording (default $(DEFAULT_WARMUP_DAYS);
                       keep ≳1 — eddies need ~15 h to grow)
  --recording-days D   recorded span per run (default $(DEFAULT_RECORDING_DAYS))
  --debug              short run (2 h warmup + 6 h recording) for smoke tests
"""

function parse_cli(args::Vector{String})
    n_runs         = 1
    season         = :winter
    seed           = 1337
    start_index    = 1
    output_dir     = joinpath(@__DIR__, "..", "data", "oceananigans")
    warmup_days    = DEFAULT_WARMUP_DAYS
    recording_days = DEFAULT_RECORDING_DAYS
    debug          = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--n-runs"
            n_runs = parse(Int, args[i+1]); i += 2
        elseif a == "--season"
            season = Symbol(args[i+1]); i += 2
        elseif a == "--seed"
            seed = parse(Int, args[i+1]); i += 2
        elseif a == "--start-index"
            start_index = parse(Int, args[i+1]); i += 2
        elseif a == "--output-dir"
            output_dir = args[i+1]; i += 2
        elseif a == "--warmup-days"
            warmup_days = parse(Float64, args[i+1]); i += 2
        elseif a == "--recording-days"
            recording_days = parse(Float64, args[i+1]); i += 2
        elseif a == "--debug"
            debug = true; i += 1
        else
            error("Unknown argument '$(a)'\n" * USAGE)
        end
    end
    season in (:winter, :summer) || error("--season must be winter or summer\n" * USAGE)
    n_runs >= 1 || error("--n-runs must be >= 1\n" * USAGE)
    start_index >= 1 || error("--start-index must be >= 1\n" * USAGE)
    warmup_days > 0 && recording_days > 0 ||
        error("--warmup-days and --recording-days must be > 0\n" * USAGE)
    return (; n_runs, season, seed, start_index, output_dir,
              warmup_days, recording_days, debug)
end

function main(cli)
    mkpath(cli.output_dir)
    warmup, recording = cli.debug ? DEBUG_DURATIONS :
                        (cli.warmup_days * 1day, cli.recording_days * 1day)
    last_index = cli.start_index + cli.n_runs - 1
    for run_index in cli.start_index:last_index
        tag       = lpad(run_index, 3, '0')
        base      = joinpath(cli.output_dir, "hydrostatic_$(cli.season)_run$(tag)")
        nc_path   = base * ".nc"
        meta_path = base * ".json"
        if isfile(meta_path)
            @warn "Skipping run $(run_index): $(meta_path) already exists (delete it to re-run)."
            continue
        end
        params = sample_params(cli.season, run_index, cli.seed)
        @info @sprintf("Run %d/%d: wind %.2f m/s from %.0f°, front y=%.0f w=%.0f ΔT=%.2f (ri_rel=%.2f), %d sources",
                       run_index, last_index, params.wind_speed, params.wind_dir_deg,
                       params.front_y, params.front_width, params.delta_T,
                       params.ri_rel, length(params.sources))
        build_and_run(ARCH, params, nc_path; warmup, recording)
        write_metadata(meta_path, params, cli.seed; warmup, recording, debug = cli.debug)
        # Drop the previous run's grid/model/writer before building the next
        GC.gc(true)
        USE_GPU && CUDA.reclaim()
    end
    @info "All runs complete."
end

main(parse_cli(ARGS))
