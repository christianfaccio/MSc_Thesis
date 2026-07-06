"""
Oceananigans NON-HYDROSTATIC LES of a 1000×1000×100 m patch of Abu Dhabi
coastal waters (Arabian Gulf shelf), driven by NW shamal wind stress + Coriolis,
with stratified T (buoyancy-active) and a salinity field built from coastal
pollution/brine plumes relaxed toward a Gaussian target — tuned to realise a
~10 PSU salinity span for the RL homing task.

WHY NON-HYDROSTATIC AT 1 km (vs. the hydrostatic 5 km script):
a 1 km box is smaller than the baroclinic deformation radius (Rd = N·H/f ≈ 2 km),
so the front-driven mesoscale-eddy mechanism the hydrostatic script relies on
CANNOT operate here. Instead the spatial structure in this domain comes from
(a) the localized brine plumes themselves and (b) wind-driven boundary-layer
turbulence stirring them into filaments — both of which need the full 3-D
pressure solve. At 1000×1000×100 m with ~8 m horizontal / ~1.6 m vertical cells
the aspect ratio (~5) is mild enough for a non-hydrostatic LES to stay stable,
unlike the 39×0.85 m hydrostatic cells. Use THIS script for the 1 km × 100 m
turbulence-resolving training domain.

MULTI-RUN DATASET GENERATION (mirrors hydrostatic.jl):
runs N independent simulations in one Julia session (amortizing compile time),
each with per-run randomized — but Abu-Dhabi-plausible — parameters: sources
(number, position, depth, Q), wind speed/direction around the seasonal shamal,
and a seeded IC temperature perturbation so identical parameters still diverge.

    julia --project=oceananigans oceananigans/non_hydrostatic.jl \\
        --n-runs 4 --season winter --seed 1337

Output → <output-dir>/nonhydro_<season>_run<NNN>.nc
        (u, v, w, T, S snapshots) plus a sidecar nonhydro_<season>_run<NNN>.json
        recording every sampled parameter (sources in the config/sources.json
        schema, depth positive-down). The sidecar is written only after a
        successful run, so it doubles as a completion marker.

ARCHITECTURE (Leonardo Booster GPU vs. Mac CPU):
choose at run time with the OCEAN_ARCH environment variable —
    Mac (CPU):            OCEAN_ARCH=CPU julia --project=oceananigans non_hydrostatic.jl
    Leonardo (A100 GPU):  OCEAN_ARCH=GPU julia --project=oceananigans non_hydrostatic.jl
On Leonardo Booster (CINECA, 4× NVIDIA A100 64 GB per node, CUDA) launch one MPI
task / one Julia process per GPU under SLURM; CUDA.jl auto-selects the visible
device. Verify first with `julia -e 'using CUDA; CUDA.functional()'`. The CPU
path is the guaranteed fallback (slow — use OCEAN_NX/NY/NZ to shrink the grid
for a local smoke test, e.g. OCEAN_NX=32 OCEAN_NY=32 OCEAN_NZ=32).

GRID RESOLUTION is overridable via OCEAN_NX / OCEAN_NY / OCEAN_NZ (defaults
128 / 128 / 64) so the same script serves the full Leonardo run and a fast local
CPU debug without editing source. Use powers of two for the FFT pressure solver.
"""

using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf
using Random

# ─────────────────────────────────────────────────────────────────────────────
# Architecture switch (CPU on the Mac, GPU/CUDA on Leonardo Booster)
# ─────────────────────────────────────────────────────────────────────────────
const USE_GPU = uppercase(get(ENV, "OCEAN_ARCH", "CPU")) == "GPU"
if USE_GPU
    using CUDA   # loads OceananigansCUDAExt, which defines zero-arg GPU()
end
const ARCH = USE_GPU ? GPU(CUDABackend()) : CPU()
@info "Oceananigans architecture: $(ARCH)  (set OCEAN_ARCH=GPU|CPU to change)"

# ─────────────────────────────────────────────────────────────────────────────
# Domain size — 1 km × 1 km × 100 m turbulence-resolving coastal patch
# ─────────────────────────────────────────────────────────────────────────────
const LX    = 1000.0   # along-shore extent [m]
const LY    = 1000.0   # cross-shore extent [m]
const DEPTH = 100.0    # water column depth [m]

# Grid resolution (overridable so a local CPU smoke test can shrink it).
const NX = parse(Int, get(ENV, "OCEAN_NX", "128"))
const NY = parse(Int, get(ENV, "OCEAN_NY", "128"))
const NZ = parse(Int, get(ENV, "OCEAN_NZ", "64"))

# ─────────────────────────────────────────────────────────────────────────────
# Abu Dhabi season presets (same physics baselines as hydrostatic.jl).
# wind_speed is the seasonal MEAN; each run samples around it and derives the
# kinematic stress via the bulk drag law τ = ρ_air · C_d · U² / ρ_water.
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

# Linear equation of state. S is kept DYNAMICALLY PASSIVE (haline_contraction = 0)
# for this first pass: buoyancy comes from the T stratification only, so the
# strong (~10 PSU) brine anomaly cannot drive resolved convective downdrafts that
# would hammer the vertical CFL on the first (expensive) Leonardo run. The plume
# is still fully advected + LES-mixed, so the sensed field has real 3-D structure.
# To make the brine buoyancy-active later, set BETA_S = 7.8e-4.
const G_GRAV  = 9.81
const ALPHA_T = 1.67e-4    # thermal expansion  [1/°C]
const BETA_S  = 0.0        # haline contraction [1/PSU]; 0 ⇒ S passive (see note)

# ─────────────────────────────────────────────────────────────────────────────
# Salinity plume shape and strength.
# Each coastal source is a Gaussian brine injection balanced by a linear
# relaxation back to baseline (timescale S_DECAY_TIME). The two balance at a
# steady core excess of S_SOURCE_ANOMALY PSU at the strongest source, giving a
# controlled realised span: baseline (≈40 PSU) far from any source up to
# baseline + anomaly at the strongest core. A SHORT decay pins the field to its
# target so turbulent stirring cannot flatten the span (the failure mode of the
# original weak-forcing runs, which realised only ~1 PSU).
#
# σ_H is sized so overlapping coastal plumes fill the domain with a navigable
# gradient rather than leaving 90 %+ of it flat at baseline. The first 1 km run
# used σ_H = 80 m (~8 % of the box) and left only ~6 % of cells above baseline —
# an agent in the interior saw a gradient-free field. Widened to 300 m (~30 % of
# the box) and the source count raised (see sample_params) so the plumes overlap
# into a domain-scale slope. σ_V spans the upper column.
#
# The relaxation brings the field to S_SOURCE_ANOMALY·(normalised plume) on the
# timescale S_DECAY_TIME as (1 − e^{−t/τ}); τ short (15 min) so a ~30 min warmup
# (= 2τ) saturates before recording. The plume sum is normalised by its domain max
# (see build_and_run) so the realised core peak is EXACTLY S_SOURCE_ANOMALY above
# baseline regardless of how many wide sources overlap — span is set solely by the
# anomaly and stays put when σ_H or the source count change. So span ≈ 10 PSU here.
const SIGMA_H          = 300.0     # plume horizontal std [m]
const SIGMA_V          = 15.0      # plume vertical std [m]
const S_SOURCE_ANOMALY = 10.0      # core salinity excess above baseline [PSU] = the span
const S_DECAY_TIME     = 15minutes # relaxation timescale back to baseline
const γ_S              = 1 / S_DECAY_TIME

# Q-weighted sum of source Gaussians (the un-normalised plume shape). Top-level
# (not a closure) so it stays type-stable / GPU-capturable when called from the
# S_target closure; srcs is passed as an isbits Tuple.
@inline function plume_excess(x, y, z, srcs, qmax)
    e = 0.0
    for src in srcs
        e += (src.Q / qmax) * exp(-((x - src.x0)^2 + (y - src.y0)^2) / (2 * SIGMA_H^2)
                                  - (z - src.z0)^2 / (2 * SIGMA_V^2))
    end
    return e
end

# Durations. Wind-driven boundary-layer turbulence spins up in ~10–20 min; the
# plumes saturate in ~2·S_DECAY_TIME. Warmup default 30 min (=2τ) so the span is
# built before recording; record long enough for a few decorrelated snapshots.
const DEFAULT_WARMUP_MINUTES    = 30.0
const DEFAULT_RECORDING_MINUTES = 60.0
const OUTPUT_INTERVAL           = 2minutes
const DEBUG_DURATIONS           = (2minutes, 5minutes)   # --debug smoke test

# ─────────────────────────────────────────────────────────────────────────────
# Per-run randomized parameters (sources + wind + IC noise; no thermal front).
# ─────────────────────────────────────────────────────────────────────────────
const SourceSpec = NamedTuple{(:Q, :x0, :y0, :z0), NTuple{4, Float64}}

struct RunParams
    season       :: Symbol
    run_index    :: Int
    run_seed     :: UInt64
    wind_speed   :: Float64   # [m/s]
    wind_dir_deg :: Float64   # direction the wind blows FROM
    tau_mag      :: Float64   # kinematic stress [m²/s²], derived from wind_speed
    t_noise_amp  :: Float64   # IC temperature perturbation amplitude [°C]
    sources      :: Vector{SourceSpec}
end

unif(rng, lo, hi) = lo + (hi - lo) * rand(rng)

function sample_params(season::Symbol, run_index::Int, base_seed::Int)
    run_seed = hash((base_seed, run_index))   # resume-stable: depends only on (seed, index)
    rng = Xoshiro(run_seed)
    sp = SEASON_PARAMS[season]

    # Wind around the seasonal shamal (NW = 315°)
    wind_speed   = unif(rng, sp.wind_speed - 2.5, sp.wind_speed + 2.5)
    wind_dir_deg = unif(rng, 275.0, 355.0)
    tau_mag      = RHO_AIR * C_DRAG * wind_speed^2 / RHO_WATER

    # Sources: border-anchored like src/utils/sources.py random_sources —
    # half on the y=0 coast, half on the periodic x edge. Depths shallow
    # (coastal outfalls), so structure lives in the upper column. Count raised
    # 3:6 → 6:10 so the wider (σ_H=300) plumes overlap into domain-wide coverage.
    n_sources = rand(rng, 6:10)
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
                     tau_mag, t_noise_amp, sources)
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
# One full simulation. Parameter-dependent functions are closures over plain
# local bindings that are NEVER reassigned (reassignment creates a Core.Box,
# breaking GPU kernel capture and CPU type stability). Sources are converted to a
# Tuple so the captured value is isbits — a Vector is not GPU-capturable.
# ─────────────────────────────────────────────────────────────────────────────
function build_and_run(arch, params::RunParams, output_path::AbstractString; warmup, recording)
    sp = SEASON_PARAMS[params.season]

    # Wind stress: a negative kinematic top-flux drives the matching velocity
    # component in the +axis direction.
    tau_x_top = -params.tau_mag * cosd(params.wind_dir_deg - 270)
    tau_y_top = +params.tau_mag * sind(params.wind_dir_deg - 270)

    # Locals captured by the closures (never reassigned)
    sbase = sp.S_baseline
    Tsurf = sp.T_surface
    Tbot  = sp.T_bottom
    srcs  = Tuple(params.sources)
    qmax  = maximum(s -> s.Q, srcs)

    # Normalise the plume sum by its domain max so the realised core peak is
    # exactly S_SOURCE_ANOMALY above baseline no matter how many wide (σ_H) sources
    # overlap — decouples the salinity SPAN (the anomaly) from spatial COVERAGE
    # (σ_H, source count). Without it, N overlapping σ_H=300 plumes drive the sum
    # past 1 and the span balloons (8 sources ⇒ ~20 PSU). The global max of a sum
    # of Gaussians sits at a source core, so scan the centres plus a coarse grid.
    # Single assignment — reassigning a captured local boxes it (Core.Box) and
    # breaks the isbits/GPU-capture check below.
    excess_max = max(
        maximum(s -> plume_excess(s.x0, s.y0, s.z0, srcs, qmax), srcs),
        maximum(plume_excess(x, y, z, srcs, qmax)
                for x in range(0, LX; length=48),
                    y in range(0, LY; length=48),
                    z in range(-DEPTH, 0; length=24)),
    )

    # Continuous injection balanced by a γ_S sink is exactly RELAXATION toward this
    # target at rate γ_S. Oceananigans' built-in `Relaxation` is type-stable — a
    # hand-rolled field-dependent Forcing boxes per cell and GC-thrashes the step.
    S_target(x, y, z, t) = sbase + S_SOURCE_ANOMALY * plume_excess(x, y, z, srcs, qmax) / excess_max

    # Stable T stratification (surface warmer) over the 100 m column.
    T_init(x, y, z) = Tbot + (Tsurf - Tbot) * (1 + z / DEPTH)
    S_init(x, y, z) = sbase

    # Fail fast if any closure captured a non-isbits value (cryptic GPU error).
    @assert isbits(S_target) && isbits(T_init) "closures must capture only isbits values"

    # Grid: Periodic along-shore (x), Bounded cross-shore (y, coast at y=0),
    # Bounded in z. ~7.8 m × 1.6 m cells at the default 128×128×64.
    grid = RectilinearGrid(arch;
        size     = (NX, NY, NZ),
        x        = (0, LX),
        y        = (0, LY),
        z        = (-DEPTH, 0),              # z negative downward, surface at 0
        topology = (Periodic, Bounded, Bounded),
    )

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_x_top))
    v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_y_top))

    # Non-hydrostatic LES: AMD subgrid closure, buoyancy-active T (S passive via
    # BETA_S = 0), Gaussian brine plumes via Relaxation.
    model = NonhydrostaticModel(grid;
        advection           = WENO(),
        coriolis            = FPlane(f = f_coriolis),
        closure             = AnisotropicMinimumDissipation(),
        tracers             = (:T, :S),
        buoyancy            = SeawaterBuoyancy(equation_of_state =
                                LinearEquationOfState(thermal_expansion  = ALPHA_T,
                                                      haline_contraction = BETA_S)),
        boundary_conditions = (u = u_bcs, v = v_bcs),
        forcing             = (S = Relaxation(rate = γ_S, target = S_target),),
    )

    # Small random kicks on u/w concentrated in the upper few metres to seed
    # boundary-layer turbulence; T stratification + S baseline.
    u★ = sqrt(params.tau_mag)
    Ξ(z) = randn() * exp(z / 4)
    uᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
    wᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
    set!(model, u = uᵢ, w = wᵢ, T = T_init, S = S_init)

    # Seeded IC temperature noise so identical parameters still diverge.
    noise_rng = Xoshiro(params.run_seed + 1)
    T_cpu = Array(interior(model.tracers.T))
    T_cpu .+= params.t_noise_amp .* randn(noise_rng, size(T_cpu))
    set!(model, T = T_cpu)

    # Adaptive Δt — LES + resolved turbulence set the CFL limit.
    simulation = Simulation(model, Δt = 0.5, stop_time = warmup)
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))
    wizard = TimeStepWizard(cfl = 0.7, max_change = 1.1, max_Δt = 3.0)
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
# Metadata sidecar: sampled parameters, sources in config/sources.json schema
# ({name, x, y, depth, Q}, depth positive-down) so src/utils/sources.load_sources
# reads it directly. Hand-rolled JSON to keep the Julia env dependency-free.
# ─────────────────────────────────────────────────────────────────────────────
function write_metadata(path::AbstractString, params::RunParams, base_seed::Int;
                        warmup, recording, debug::Bool=false)
    src_lines = join(["""    { "name": "$(k)", "x": $(s.x0), "y": $(s.y0), "depth": $(-s.z0), "Q": $(s.Q) }"""
                      for (k, s) in enumerate(params.sources)], ",\n")
    open(path, "w") do io
        print(io, """
        {
          "model": "non_hydrostatic",
          "season": "$(params.season)",
          "run_index": $(params.run_index),
          "base_seed": $(base_seed),
          "run_seed": $(params.run_seed),
          "debug": $(debug),
          "domain": { "LX": $(LX), "LY": $(LY), "DEPTH": $(DEPTH) },
          "grid": { "NX": $(NX), "NY": $(NY), "NZ": $(NZ) },
          "wind_speed": $(params.wind_speed),
          "wind_dir_deg": $(params.wind_dir_deg),
          "tau_mag": $(params.tau_mag),
          "t_noise_amp": $(params.t_noise_amp),
          "warmup_seconds": $(Float64(warmup)),
          "recording_seconds": $(Float64(recording)),
          "output_interval_seconds": $(Float64(OUTPUT_INTERVAL)),
          "sigma_h": $(SIGMA_H),
          "sigma_v": $(SIGMA_V),
          "s_source_anomaly": $(S_SOURCE_ANOMALY),
          "s_decay_seconds": $(Float64(S_DECAY_TIME)),
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
Usage: julia --project=oceananigans non_hydrostatic.jl [options]
  --n-runs N              number of simulations to run (default 1)
  --season S              winter | summer (default winter)
  --seed N                base seed; run k is seeded by hash((seed, k)) (default 1337)
  --start-index N         index of the first run (default 1)
  --output-dir DIR        output directory (default data/oceananigans)
  --warmup-minutes M      spin-up before recording (default $(DEFAULT_WARMUP_MINUTES))
  --recording-minutes M   recorded span per run (default $(DEFAULT_RECORDING_MINUTES))
  --debug                 short run (2 min warmup + 5 min recording) for smoke tests
Env: OCEAN_ARCH=GPU|CPU, OCEAN_NX/NY/NZ override grid resolution.
"""

function parse_cli(args::Vector{String})
    n_runs            = 1
    season            = :winter
    seed              = 1337
    start_index       = 1
    output_dir        = joinpath(@__DIR__, "..", "data", "oceananigans")
    warmup_minutes    = DEFAULT_WARMUP_MINUTES
    recording_minutes = DEFAULT_RECORDING_MINUTES
    debug             = false
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
        elseif a == "--warmup-minutes"
            warmup_minutes = parse(Float64, args[i+1]); i += 2
        elseif a == "--recording-minutes"
            recording_minutes = parse(Float64, args[i+1]); i += 2
        elseif a == "--debug"
            debug = true; i += 1
        else
            error("Unknown argument '$(a)'\n" * USAGE)
        end
    end
    season in (:winter, :summer) || error("--season must be winter or summer\n" * USAGE)
    n_runs >= 1 || error("--n-runs must be >= 1\n" * USAGE)
    start_index >= 1 || error("--start-index must be >= 1\n" * USAGE)
    warmup_minutes > 0 && recording_minutes > 0 ||
        error("--warmup-minutes and --recording-minutes must be > 0\n" * USAGE)
    return (; n_runs, season, seed, start_index, output_dir,
              warmup_minutes, recording_minutes, debug)
end

function main(cli)
    mkpath(cli.output_dir)
    warmup, recording = cli.debug ? DEBUG_DURATIONS :
                        (cli.warmup_minutes * 1minute, cli.recording_minutes * 1minute)
    last_index = cli.start_index + cli.n_runs - 1
    for run_index in cli.start_index:last_index
        tag       = lpad(run_index, 3, '0')
        base      = joinpath(cli.output_dir, "nonhydro_$(cli.season)_run$(tag)")
        nc_path   = base * ".nc"
        meta_path = base * ".json"
        if isfile(meta_path)
            @warn "Skipping run $(run_index): $(meta_path) already exists (delete it to re-run)."
            continue
        end
        params = sample_params(cli.season, run_index, cli.seed)
        @info @sprintf("Run %d/%d: wind %.2f m/s from %.0f°, %d sources, grid %d×%d×%d",
                       run_index, last_index, params.wind_speed, params.wind_dir_deg,
                       length(params.sources), NX, NY, NZ)
        build_and_run(ARCH, params, nc_path; warmup, recording)
        write_metadata(meta_path, params, cli.seed; warmup, recording, debug = cli.debug)
        GC.gc(true)
        USE_GPU && CUDA.reclaim()
    end
    @info "All runs complete."
end

main(parse_cli(ARGS))
