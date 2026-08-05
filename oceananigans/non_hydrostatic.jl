"""
Oceananigans NON-HYDROSTATIC LES of a 1000×1000×100 m patch of Abu Dhabi
coastal waters (Arabian Gulf shelf), driven by NW-quadrant wind stress (the
seasonal shamal DIRECTION, at climatological-mean SPEED — not a storm) +
Coriolis, with stratified T carrying the buoyancy and a DYNAMICALLY PASSIVE
salinity field built from coastal pollution/brine plumes relaxed toward a
Gaussian target — tuned to realise a ~10 PSU salinity span for the RL homing
task.

SPIN-UP IS THE BINDING CONSTRAINT ON THE CURRENTS (fixed 2026-08-05).
The wind-driven flow equilibrates on the Ekman timescale 1/f = 4.6 h (inertial
period 2π/f = 28.9 h). The original 30 min warmup + 60 min recording stopped the
runs at 0.05 inertial periods: measured currents reached only ~2.5% of their
equilibrium value (mean |u,v| ≈ 0.004 m/s) and were still rising ~2× per hour
when recording ended. Since the RL agent swims at 1 m/s, that produced an
essentially motionless ocean, AND left the passive tracer unstirred — advected
just 11 m in an hour (4% of a plume width), so the sensed field was frozen at
the analytic Gaussian target (corr 0.999 between first and last snapshot).
Warmup is now 12 h (≈2.6/f) so the flow equilibrates to the ~3%-of-wind-speed
rule of thumb, i.e. ~0.15 m/s for a 5 m/s wind — realistic for the southern
Gulf shelf and a comfortable 6× below the vehicle's own speed.

DO NOT "FIX" WEAK CURRENTS BY MAKING SALINITY BUOYANT. Setting BETA_S = 7.8e-4
with the 10 PSU anomaly gives a reduced gravity g' = g·β_S·ΔS = 0.077 m/s², a
gravity-current velocity scale √(g'·σ_V) ≈ 1.75 m/s, and measured currents 164×
stronger than the passive case — faster than the vehicle over 12% of the domain.
It also HALVES the salinity span (9.9 → 4.8 PSU) and halves the vertical
gradient, because the convection mixes the column. That was the state of the
`buoyancy_active` dataset; keep S passive and lengthen the spin-up instead.

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
using Statistics: mean

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
# The stratification is specified by the T end-points, NOT by an N² knob: the
# old `N2` field here was dead (nothing ever read it) and disagreed with the
# profile actually built by T_init, so it has been removed rather than left as a
# phantom parameter. The implied buoyancy frequency is N² = g·α_T·dT/dz:
#   winter  ΔT = 2 °C over 100 m  ->  N² = 3.28e-5 s⁻²
#   summer  ΔT = 11 °C over 100 m ->  N² = 1.80e-4 s⁻²
const SEASON_PARAMS = Dict(
    :winter => (T_surface = 22.0, T_bottom = 20.0, S_baseline = 40.0,
                wind_speed = 5.0),
    :summer => (T_surface = 33.0, T_bottom = 22.0, S_baseline = 42.0,
                wind_speed = 8.0),
)

const RHO_AIR   = 1.225    # air density [kg/m³]
const C_DRAG    = 1.3e-3   # 10-m neutral drag coefficient
const RHO_WATER = 1027.0   # seawater reference density [kg/m³]

# Abu Dhabi reference latitude 24.5°N → f ≈ 6.06e-5 s⁻¹.
const LATITUDE_DEG = 24.5
const Ω_EARTH      = 7.292e-5
const f_coriolis   = 2 * Ω_EARTH * sind(LATITUDE_DEG)

# Linear equation of state. S is DYNAMICALLY PASSIVE (haline_contraction = 0):
# buoyancy comes from the T stratification only, so the strong (~10 PSU) brine
# anomaly cannot drive resolved convective downdrafts. "Passive" means no
# feedback into momentum — the plume is still fully advected + LES-mixed by the
# flow, so the sensed field has real 3-D structure.
#
# This is a DELIBERATE, LOAD-BEARING ZERO — see the header. β_S = 7.8e-4 makes
# the brine buoyancy-active and blows the currents up 164× while halving the
# salinity span. If you change it, regenerate everything and re-check the
# current statistics with scripts/plot_oceananigans_currents.py.
const G_GRAV  = 9.81
const ALPHA_T = 1.67e-4    # thermal expansion  [1/°C]
const BETA_S  = 0.0        # haline contraction [1/PSU]; 0 = passive S (see header)

# M2 principal lunar semidiurnal tide, 12.4206 h. The Arabian Gulf is tidally
# energetic and wind-only forcing is the largest physical omission here, but the
# tide is OFF by default (--tide-amplitude 0) so the wind-driven spin-up baseline
# stays isolated and reproducible. Enable it once that baseline is validated.
const M2_PERIOD = 12.4206 * 3600      # [s]
const M2_OMEGA  = 2π / M2_PERIOD      # [rad/s]

# Barotropic tidal body force. du/dt = F·cos(ωt) integrates to u = (F/ω)·sin(ωt),
# so F = U_tide·ω sets the tidal current AMPLITUDE directly. Only u is forced;
# Coriolis turns the rectilinear forcing into the observed rotary ellipse.
# Top-level (not a closure) so it stays GPU-capturable.
@inline tide_forcing(x, y, z, t, p) = p.F * cos(p.ω * t)

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
const SIGMA_V          = 40.0      # plume vertical std [m]
const S_SOURCE_ANOMALY = 10.0      # core salinity excess above baseline [PSU] = the span
# Relaxation timescale. RAISED 15 min -> 1 h (2026-08-05): the relaxation and
# advection compete to set the field, and at 15 min the forcing won by ~80×, so
# the tracer stayed pinned to the smooth analytic target no matter how well the
# flow was stirred. With an equilibrated ~0.15 m/s flow the advective timescale
# across a plume is σ_H/U ≈ 2000 s, so a 1 h relaxation lets stirring actually
# fold the field into filaments while still holding the span up against mixing.
# This is the knob to check first if the smoke test comes out too smooth (raise
# it) or if the span has collapsed below ~8 PSU (lower it).
const S_DECAY_TIME     = 1hour
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
# Warmup 12 h ≈ 2.6 Ekman timescales (1/f = 4.6 h) so the wind-driven flow
# EQUILIBRATES — see the header; the old 30 min stopped it at 2.5% of its final
# speed. Also comfortably past 2·S_DECAY_TIME, so the plumes are saturated.
# Recording 6 h at 12 min intervals keeps the snapshot count (and so the file
# size, ~650 MB) where it was while spreading the frames over a window long
# enough that consecutive snapshots decorrelate: the advective timescale across
# a plume is σ_H/U ≈ 33 min at the equilibrated speed, so 2 min frames were
# heavily redundant. The env draws a random frozen snapshot per episode, so
# decorrelated frames are what buys episode-to-episode variety.
const DEFAULT_WARMUP_MINUTES          = 720.0
const DEFAULT_RECORDING_MINUTES       = 360.0
const DEFAULT_OUTPUT_INTERVAL_MINUTES = 12.0
# --debug is a plumbing smoke test ONLY (does it run, does it write a file). It
# is FAR too short to say anything about spin-up or the current statistics.
# It carries its OWN output interval: the production 12 min interval is longer
# than the whole debug recording, which would silently write an EMPTY file.
const DEBUG_DURATIONS           = (2minutes, 5minutes)
const DEBUG_OUTPUT_INTERVAL     = 1minute

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
    tide_amp     :: Float64   # M2 tidal current amplitude [m/s]; 0 = no tide
    sources      :: Vector{SourceSpec}
end

unif(rng, lo, hi) = lo + (hi - lo) * rand(rng)

function sample_params(season::Symbol, run_index::Int, base_seed::Int, tide_amp::Float64)
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
    n_sources = rand(rng, 10:30)
    sources = SourceSpec[]
    for _ in 1:n_sources
        Q     = unif(rng, 2.0, 10.0)
        depth = unif(rng, 0.0, 0.7*DEPTH)
        if rand(rng) < 0.5
            x0, y0 = unif(rng, 0.0, LX), 0.0
        else
            x0, y0 = 0.0, unif(rng, 0.0, LY)
        end
        push!(sources, (Q = Q, x0 = x0, y0 = y0, z0 = -depth))
    end

    t_noise_amp = unif(rng, 0.01, 0.02)

    return RunParams(season, run_index, run_seed, wind_speed, wind_dir_deg,
                     tau_mag, t_noise_amp, tide_amp, sources)
end

# ─────────────────────────────────────────────────────────────────────────────
# Progress callback
# ─────────────────────────────────────────────────────────────────────────────
# `umean` is the spin-up diagnostic: with the wind-driven flow equilibrating on
# 1/f = 4.6 h, this is the number to watch flatten out. If it is still climbing
# when the warmup ends, the run is too short (that was the original bug).
function progress(simulation)
    u, v, w = simulation.model.velocities
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umean = %.3e, umax = (%.1e, %.1e, %.1e) m/s, wall: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   (mean(abs, u) + mean(abs, v)) / 2,
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
function build_and_run(arch, params::RunParams, output_path::AbstractString;
                       warmup, recording, output_interval)
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

    # Forcing: always the Gaussian brine relaxation on S; optionally a barotropic
    # M2 tide as a body force on u. Built as a NamedTuple so the tide-free case
    # is byte-identical to the pre-tide behaviour.
    S_forcing = Relaxation(rate = γ_S, target = S_target)
    forcing = params.tide_amp > 0 ?
        (S = S_forcing,
         u = Forcing(tide_forcing,
                     parameters = (F = params.tide_amp * M2_OMEGA, ω = M2_OMEGA))) :
        (S = S_forcing,)

    # Non-hydrostatic LES: AMD subgrid closure, buoyancy-active T, S passive via
    # BETA_S = 0 (advected and LES-mixed, but no feedback into momentum).
    model = NonhydrostaticModel(grid;
        advection           = WENO(),
        coriolis            = FPlane(f = f_coriolis),
        closure             = AnisotropicMinimumDissipation(),
        tracers             = (:T, :S),
        buoyancy            = SeawaterBuoyancy(equation_of_state =
                                LinearEquationOfState(thermal_expansion  = ALPHA_T,
                                                      haline_contraction = BETA_S)),
        boundary_conditions = (u = u_bcs, v = v_bcs),
        forcing             = forcing,
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
    # max_Δt RAISED 3.0 -> 10.0 s (2026-08-05). The 3 s cap was sized for the
    # buoyancy-active case, where ~1 m/s convective downdrafts against the 1.56 m
    # vertical cell make the CFL condition itself pick ~1.1 s and the cap never
    # binds. With S passive the velocities are ~100× smaller, so the cap became
    # the binding constraint and made the now-12× longer run needlessly
    # expensive. At the expected ~0.15 m/s over 7.8 m cells, Δt = 10 s is CFL
    # 0.19 — the wizard still governs, this only stops it capping too early.
    simulation = Simulation(model, Δt = 0.5, stop_time = warmup)
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))
    wizard = TimeStepWizard(cfl = 0.7, max_change = 1.1, max_Δt = 10.0)
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
        schedule           = TimeInterval(output_interval),
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
                        warmup, recording, output_interval, debug::Bool=false)
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
          "alpha_t": $(ALPHA_T),
          "beta_s": $(BETA_S),
          "tide_amplitude": $(params.tide_amp),
          "tide_period_seconds": $(M2_PERIOD),
          "max_dt_seconds": 10.0,
          "warmup_seconds": $(Float64(warmup)),
          "recording_seconds": $(Float64(recording)),
          "output_interval_seconds": $(Float64(output_interval)),
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
                          NOTE: the wind-driven flow needs ~1/f = 4.6 h to
                          equilibrate; anything under ~6 h gives a near-motionless
                          ocean (see the header).
  --recording-minutes M   recorded span per run (default $(DEFAULT_RECORDING_MINUTES))
  --output-interval-minutes M  snapshot spacing (default $(DEFAULT_OUTPUT_INTERVAL_MINUTES)).
                          Must be <= --recording-minutes or NO snapshots are written.
  --tide-amplitude U      M2 barotropic tidal current amplitude [m/s], 0 = off
                          (default 0). ~0.2 is representative of the southern
                          Gulf shelf. Off by default so the wind-only spin-up
                          baseline stays isolated.
  --debug                 2 min warmup + 5 min recording. PLUMBING SMOKE TEST
                          ONLY — far too short to judge spin-up or currents.
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
    output_interval_minutes = DEFAULT_OUTPUT_INTERVAL_MINUTES
    tide_amplitude    = 0.0
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
        elseif a == "--output-interval-minutes"
            output_interval_minutes = parse(Float64, args[i+1]); i += 2
        elseif a == "--tide-amplitude"
            tide_amplitude = parse(Float64, args[i+1]); i += 2
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
    tide_amplitude >= 0 || error("--tide-amplitude must be >= 0\n" * USAGE)
    output_interval_minutes > 0 ||
        error("--output-interval-minutes must be > 0\n" * USAGE)
    return (; n_runs, season, seed, start_index, output_dir,
              warmup_minutes, recording_minutes, output_interval_minutes,
              tide_amplitude, debug)
end

function main(cli)
    mkpath(cli.output_dir)
    warmup, recording, output_interval =
        cli.debug ? (DEBUG_DURATIONS[1], DEBUG_DURATIONS[2], DEBUG_OUTPUT_INTERVAL) :
                    (cli.warmup_minutes * 1minute, cli.recording_minutes * 1minute,
                     cli.output_interval_minutes * 1minute)
    # An output interval longer than the recording window writes an EMPTY file —
    # the writer's first scheduled tick never arrives. Fail loudly instead.
    recording >= output_interval || error(
        "--recording-minutes ($(recording/60)) must be >= --output-interval-minutes " *
        "($(output_interval/60)), otherwise no snapshots are written at all.")
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
        params = sample_params(cli.season, run_index, cli.seed, cli.tide_amplitude)
        @info @sprintf("Run %d/%d: wind %.2f m/s from %.0f°, tide %.2f m/s, β_S %.1e, %d sources, grid %d×%d×%d",
                       run_index, last_index, params.wind_speed, params.wind_dir_deg,
                       params.tide_amp, BETA_S, length(params.sources), NX, NY, NZ)
        build_and_run(ARCH, params, nc_path; warmup, recording, output_interval)
        write_metadata(meta_path, params, cli.seed;
                       warmup, recording, output_interval, debug = cli.debug)
        GC.gc(true)
        USE_GPU && CUDA.reclaim()
    end
    @info "All runs complete."
end

main(parse_cli(ARGS))
