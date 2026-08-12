"""
Oceananigans NON-HYDROSTATIC LES of a 1000×1000×100 m patch of Abu Dhabi
coastal waters (Arabian Gulf shelf), driven by NW-quadrant wind stress (the
seasonal shamal DIRECTION, at climatological-mean SPEED — not a storm) +
Coriolis, with stratified T carrying the buoyancy and a DYNAMICALLY PASSIVE
salinity field built from coastal pollution/brine plumes relaxed toward a
Gaussian target — tuned to realise a ~10 PSU salinity span for the RL homing
task.

GEOMETRY: STRAIGHT N-S COAST, ALONG-SHORE PERIODIC (changed 2026-08-06b).
Land occupies the west (x=0) border only. x is Bounded (coast at x=0, open
ocean at x=LX); y is the ALONG-SHORE axis and is Periodic. z is Bounded.

    History of this line, because it has now moved twice and the current
    setting is a considered compromise, not a default:
      * originally  (Periodic, Bounded, Bounded) — L-shaped coast, west+south
        land, x wrapping. Water flowed through the west coast and back in from
        the east, and half the sources sat on the x=0/x=LX seam where S_target
        jumped by the full 10 PSU anomaly at the same physical point.
      * then        (Bounded, Bounded, Bounded) — L-shaped coast, all walls.
        Fixed the seam and the through-flow, but a CLOSED basin with a rigid
        lid forces the domain-integrated horizontal transport to be identically
        zero: ∫∫u dy dz = 0 through every x-section, and likewise for v. No
        sustained current of ANY kind is possible, only closed overturning
        cells — downwind at the surface, return flow underneath. That is a
        model of an embayment, not of a 1 km patch of open shelf.
      * now         (Bounded, Periodic, Bounded) — straight coast, along-shore
        wrap. A mean along-shore current is possible again (this is the
        standard periodic-channel shelf configuration), the tide works (see
        below), and there is NO seam in S_target because the plume kernel uses
        the wrapped y-distance and every source sits at x=0, away from the only
        remaining boundary pair.

    Residual approximation: x=LX is open ocean but is modelled as a wall, so
    cross-shore transport reflects instead of leaving. A sponge layer over the
    last ~150 m would fix it; deliberately NOT added, to keep the configuration
    free of extra tuning knobs. Watch for pile-up against the east face,
    especially with the tide on.

    Do NOT compare current statistics across either topology change.

    NOTE for the reader side: the staggered-grid face counts have changed.
    Bounded x  -> `u` written with NX+1 faces (unchanged since 2026-08-06).
    Periodic y -> `v` written with NY faces, NOT NY+1 as in the all-bounded
    files. SwarmSwIM's FieldLoader already sniffs the `u` convention; it needs
    the same treatment for `v`. The sidecar JSON carries "topology" and
    "periodic_axes" so the reader can branch on those rather than on shapes.

    NOTE for anything that RECONSTRUCTS the salinity target from the sidecar
    sources (e.g. src/utils/sources.load_sources + a Python re-evaluation of the
    Gaussian sum): it MUST use the wrapped y-distance
    dy = min(|y-y0|, LY-|y-y0|) to match plume_excess below. A naive Euclidean
    distance disagrees with the simulated field by up to the full anomaly near
    the y=0/y=LY wrap.

SPIN-UP: READ THIS BEFORE BURNING GPU HOURS (updated 2026-08-06b).
The wind-driven flow adjusts on the Ekman timescale 1/f = 4.6 h (inertial
period 2π/f = 28.9 h). The original 30 min warmup + 60 min recording stopped the
runs at 0.05 inertial periods: measured currents reached only ~2.5% of their
value and were still rising ~2× per hour when recording ended. Warmup is now
12 h (≈2.6/f), which is also comfortably past 2·S_DECAY_TIME so the plumes are
saturated.

    *** THERE IS NO BOTTOM DRAG IN THIS CONFIGURATION, AND WITH y PERIODIC
    THE ALONG-SHORE FLOW THEREFORE DOES NOT EQUILIBRATE. ***
    Depth-integrating the along-shore momentum equation in a periodic channel
    with a cross-shore wall gives, in the absence of drag, dV/dt = τ_y/H with no
    sink: V ramps roughly LINEARLY forever. At τ = 3.9e-5 m²/s² (5 m/s wind) and
    H = 100 m that is 1.4 cm/s per hour of depth-mean acceleration, so 12 h of
    warmup gives ~1.7 cm/s depth-mean and nothing is converging. `umean` in the
    progress callback will ramp, not flatten — the "watch it flatten" diagnostic
    from the all-bounded version no longer has anything to detect.

    Fix, one line, strongly recommended before generating a production dataset
    (this is item (b) and is now effectively mandatory rather than optional):
        drag(x, y, t, u, v, Cd) = -Cd * u * sqrt(u^2 + v^2)   # and v-analogue
        u_bcs = FieldBoundaryConditions(
            top    = FluxBoundaryCondition(tau_x_top),
            bottom = FluxBoundaryCondition(drag, field_dependencies = (:u, :v),
                                           parameters = 2.5e-3))
    With C_d = 2.5e-3 the equilibrium depth-mean is √(τ/C_d) ≈ 0.125 m/s — right
    in the target band — but the approach timescale is H/(C_d·V) ≈ 89 h, so the
    12 h warmup still will not reach it.

    The arithmetic that constrains all of this: wind stress injects momentum at
    τ/H, so reaching a depth-mean V takes t = V·H/τ REGARDLESS of the drag law.
    V = 0.15 m/s needs ~107 h of spin-up. Wind forcing alone cannot produce
    tens-of-cm/s depth-mean flow on an affordable timescale. It CAN produce
    5–15 cm/s in the wind-mixed surface layer within 12 h, because the momentum
    is concentrated in the top ~10–20 m; the depth-mean stays ~1–2 cm/s.
    If you want realistic shelf currents cheaply, USE THE TIDE (--tide-amplitude
    0.2): it is imposed rather than spun up, so it is at full amplitude
    immediately, and it supplies the oscillating cross-shore shear that actually
    strains the tracer into filaments.

Recording is 6 h at 12 min intervals: long enough that consecutive snapshots
decorrelate (the advective timescale across a plume is σ_H/U), while keeping the
snapshot count and file size (~650 MB) where they were. The env draws a random
frozen snapshot per episode, so decorrelated frames are what buys
episode-to-episode variety.

DO NOT "FIX" WEAK CURRENTS BY MAKING SALINITY BUOYANT. Setting BETA_S = 7.8e-4
with the 10 PSU anomaly gives a reduced gravity g' = g·β_S·ΔS = 0.077 m/s², a
gravity-current velocity scale √(g'·σ_V) ≈ 1.75 m/s, and measured currents 164×
stronger than the passive case — faster than the vehicle over 12% of the domain.
It also HALVES the salinity span (9.9 → 4.8 PSU) and halves the vertical
gradient, because the convection mixes the column. That was the state of the
`buoyancy_active` dataset; keep S passive and use the tide instead.

WHY NON-HYDROSTATIC AT 1 km (vs. the hydrostatic 5 km script):
a 1 km box is smaller than the baroclinic deformation radius (Rd = N·H/f ≈ 2 km),
so the front-driven mesoscale-eddy mechanism the hydrostatic script relies on
CANNOT operate here. Instead the spatial structure in this domain comes from
(a) the localized brine plumes themselves and (b) wind- and tide-driven
boundary-layer turbulence stirring them into filaments — both of which need the
full 3-D pressure solve. At 1000×1000×100 m with ~8 m horizontal / ~1.6 m
vertical cells the aspect ratio (~5) is mild enough for a non-hydrostatic LES to
stay stable, unlike the 39×0.85 m hydrostatic cells. Use THIS script for the
1 km × 100 m turbulence-resolving training domain.

MULTI-RUN DATASET GENERATION (mirrors hydrostatic.jl):
runs N independent simulations in one Julia session (amortizing compile time),
each with per-run randomized — but Abu-Dhabi-plausible — parameters: sources
(number, along-shore position, depth, Q), wind speed/direction around the
seasonal shamal, and a seeded IC temperature perturbation so identical
parameters still diverge.

    julia --project=oceananigans oceananigans/non_hydrostatic.jl \\
        --n-runs 4 --season winter --seed 1337 --tide-amplitude 0.2

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
# The coastline is STRAIGHT and runs north-south: LAND on the west (x=0) border,
# open ocean to the east. x is therefore the CROSS-SHORE axis (Bounded) and y is
# the ALONG-SHORE axis (Periodic). See the header for why.
const LX    = 1000.0   # cross-shore extent [m]; x=0 is the coast, x=LX offshore
const LY    = 1000.0   # along-shore extent [m]; PERIODIC — y=0 and y=LY are the
                       # same physical location
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
const BETA_S  = 6.0e-5     # haline contraction [1/PSU]; 0 = passive S 

# M2 principal lunar semidiurnal tide, 12.4206 h. The Arabian Gulf is tidally
# energetic and the tide is the cheapest route to realistic current speeds (see
# the spin-up section of the header), but it is still OFF by default
# (--tide-amplitude 0) so the wind-only baseline stays isolated and
# reproducible. Turn it on once that baseline is validated.
const M2_PERIOD = 12.4206 * 3600      # [s]
const M2_OMEGA  = 2π / M2_PERIOD      # [rad/s]

# Barotropic tidal body force. dv/dt = F·cos(ωt) integrates to v = (F/ω)·sin(ωt),
# so F = U_tide·ω sets the tidal current AMPLITUDE directly. Coriolis turns the
# rectilinear forcing into the observed rotary ellipse (f/ω = 0.43 here, so the
# ellipse is distinctly non-degenerate).
#
# *** FORCED ON v, NOT u (changed 2026-08-06b). *** A spatially uniform body
# force on the CROSS-SHORE component would be almost exactly cancelled by the
# pressure gradient that develops between the two x-walls: the tide would be
# silently inert and the run would still complete cleanly. It must be applied
# along the PERIODIC axis, which is also where shelf tidal currents actually run
# (the coast blocks cross-shore tidal flow). This is why the topology change and
# the tide are coupled — under the previous all-bounded topology no axis could
# carry it.
#
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
# used σ_H = 80 m (~8 % of the box) and left only ~6 % of cells above baseline.
# Widened to 300 m, then to 400 m on 2026-08-06b: with the L-shaped coast the
# domain was covered from TWO walls, and dropping the south wall halved the
# coverage. At σ_H = 300 the offshore corner sat at 0.4 % of peak (0.04 PSU
# above baseline, gradient ~4e-4 PSU/m — under any realistic sensor noise
# floor); at σ_H = 400 it is 4.4 % of peak with a ~2.7e-3 PSU/m gradient. RAISE
# THIS FURTHER if the offshore third still reads as flat. σ_V spans the upper
# column.
#
# The relaxation brings the field to S_SOURCE_ANOMALY·(normalised plume) on the
# timescale S_DECAY_TIME as (1 − e^{−t/τ}). The plume sum is normalised by its
# domain max (see build_and_run) so the realised core peak is EXACTLY
# S_SOURCE_ANOMALY above baseline regardless of how many wide sources overlap —
# span is set solely by the anomaly and stays put when σ_H or the source count
# change. So span ≈ 10 PSU here.
const SIGMA_H          = 400.0     # plume horizontal std [m]
const SIGMA_V          = 40.0      # plume vertical std [m]
const S_SOURCE_ANOMALY = 10.0      # core salinity excess above baseline [PSU] = the span
# Relaxation timescale. RAISED 15 min -> 1 h (2026-08-05): the relaxation and
# advection compete to set the field, and at 15 min the forcing won by ~80×, so
# the tracer stayed pinned to the smooth analytic target no matter how well the
# flow was stirred.
#
# The competition is γ_S⁻¹ vs. σ_H/U. At the wind-only depth-mean speed
# (~0.02 m/s) σ_H/U ≈ 20,000 s and the 1 h relaxation still wins by ~6×: expect
# a field close to the smooth analytic target. With the tide on at 0.2 m/s,
# σ_H/U ≈ 2000 s and stirring wins by ~2×, which is the regime where filaments
# can actually form. THIS IS THE KNOB TO CHECK FIRST if the smoke test comes out
# too smooth (raise it) or if the span has collapsed below ~8 PSU (lower it).
#
# Structural caveat: relaxation toward a fixed Eulerian target is a low-pass
# filter — it erases advected structure everywhere on the γ_S⁻¹ timescale, so a
# filament survives only ~U·γ_S⁻¹ of downstream extent. If filaments remain
# elusive after turning the tide on, the real fix is to make γ_S spatially
# varying (full strength within ~1σ_H of the source cores, zero in the
# interior), which keeps the span guarantee while freeing the far field.
const S_DECAY_TIME     = 1hour
const γ_S              = 1 / S_DECAY_TIME

# Q-weighted sum of source Gaussians (the un-normalised plume shape). Top-level
# (not a closure) so it stays type-stable / GPU-capturable when called from the
# S_target closure; srcs is passed as an isbits Tuple.
#
# y is PERIODIC, so the along-shore separation is the MINIMUM-IMAGE distance,
# min(|Δy|, LY-|Δy|). Without this the target field is discontinuous across the
# y=0/y=LY seam — the same physical point would be assigned two different
# salinities, which is exactly the pathology the 2026-08-06 change removed from
# the x axis. Any code outside this file that rebuilds the target from the
# sidecar sources must apply the same wrap.
@inline function plume_excess(x, y, z, srcs, qmax)
    e = 0.0
    for src in srcs
        dy_raw = abs(y - src.y0)
        dy     = min(dy_raw, LY - dy_raw)          # minimum image in periodic y
        e += (src.Q / qmax) * exp(-((x - src.x0)^2 + dy^2) / (2 * SIGMA_H^2)
                                  - (z - src.z0)^2 / (2 * SIGMA_V^2))
    end
    return e
end

# Durations. See the spin-up section of the header — 12 h ≈ 2.6 Ekman timescales
# and past 2·S_DECAY_TIME, but with no bottom drag the along-shore flow is still
# ramping when recording starts. Recording 6 h at 12 min intervals so
# consecutive snapshots decorrelate.
const DEFAULT_WARMUP_MINUTES          = 240.0
const DEFAULT_RECORDING_MINUTES       = 240.0
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

    # Wind around the seasonal shamal (NW = 315°). With the coast at x=0 this
    # always has an offshore/onshore component, so expect coastal Ekman
    # convergence or divergence at the west wall depending on the sampled
    # direction, plus an along-shore component that the periodic y axis can now
    # actually sustain.
    wind_speed   = unif(rng, sp.wind_speed - 2.5, sp.wind_speed + 2.5)
    wind_dir_deg = unif(rng, 275.0, 355.0)
    tau_mag      = RHO_AIR * C_DRAG * wind_speed^2 / RHO_WATER

    # Sources: ALL on the x=0 west coast, at random along-shore positions
    # (2026-08-06b — previously split 50/50 between a west and a south wall,
    # matching the old L-shaped geometry). y is periodic so y0 may sit anywhere
    # in [0, LY) without creating a seam; plume_excess wraps.
    #
    # NOTE: the per-source RNG draw sequence changed (the coin flip that chose a
    # wall is gone), so a given (seed, run_index) samples DIFFERENT sources than
    # it did before this date. Runs are not reproducible across the change; the
    # topology change already made them incomparable anyway.
    #
    # Depths shallow (coastal outfalls), so structure lives in the upper column.
    # The count is high so the wide (σ_H=400) plumes overlap into along-shore
    # coverage.
    n_sources = rand(rng, 10:30)
    sources = SourceSpec[]
    for _ in 1:n_sources
        Q     = unif(rng, 2.0, 10.0)
        depth = unif(rng, 0.0, 0.7*DEPTH)
        x0    = 0.0                      # the coast
        y0    = unif(rng, 0.0, LY)       # anywhere along it
        push!(sources, (Q = Q, x0 = x0, y0 = y0, z0 = -depth))
    end

    t_noise_amp = unif(rng, 0.01, 0.02)

    return RunParams(season, run_index, run_seed, wind_speed, wind_dir_deg,
                     tau_mag, t_noise_amp, tide_amp, sources)
end

# ─────────────────────────────────────────────────────────────────────────────
# Progress callback
# ─────────────────────────────────────────────────────────────────────────────
# `umean` was the spin-up diagnostic under the all-bounded topology, where the
# flow equilibrated and the number was expected to flatten. With y periodic and
# no bottom drag it RAMPS instead (see the header), and with the tide on it
# OSCILLATES at 12.42 h on top of that ramp. Read it as "is the flow the
# magnitude I expect", not as a convergence test.
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
    # component in the +axis direction. Sanity check with the NW shamal (315°):
    # the (θ-270) offset is 45°, giving tau_x_top = -0.707τ (drives +x, offshore)
    # and tau_y_top = +0.707τ (drives -y, along-shore southward) — a wind out of
    # the northwest pushing surface water to the southeast. Correct.
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
    # (σ_H, source count). Without it, N overlapping σ_H=400 plumes drive the sum
    # past 1 and the span balloons. The global max of a sum of Gaussians sits at a
    # source core, so scan the centres plus a coarse grid. (With every source on
    # x=0 the max is guaranteed to be on the x=0 face, which the source scan
    # covers exactly; the grid scan is belt-and-braces.)
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

    # Grid: BOUNDED CROSS-SHORE (x), PERIODIC ALONG-SHORE (y), BOUNDED (z).
    # Changed 2026-08-06b from all-Bounded; see the header for the full history
    # and for why the all-Bounded version could not sustain any mean current.
    #
    # Land is the west wall (x=0). x=LX is open ocean approximated as a wall —
    # a sponge over the last ~150 m would fix it, deliberately omitted to avoid
    # another tuning knob. Watch for pile-up against the east face.
    #
    # Numerically free: the FFT pressure solver uses a real FFT in the periodic
    # direction and a cosine transform in the bounded ones (same
    # FFTBasedPoissonSolver, no slowdown).
    #
    # NOTE for the reader side: bounded x makes Oceananigans write `u` with
    # NX+1 faces; PERIODIC y makes it write `v` with NY faces (the all-bounded
    # files had NY+1). SwarmSwIM's FieldLoader must branch on the sidecar's
    # "topology"/"periodic_axes" fields, not just sniff `u`.
    #
    # ~7.8 m × 1.6 m cells at the default 128×128×64.
    grid = RectilinearGrid(arch;
        size     = (NX, NY, NZ),
        x        = (0, LX),
        y        = (0, LY),
        z        = (-DEPTH, 0),              # z negative downward, surface at 0
        topology = (Bounded, Periodic, Bounded),
    )

    # Surface wind stress only. There is NO bottom drag — see the header; this
    # is the single most consequential omission now that y is periodic, because
    # it leaves the along-shore flow with no momentum sink.
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_x_top))
    v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_y_top))

    # Forcing: always the Gaussian brine relaxation on S; optionally a barotropic
    # M2 tide as a body force on v (the ALONG-SHORE, periodic component — see the
    # comment at tide_forcing for why it cannot go on u). Built as a NamedTuple
    # so the tide-free case is byte-identical to the pre-tide behaviour.
    S_forcing = Relaxation(rate = γ_S, target = S_target)
    forcing = params.tide_amp > 0 ?
        (S = S_forcing,
         v = Forcing(tide_forcing,
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

    # Small random kicks on u/v/w concentrated in the upper few metres to seed
    # boundary-layer turbulence; T stratification + S baseline. v is now
    # perturbed too (it was not before): it is the free, periodic direction and
    # the one the along-shore current develops in.
    #
    # KNOWN GAP: these use the global RNG, not params.run_seed, so the kicks are
    # not reproducible run-to-run. Seeding them would require capturing an RNG
    # inside a function passed to set!, which breaks isbits/GPU capture. The
    # seeded T noise below is what makes identical parameters diverge
    # deliberately; this is unintended non-determinism on top of it.
    u★ = sqrt(params.tau_mag)
    Ξ(z) = randn() * exp(z / 4)
    uᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
    vᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
    wᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
    set!(model, u = uᵢ, v = vᵢ, w = wᵢ, T = T_init, S = S_init)

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
    # expensive. At 0.2 m/s (tide on) over 7.8 m cells, Δt = 10 s is CFL 0.26 —
    # the wizard still governs, this only stops it capping too early.
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
#
# "topology" / "periodic_axes" / "domain" are what the reader needs to (a) pick
# the right staggered-face convention for u and v and (b) apply the minimum-image
# wrap in y when rebuilding the salinity target from these sources.
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
          "topology": "bounded_periodic_bounded",
          "periodic_axes": ["y"],
          "land_borders": ["west"],
          "open_borders_modelled_as_walls": ["east"],
          "u_faces": $(NX + 1),
          "v_faces": $(NY),
          "w_faces": $(NZ + 1),
          "plume_distance_metric": "minimum_image_in_y",
          "wind_speed": $(params.wind_speed),
          "wind_dir_deg": $(params.wind_dir_deg),
          "tau_mag": $(params.tau_mag),
          "bottom_drag": null,
          "t_noise_amp": $(params.t_noise_amp),
          "alpha_t": $(ALPHA_T),
          "beta_s": $(BETA_S),
          "tide_amplitude": $(params.tide_amp),
          "tide_forced_component": "v",
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
                          NOTE: with y periodic and NO bottom drag the along-shore
                          flow ramps rather than equilibrating — longer warmup
                          means a faster current, without limit. 12 h gives
                          ~1.7 cm/s depth-mean at a 5 m/s wind. See the header.
  --recording-minutes M   recorded span per run (default $(DEFAULT_RECORDING_MINUTES))
  --output-interval-minutes M  snapshot spacing (default $(DEFAULT_OUTPUT_INTERVAL_MINUTES)).
                          Must be <= --recording-minutes or NO snapshots are written.
  --tide-amplitude U      M2 barotropic tidal current amplitude [m/s], 0 = off
                          (default 0). ~0.2 is representative of the southern
                          Gulf shelf and is the CHEAPEST route to realistic
                          current speeds, since it is imposed rather than spun
                          up. Applied to the along-shore (v) component; it would
                          be cancelled by the pressure gradient on the bounded x
                          axis. Off by default so the wind-only baseline stays
                          isolated.
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
    cli.tide_amplitude == 0 && @warn(
        "Tide is OFF. With wind forcing alone the depth-mean current is set by " *
        "τ·t/H ≈ 1.4 cm/s per hour of warmup and the salinity field will stay " *
        "close to the smooth analytic target. Consider --tide-amplitude 0.2.")
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
        @info @sprintf("Run %d/%d: wind %.2f m/s from %.0f°, tide %.2f m/s (on v), β_S %.1e, %d sources on x=0, grid %d×%d×%d",
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