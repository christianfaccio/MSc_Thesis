"""
Oceananigans HYDROSTATIC free-surface model of a 5×5 km × 40 m patch of Abu
Dhabi coastal waters (Arabian Gulf shelf), driven by NW shamal wind stress +
Coriolis, with a cross-shore thermal FRONT (the source of spatial structure),
stratified T, buoyancy-active S ≈ 40 PSU, and four pollution sources mirrored
from config/sources.json.

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

Output → data/oceananigans/hydrostatic_<season>.nc
        (u, v, w, T, S snapshots every 15 min of simulated time).

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

Random.seed!(1337)

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
# Abu Dhabi season presets (same physics baselines as non_hydrostatic.jl)
# tau_mag = |τ| from wind_speed via τ = ρ_air · C_d · U² / ρ_water
# with ρ_air = 1.225, C_d = 1.3e-3, ρ_water = 1027.
# ─────────────────────────────────────────────────────────────────────────────
const SEASON = :winter   # :winter | :summer

const SEASON_PARAMS = Dict(
    :winter => (T_surface = 22.0, T_bottom = 20.0, S_baseline = 40.0,
                N2 = 1.0e-5, wind_speed = 5.0, tau_mag = 3.87e-5),
    :summer => (T_surface = 33.0, T_bottom = 22.0, S_baseline = 42.0,
                N2 = 1.0e-4, wind_speed = 8.0, tau_mag = 1.05e-4),
)
const p = SEASON_PARAMS[SEASON]

# Abu Dhabi reference latitude 24.5°N → f ≈ 6.06e-5 s⁻¹.
const LATITUDE_DEG = 24.5
const Ω_EARTH      = 7.292e-5
const f_coriolis   = 2 * Ω_EARTH * sind(LATITUDE_DEG)

# Linear equation of state coefficients (set explicitly so the thermal-wind jet
# below uses the SAME α the model does).
const G_GRAV  = 9.81
const ALPHA_T = 1.67e-4    # thermal expansion  [1/°C]
const BETA_S  = 7.8e-4     # haline contraction [1/PSU]

# NW shamal wind (315° = "from NW", blowing toward SE). A negative kinematic
# top-flux drives the matching velocity component in the +axis direction.
const WIND_DIR_DEG = 315.0
const tau_x_top    = -p.tau_mag * cosd(WIND_DIR_DEG - 270)   # negative → +x flow
const tau_y_top    = +p.tau_mag * sind(WIND_DIR_DEG - 270)   # positive → -y flow

# ─────────────────────────────────────────────────────────────────────────────
# Cross-shore temperature FRONT (breaks the horizontal symmetry).
# Warm inshore (low y) → cooler offshore over ≈2·FRONT_WIDTH at y = FRONT_Y, on
# top of the vertical stratification. The accompanying thermal-wind jet keeps the
# front balanced (no slump); it then meanders and sheds eddies. Stability levers:
#   Rd = N·H/f ≈ 2 km < 5 km        ⇒ eddies fit
#   Ri = N²/(∂u/∂z)² ≈ 1.3 > 0.25   ⇒ no Kelvin-Helmholtz blow-up
# Stronger front (↑DELTA_T_FRONT or ↓FRONT_WIDTH) ⇒ more vigorous eddies, lower
# Ri — keep Ri > 0.25.
# ─────────────────────────────────────────────────────────────────────────────
const FRONT_Y       = 2500.0   # front centre [m]
const FRONT_WIDTH   = 500.0    # tanh half-width [m] (front spans ≈ 1 km)
const DELTA_T_FRONT = 0.3      # cross-front temperature contrast [°C]

@inline T_front(y)    = (DELTA_T_FRONT / 2) * tanh((y - FRONT_Y) / FRONT_WIDTH)
@inline dTdy_front(y) = (DELTA_T_FRONT / (2 * FRONT_WIDTH)) * sech((y - FRONT_Y) / FRONT_WIDTH)^2

# ─────────────────────────────────────────────────────────────────────────────
# Pollution sources — positions mirror config/sources.json: four coastal outfalls
# on the y=0 shore (the only true coast — x is periodic, so x=0 is NOT a boundary),
# spread across the 5 km along-shore extent at staggered depths (3–12 m) so the
# salinity field carries genuine 3-D structure. The brine is then carried offshore
# (across the front) by the meandering jet and eddies over the multi-day run, so it
# need not be pre-placed in the interior. This matches src/utils/sources.py, which
# anchors randomized training sources to the domain borders.
# Salinity stays buoyancy-active, so the brine must be REALISTIC:
# a desalination outfall raises local salinity by a few PSU, not the hundreds an
# unbounded continuous source would accumulate. Each source is a Gaussian brine
# injection balanced by a linear relaxation back to baseline (timescale
# S_DECAY_TIME); the two balance at a steady core excess of S_SOURCE_ANOMALY PSU
# at the strongest source. Bounded brine ⇒ gentle sinking ⇒ no vertical-CFL blow-up.
#   ↑ S_SOURCE_ANOMALY  = more realistic / stronger plume, but slower (more sinking)
#   ↓ S_SOURCE_ANOMALY  = faster run.  Watch umax_w & Δt in the log and tune.
# ─────────────────────────────────────────────────────────────────────────────
const SIGMA_H = 150.0    # plume horizontal std [m]
const SIGMA_V = 12.0     # plume vertical std [m]

const sources = [
    (Q = 3.0, x0 = 1000.0, y0 = 4000.0, z0 =  0.0),
    (Q = 8.0, x0 = 0.0,    y0 = 1800.0, z0 = -2.0),
    (Q = 6.0, x0 = 1000.0, y0 = 0.0,    z0 = -8.0),
    (Q = 4.0, x0 = 3200.0, y0 = 0.0,    z0 =-12.0),
    (Q = 5.0, x0 = 5000.0, y0 = 1500.0, z0 = -6.0)
]

const S_SOURCE_ANOMALY = 10.0          # steady-state core salinity excess [PSU]
const S_DECAY_TIME     = 6hours        # relaxation timescale back to baseline
const γ_S              = 1 / S_DECAY_TIME
const Q_MAX            = maximum(src.Q for src in sources)

# Continuous injection balanced by a γ_S sink is exactly RELAXATION toward
# `S_baseline + S_SOURCE_ANOMALY·(normalised plume)` at rate γ_S. We give that
# target to Oceananigans' built-in `Relaxation`, which is type-stable — a
# hand-rolled field-dependent `Forcing` boxed per cell and GC-thrashed the step
# (Δt pinned, run stalled). The plume is the Q-weighted sum of source Gaussians,
# normalised so the strongest source's core reaches S_SOURCE_ANOMALY.
@inline function S_target(x, y, z, t)
    excess = 0.0
    for src in sources
        excess += (src.Q / Q_MAX) * exp(-((x-src.x0)^2 + (y-src.y0)^2) / (2*SIGMA_H^2)
                                        - (z-src.z0)^2 / (2*SIGMA_V^2))
    end
    return p.S_baseline + S_SOURCE_ANOMALY * excess
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
# Grid: 5 km × 5 km × 40 m at ≈39 m horizontal / ≈0.85 m vertical resolution.
# Periodic along-shore (x), Bounded cross-shore (y, coast at y=0) so the front can
# live across y; Bounded in z. Hydrostatic dynamics tolerate the large aspect ratio.
# ─────────────────────────────────────────────────────────────────────────────
grid = RectilinearGrid(ARCH;
    size     = (128, 128, 48),
    x        = (0, LX),
    y        = (0, LY),
    z        = (-DEPTH, 0),              # z negative downward, surface at 0
    topology = (Periodic, Bounded, Bounded),
)

# ─────────────────────────────────────────────────────────────────────────────
# Boundary conditions: NW shamal wind stress on (u, v); no buoyancy fluxes.
# ─────────────────────────────────────────────────────────────────────────────
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_x_top))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_y_top))

# ─────────────────────────────────────────────────────────────────────────────
# Model: hydrostatic free-surface with buoyancy-active T and S.
#
# buoyancy: S stays coupled to density (denser brine sinks). The bounded
#   source+sink above keeps the excess to a few PSU so this convection is gentle.
# closure: (1) vertical eddy diffusivity with VerticallyImplicitTimeDiscretization
#   to remove the hidden explicit-diffusion step limit (Δt < dz²/2ν ≈ 35 s);
#   (2) horizontal biharmonic diffusivity ν₄ = 200 m⁴/s to scrub the 2Δx grid
#   noise the front/eddies generate (explicit, stable while Δt·ν₄ ≲ 4e4).
# free_surface: ImplicitFreeSurface avoids the split-explicit barotropic
#   instability (InexactError: Int64(NaN) from the substep-count calc).
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Initial conditions: vertical T stratification + cross-shore front, uniform S
# baseline, and a thermal-wind-balanced along-shore jet U_geo so the front does
# not slump. ∂u/∂z = -(gα/f) ∂T/∂y, integrated from rest at the bed (z = -H).
# ─────────────────────────────────────────────────────────────────────────────
T_init(x, y, z) = p.T_bottom + (p.T_surface - p.T_bottom) * (1 + z / DEPTH) + T_front(y)
S_init(x, y, z) = p.S_baseline
U_geo(x, y, z)  = -(G_GRAV * ALPHA_T / f_coriolis) * dTdy_front(y) * (z + DEPTH)

set!(model, T = T_init, S = S_init, u = U_geo)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation + adaptive Δt. Coarse hydrostatic grid permits large steps; the
# explicit biharmonic term caps max_Δt (Δt·ν₄ ≲ 4e4 ⇒ Δt ≲ 200 s at ν₄=200, so
# 60 s leaves a 3× margin).
# ─────────────────────────────────────────────────────────────────────────────
const WARMUP_TIME = 1day        # let the front spin up & start meandering
const RECORDING_TIME = 3days
simulation = Simulation(model, Δt = 5.0, stop_time = WARMUP_TIME)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))

wizard = TimeStepWizard(cfl = 0.7, max_change = 1.1, max_Δt = 60.0)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

@info "Warmup phase ($(SEASON), $(ARCH)): spinning up for $(prettytime(WARMUP_TIME)) with no output..."
run!(simulation)
@info "Warmup complete at t = $(prettytime(time(simulation))); attaching output writer."

# ─────────────────────────────────────────────────────────────────────────────
# NetCDF output → data/oceananigans/hydrostatic_<season>.nc
# ─────────────────────────────────────────────────────────────────────────────
output_dir  = joinpath(@__DIR__, "..", "data", "oceananigans")
mkpath(output_dir)
output_path = joinpath(output_dir, "hydrostatic_$(SEASON).nc")

simulation.output_writers[:fields] = NetCDFWriter(
    model,
    (u = model.velocities.u,
     v = model.velocities.v,
     w = model.velocities.w,
     T = model.tracers.T,
     S = model.tracers.S),
    filename           = output_path,
    schedule           = TimeInterval(15minutes),
    overwrite_existing = true,
)

simulation.stop_time = WARMUP_TIME + RECORDING_TIME

@info "Starting Oceananigans hydrostatic run ($(SEASON), $(ARCH)) → $output_path"
run!(simulation)
@info "Done."
