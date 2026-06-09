"""
Oceananigans HYDROSTATIC free-surface model of a 5×5 km × 40 m patch of Abu
Dhabi coastal waters (Arabian Gulf shelf), driven by NW shamal wind stress +
Coriolis, with stratified T, baseline S ≈ 40 PSU, and four pollution sources
mirrored from config/sources_5km.json (Q rescaled to a PSU/s emission rate via
Q_SCALE).

WHY HYDROSTATIC (vs. the non-hydrostatic LES in abu_dhabi_les.jl):
at km horizontal scale over tens of metres depth the flow is hydrostatic
(horizontal ≫ vertical scales), so a HydrostaticFreeSurfaceModel is both the
physically correct regime and far cheaper — it skips the 3-D pressure Poisson
solve and tolerates the large-aspect-ratio cells (≈39 m horizontal / ≈0.85 m
vertical) that would wreck a non-hydrostatic LES. Use this script for the
1-10 km training domain; keep abu_dhabi_les.jl for the ~100 m turbulence-
resolving patch.

Output → data/oceananigans/coastal_5km_<season>.nc
        (u, v, w, T, S snapshots every 10 min of simulated time).

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
# Abu Dhabi season presets (same physics baselines as abu_dhabi_les.jl)
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

# NW shamal wind (315° = "from NW", blowing toward SE). A negative kinematic
# top-flux drives the matching velocity component in the +axis direction.
const WIND_DIR_DEG = 315.0
const tau_x_top    = -p.tau_mag * cosd(WIND_DIR_DEG - 270)   # negative → +x flow
const tau_y_top    = +p.tau_mag * sind(WIND_DIR_DEG - 270)   # positive → -y flow

# ─────────────────────────────────────────────────────────────────────────────
# Pollution sources — km positions mirroring config/sources_5km.json.
# σh/σv are the km-scale plume widths (≫ the 15 m used on the 100 m LES domain).
# Q_SCALE converts the synthetic-model "peak amplitude" Q into a PSU/s rate.
# Keep in sync with config/sources_5km.json.
# ─────────────────────────────────────────────────────────────────────────────
const Q_SCALE = 1.0e-3   # PSU/s per unit Q
const SIGMA_H = 100.0    # plume horizontal std [m]
const SIGMA_V = 12.0     # plume vertical std [m]

const sources = [
    (Q = 8.0, x0 = 224.0, y0 =    0.0, z0 = -2.0),
    (Q = 6.0, x0 = 620.0, y0 =    0.0, z0 = -2.0),
    (Q = 4.0, x0 =   0.0, y0 =  114.0, z0 = -2.0),
    (Q = 5.0, x0 =   0.0, y0 =  452.0, z0 = -2.0),
]

@inline function salinity_source(x, y, z, t)
    s = 0.0
    for src in sources
        s += Q_SCALE * src.Q * exp(-((x-src.x0)^2 + (y-src.y0)^2) / (2*SIGMA_H^2)
                                   - (z-src.z0)^2 / (2*SIGMA_V^2))
    end
    return s
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
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Grid: 5 km × 5 km × 40 m at ≈39 m horizontal / ≈0.85 m vertical resolution.
# Hydrostatic dynamics tolerate this large aspect ratio (unlike LES).
# ─────────────────────────────────────────────────────────────────────────────
grid = RectilinearGrid(ARCH;
    size     = (128, 128, 48),
    x        = (0, 1000),
    y        = (0, 1000),
    z        = (-40, 0),                # z negative downward, surface at 0
    topology = (Periodic, Periodic, Bounded),
)

# ─────────────────────────────────────────────────────────────────────────────
# Boundary conditions: NW shamal wind stress on (u, v); no buoyancy fluxes.
# ─────────────────────────────────────────────────────────────────────────────
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_x_top))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_y_top))

# ─────────────────────────────────────────────────────────────────────────────
# Model: hydrostatic free-surface with T (buoyancy-active) and S (source-forced).
#
# closure: a modest vertical eddy diffusivity for mixing; WENO advection supplies
#   the horizontal dissipation. For a more physical wind-driven mixed layer swap
#   in `closure = CATKEVerticalDiffusivity()` (adds a prognostic TKE tracer :e).
# free_surface: split-explicit is GPU-friendly. If the installed Oceananigans
#   version rejects the `cfl` kwarg, use `SplitExplicitFreeSurface(substeps = 48)`
#   or the simpler `ImplicitFreeSurface()`.
# ─────────────────────────────────────────────────────────────────────────────
model = HydrostaticFreeSurfaceModel(grid;
    momentum_advection  = WENO(),
    tracer_advection    = WENO(),
    coriolis            = FPlane(f = f_coriolis),
    buoyancy            = SeawaterBuoyancy(equation_of_state = LinearEquationOfState()),
    tracers             = (:T, :S),
    closure             = VerticalScalarDiffusivity(ν = 1e-2, κ = 1e-3),
    free_surface        = SplitExplicitFreeSurface(grid; cfl = 0.7),
    boundary_conditions = (u = u_bcs, v = v_bcs),
    forcing             = (S = Forcing(salinity_source),),
)

# ─────────────────────────────────────────────────────────────────────────────
# Initial conditions: linear T stratification (surface warmer), uniform S
# baseline, starting from rest. Hydrostatic flow needs no turbulent kick — the
# wind stress spins up the circulation.
# ─────────────────────────────────────────────────────────────────────────────
T_init(x, y, z) = p.T_bottom + (p.T_surface - p.T_bottom) * (1 + z / grid.Lz)
S_init(x, y, z) = p.S_baseline

set!(model, T = T_init, S = S_init)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation + adaptive Δt. Coarse hydrostatic grid permits large steps.
# ─────────────────────────────────────────────────────────────────────────────
const WARMUP_TIME = 12hours
const RECORDING_TIME = 3days
simulation = Simulation(model, Δt = 5.0, stop_time = WARMUP_TIME)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))

wizard = TimeStepWizard(cfl = 0.7, max_change = 1.1, max_Δt = 60.0)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

@info "Warmup phase ($(SEASON), $(ARCH)): spinning up for $(prettytime(WARMUP_TIME)) with no output..."
run!(simulation)
@info "Warmup complete at t = $(prettytime(time(simulation))); ataching output writer."

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
