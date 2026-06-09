"""
Oceananigans non-hydrostatic LES of a 100×100×30 m patch of Abu Dhabi
coastal waters (Arabian Gulf shelf depth), driven by NW shamal wind stress
+ Coriolis, with stratified T, baseline S ≈ 40 PSU, and four pollution
sources mirrored from config/sources.json (with Q rescaled from peak-PSU
to PSU/s emission rate via Q_SCALE).

Output → data/oceananigans/scenario_01_<season>.nc
        (u, v, w, T, S snapshots every minute of simulated time).

Edit SEASON below to switch between winter (well-mixed, weaker stratification)
and summer (stratified, stronger winds) presets.
"""

using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf
using Random

Random.seed!(1337)

# ─────────────────────────────────────────────────────────────────────────────
# Abu Dhabi season presets
# ─────────────────────────────────────────────────────────────────────────────
const SEASON = :winter   # :winter | :summer

# tau_mag = |τ| derived from wind_speed via τ = ρ_air · C_d · U² / ρ_water
# with ρ_air = 1.225, C_d = 1.3e-3, ρ_water = 1027.
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

# NW shamal wind. WIND_DIR_DEG is the meteorological "wind from" direction
# (315° = NW, blowing toward SE → stress on water in (+x, -y) coords).
# Oceananigans top-flux sign convention: a negative kinematic top-flux drives
# the corresponding velocity component in the +axis direction.
const WIND_DIR_DEG = 315.0
const tau_x_top    = -p.tau_mag * cosd(WIND_DIR_DEG - 270)   # negative → +x flow
const tau_y_top    = +p.tau_mag * sind(WIND_DIR_DEG - 270)   # positive → -y flow

# ─────────────────────────────────────────────────────────────────────────────
# Pollution sources (mirrors config/sources.json — keep Q values in sync).
# Q_SCALE converts the synthetic-model "peak amplitude" interpretation of Q
# (used in src/models/salinity.py) into an LES emission *rate* in PSU/s.
# Order-of-magnitude estimate for a desal outfall driving a ~5–10 PSU plume
# above ambient is ~1e-3 PSU/s.
# ─────────────────────────────────────────────────────────────────────────────
const Q_SCALE = 1.0e-3   # PSU/s per unit Q

const sources = [
    (Q = 8.0, x0 = 22.4, y0 =  0.0, z0 = -2.0, σh = 15.0, σv = 10.0),
    (Q = 6.0, x0 = 75.7, y0 =  0.0, z0 = -2.0, σh = 15.0, σv = 10.0),
    (Q = 4.0, x0 =  0.0, y0 = 11.4, z0 = -2.0, σh = 15.0, σv = 10.0),
    (Q = 5.0, x0 =  0.0, y0 = 45.2, z0 = -2.0, σh = 15.0, σv = 10.0),
]

@inline function salinity_source(x, y, z, t)
    s = 0.0
    for src in sources
        s += Q_SCALE * src.Q * exp(-((x-src.x0)^2 + (y-src.y0)^2) / (2*src.σh^2)
                                   - (z-src.z0)^2 / (2*src.σv^2))
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
# Grid: 100×100×30 m at ~0.78 m horiz / ~0.625 m vert resolution.
# 30 m depth matches Abu Dhabi coastal bathymetry (Arabian Gulf avg ~35 m,
# coastal 15–30 m).
# ─────────────────────────────────────────────────────────────────────────────
grid = RectilinearGrid(
    size     = (128, 128, 48),
    x        = (0, 100),
    y        = (0, 100),
    z        = (-30, 0),                # z negative downward, surface at 0
    topology = (Periodic, Periodic, Bounded),
)

# ─────────────────────────────────────────────────────────────────────────────
# Boundary conditions: NW shamal wind stress on (u, v); no buoyancy fluxes.
# ─────────────────────────────────────────────────────────────────────────────
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_x_top))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_y_top))

# ─────────────────────────────────────────────────────────────────────────────
# Model: non-hydrostatic LES with T (buoyancy-active) and S (with source forcing)
# ─────────────────────────────────────────────────────────────────────────────
model = NonhydrostaticModel(grid;
    coriolis            = FPlane(f = f_coriolis),
    advection           = WENO(),
    closure             = AnisotropicMinimumDissipation(),
    tracers             = (:T, :S),
    buoyancy            = SeawaterBuoyancy(equation_of_state = LinearEquationOfState()),
    boundary_conditions = (u = u_bcs, v = v_bcs),
    forcing             = (S = Forcing(salinity_source),),
)

# ─────────────────────────────────────────────────────────────────────────────
# Initial conditions: linear T stratification (surface warmer), uniform S baseline.
# Small random kicks on u/w concentrated in the upper few meters.
# ─────────────────────────────────────────────────────────────────────────────
T_init(x, y, z) = p.T_bottom + (p.T_surface - p.T_bottom) * (1 + z / grid.Lz)
S_init(x, y, z) = p.S_baseline

u★ = sqrt(p.tau_mag)
Ξ(z) = randn() * exp(z / 4)
uᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
wᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)

set!(model, u = uᵢ, w = wᵢ, T = T_init, S = S_init)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation + adaptive Δt
# ─────────────────────────────────────────────────────────────────────────────
simulation = Simulation(model, Δt = 0.5, stop_time = 30minutes)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))

wizard = TimeStepWizard(cfl = 0.7, max_change = 1.1, max_Δt = 2.0)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# ─────────────────────────────────────────────────────────────────────────────
# NetCDF output → data/oceananigans/scenario_01_<season>.nc
# ─────────────────────────────────────────────────────────────────────────────
output_dir  = joinpath(@__DIR__, "..", "data", "oceananigans")
mkpath(output_dir)
output_path = joinpath(output_dir, "scenario_01_$(SEASON).nc")

simulation.output_writers[:fields] = NetCDFWriter(
    model,
    (u = model.velocities.u,
     v = model.velocities.v,
     w = model.velocities.w,
     T = model.tracers.T,
     S = model.tracers.S),
    filename           = output_path,
    schedule           = TimeInterval(60),       # one snapshot per simulated minute
    overwrite_existing = true,
)

@info "Starting Oceananigans LES ($(SEASON)) → $output_path"
run!(simulation)
@info "Done."
