"""
3D Demo: Hybrid ocean current model
Surface currents from CMEMS data + Ekman spiral propagation with depth.

Shows how real surface current vectors rotate and decay as they go deeper,
following Ekman dynamics driven by actual measured surface flow.
NaN (land) points are set to zero current (no flow over land).
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ──────────────────── Config ────────────────────
FILEPATH = 'data/abu_dhabi_ocean_data.nc'
TIME = '2020-01-01'
LATITUDE = 24.5       # Abu Dhabi latitude for Coriolis
EDDY_VISCOSITY = 0.05 # turbulent mixing coefficient [m²/s]
DEPTH_LEVELS = np.array([0, 5, 10, 20, 30, 50, 75, 100])  # meters
ARROW_SCALE = 1.5     # quiver arrow length multiplier

# ──────────────────── Ekman parameters ────────────────────
OMEGA = 7.2921e-5  # Earth angular velocity [rad/s]
sin_lat = np.sin(np.deg2rad(LATITUDE))
f = 2.0 * OMEGA * sin_lat  # Coriolis parameter
D_E = np.sqrt(2.0 * EDDY_VISCOSITY / abs(f))  # Ekman depth

print(f'Ekman depth D_E = {D_E:.1f} m')
print(f'Coriolis parameter f = {f:.2e} rad/s')

# ──────────────────── Load surface data ────────────────────
ds = xr.open_dataset(FILEPATH)
snap = ds.sel(time=TIME, method='nearest')
surface = snap.isel(depth=0)

lons = surface.longitude.values
lats = surface.latitude.values

# ──────────────────── Zero out NaN (land) points ────────────────────
u_surface = np.nan_to_num(surface.uo.values, nan=0.0)
v_surface = np.nan_to_num(surface.vo.values, nan=0.0)

# Surface speed and direction at each grid point
speed_surface = np.sqrt(u_surface**2 + v_surface**2)
direction_surface = np.arctan2(v_surface, u_surface)

n_total = u_surface.size
print(f'Grid: {len(lats)} lat x {len(lons)} lon')
n_ocean = int((~np.isnan(surface.uo.values)).sum())
print(f'Grid points: {n_total} ({n_ocean} ocean, {n_total - n_ocean} land set to zero)')

# ──────────────────── Apply Ekman spiral at each depth ────────────────────
all_lon, all_lat, all_z = [], [], []
all_u, all_v = [], []
all_speed = []

for d in DEPTH_LEVELS:
    zn = np.pi * d / D_E
    decay = np.exp(-zn)

    u_at_depth = speed_surface * decay * np.cos(direction_surface + np.pi/4 - zn)
    v_at_depth = speed_surface * decay * np.sin(direction_surface + np.pi/4 - zn)
    speed_at_depth = speed_surface * decay

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            all_lon.append(lon)
            all_lat.append(lat)
            all_z.append(-d)
            all_u.append(u_at_depth[i, j])
            all_v.append(v_at_depth[i, j])
            all_speed.append(speed_at_depth[i, j])

all_lon = np.array(all_lon)
all_lat = np.array(all_lat)
all_z = np.array(all_z)
all_u = np.array(all_u)
all_v = np.array(all_v)
all_speed = np.array(all_speed)

# ──────────────────── 3D quiver plot ────────────────────
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

speed_norm = plt.Normalize(vmin=0, vmax=max(all_speed.max(), 0.01))
cmap = plt.cm.viridis
colors = cmap(speed_norm(all_speed))

ax.quiver(
    all_lon, all_lat, all_z,
    all_u * ARROW_SCALE, all_v * ARROW_SCALE, np.zeros_like(all_u),
    colors=colors, arrow_length_ratio=0.3, alpha=0.6, linewidth=0.8
)

ax.set_xlabel('Longitude [°E]')
ax.set_ylabel('Latitude [°N]')
ax.set_zlabel('Depth [m]')
ax.set_title(f'Hybrid Model: CMEMS Surface + Ekman Spiral ({TIME})\n'
             f'D_E = {D_E:.1f}m, lat = {LATITUDE}°N, A_z = {EDDY_VISCOSITY} m²/s')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=speed_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label('Current speed [m/s]')

for d in DEPTH_LEVELS:
    ax.plot([], [], [], ' ', label=f'{d}m')
ax.legend(title='Depth levels', loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()
ds.close()
