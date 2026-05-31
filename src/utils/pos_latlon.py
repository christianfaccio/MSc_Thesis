import numpy as np

# Earth radius for local meter <-> lat/lon conversion
EARTH_RADIUS = 6371000.0  # meters

def pos_to_latlon(x: float, y: float, z: float, 
                  origin_lon: float, origin_lat: float) -> float:
        """Convert position [x, y, z] in meters to (lat, lon, depth)."""
        meters_per_deg_lat = np.pi / 180.0 * EARTH_RADIUS
        meters_per_deg_lon = (np.pi / 180.0 * EARTH_RADIUS * np.cos(np.deg2rad(origin_lat)))
        lon = origin_lon + x / meters_per_deg_lon
        lat = origin_lat + y / meters_per_deg_lat
        depth = abs(z)
        return lat, lon, depth

def latlon_to_pos(lat: float, lon: float, depth: float,
                  origin_lon: float, origin_lat: float) -> float:
        "Convert (lat, lon, depth) in position (m) (x, y, z)"
        meters_per_deg_lat = np.pi / 180.0 * EARTH_RADIUS
        meters_per_deg_lon = (np.pi / 180.0 * EARTH_RADIUS * np.cos(np.deg2rad(origin_lat)))
        x = (lon - origin_lon) * meters_per_deg_lon
        y = (lat - origin_lat) * meters_per_deg_lat
        z = -depth
        return x, y, z