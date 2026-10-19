"""GTA metres -> Web Mercator in the repo's lat0 frame (navmodel/frame.py). Single source of truth for tiles and roads."""
import math
import numpy as np

EARTH_CIRCUMFERENCE = 40075016.68557849
LAT0 = 34.0
K = EARTH_CIRCUMFERENCE * math.cos(math.radians(LAT0))       # metres per unit of normalised mercator
MY0 = (1.0 - math.asinh(math.tan(math.radians(LAT0))) / math.pi) / 2.0
TILE = 256


def merc(x, y):
    """GTA metres -> normalised mercator (0..1, y down)."""
    return 0.5 + np.asarray(x, dtype=np.float64) / K, MY0 - np.asarray(y, dtype=np.float64) / K


def px_per_m(z):
    return TILE * (1 << z) / K


# verbatim copy of navmodel/frame.py lat_lon, used only to verify
def lat_lon(x, y, lat0=LAT0):
    k = EARTH_CIRCUMFERENCE * math.cos(math.radians(lat0))
    my0 = (1.0 - math.asinh(math.tan(math.radians(lat0))) / math.pi) / 2.0
    mx, my = 0.5 + np.asarray(x, dtype=np.float64) / k, my0 - np.asarray(y, dtype=np.float64) / k
    lon = mx * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * my))))
    return lat, lon


def latlon_to_merc(lat, lon):
    mx = (np.asarray(lon, dtype=np.float64) + 180.0) / 360.0
    my = (1.0 - np.arcsinh(np.tan(np.radians(np.asarray(lat, dtype=np.float64)))) / np.pi) / 2.0
    return mx, my
