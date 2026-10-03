import math

# game metres to degrees on the sphere OSM tools assume, about (0, 0) where a degree is the same length both ways
METRES_PER_DEGREE = 6378137.0 * math.pi / 180


def to_lat_lon(x, y):
  return y / METRES_PER_DEGREE, x / METRES_PER_DEGREE


def to_game(lat, lon):
  return lon * METRES_PER_DEGREE, lat * METRES_PER_DEGREE
