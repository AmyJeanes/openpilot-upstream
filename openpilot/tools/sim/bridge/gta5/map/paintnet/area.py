"""area.py x0 x1 y0 y1 out.geojson: the pilot box's outline for the web map (its sides cut in 50, so they follow the
map's frame)."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe

x0, x1, y0, y1 = map(float, sys.argv[1:5])
t = np.linspace(0, 1, 50)
X = np.r_[x0 + (x1 - x0) * t, np.full(50, x1), x1 - (x1 - x0) * t, np.full(50, x0), x0]
Y = np.r_[np.full(50, y0), y0 + (y1 - y0) * t, np.full(50, y1), y1 - (y1 - y0) * t, y0]
lat, lon = gframe.lat_lon(X, Y)
json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"kind": "area", "name": "interchange pilot", "box": [x0, x1, y0, y1]},
           "geometry": {"type": "Polygon", "coordinates": [np.stack([lon, lat], 1).round(8).tolist()]}}]}, open(sys.argv[5], "w"))
