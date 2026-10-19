# GTA V web map

A zoomable map of Los Santos (satellite, hybrid and street views) from the ortho captures, in MapLibre GL with PMTiles.
The tiles use the repo's lat0 Web Mercator frame (`gframe.py`, the same as `navmodel/frame.py`, lat0 34): z22 is about
3.1 cm/px, so map zoom 21 with 256 px raster tiles shows the 3 cm captures 1:1.

Rebuilding everything from the captures (`/mnt/e/gta5_ortho_20261018`):

1. Water: `imgcheck/water/bodies.py <dir>` puts every height tile's water probe on one 1 m world grid and assigns each
   water point a body; `imgcheck/water/compose_full.py <out> --all` draws the water into the albedo tiles with each
   body's colour and opacity (`imgcheck/water/params.json`); `imgcheck/water/fill_cells.py <out>` paints the cells the
   capture skipped as open water (the middle of the Alamo Sea, the port basins) in their body's deep colour.
2. Satellite: `satfull.py <out dir>` mosaics every cell (water tiles first, then the filled ones, then the raw albedo)
   into WebP tiles z22..z10 as `sat.pmtiles`. It works in z16 blocks and resumes from its spool; `--redo-fill`
   rebuilds the blocks holding painted cells.
3. Roads: `roads.py <gta5.osm.pbf> --pmtiles roads.pmtiles` builds the vector roads for the street and hybrid views
   (it needs `osmium`, `mapbox_vector_tile` and `pmtiles` in one environment).
4. 3D: `demfull.py <out dir>` averages the height tiles to z18 (~0.5 m) as terrarium elevation tiles, `dem.pmtiles`.
5. View: put `index.html` beside `sat.pmtiles`, `roads.pmtiles` and `dem.pmtiles` and serve the folder with `rangeserver.py <dir> <port>`
   (PMTiles needs HTTP range requests).

Covered roads (tunnels, roads under buildings and sheds) for the labeller, not the public map: `imgcheck/covered/find.py`
finds road nodes the height tiles show covered, `classify.py` drops foliage and groups the rest, `plan_all.py` writes
an under-deck plan for `ortho_depth.py under`.
