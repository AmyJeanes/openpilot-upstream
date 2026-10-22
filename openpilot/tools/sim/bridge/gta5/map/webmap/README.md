# GTA V web map

A zoomable map of Los Santos (satellite, hybrid and street views) from the ortho captures, in MapLibre GL with PMTiles.
The tiles use the repo's lat0 Web Mercator frame (`gframe.py`, the same as `navmodel/frame.py`, lat0 34): z22 is about
3.1 cm/px, so map zoom 21 with 256 px raster tiles shows the 3 cm captures 1:1.

Rebuilding everything from the captures (`/mnt/e/gta5_ortho_20261018`):

1. Gaps: `imgcheck/water/fill_all.py <tiles> <out>` fills the albedo's empty pixels (water one colour, glass and the
   like from the lit frame) into `full_filled`, writing only the tiles that change.
2. Water: `imgcheck/water/bodies.py <dir>` puts every height tile's water probe on one 1 m world grid and assigns each
   water point a body; `imgcheck/water/compose_full.py <out> --all` draws the water into the albedo tiles with each
   body's colour and opacity (`imgcheck/water/params.json`); `imgcheck/water/fill_cells.py <out>` paints the cells the
   capture skipped as open water (the middle of the Alamo Sea, the port basins) in their body's deep colour.
3. Satellite: `satfull.py <out dir>` mosaics every cell (water tiles first, then the filled ones, then the raw albedo)
   into WebP tiles z22..z10 as `sat.pmtiles`. It works in z16 blocks and resumes from its spool; `--redo-fill`
   rebuilds the blocks holding painted cells.
4. Roads: `roads.py <gta5.osm.pbf> --pmtiles roads.pmtiles` builds the vector roads for the street and hybrid views
   (it needs `osmium`, `mapbox_vector_tile` and `pmtiles` in one environment).
5. View: put `index.html` beside `sat.pmtiles` and `roads.pmtiles` and serve the folder with `rangeserver.py <dir> <port>`
   (PMTiles needs HTTP range requests).
6. Beneath decks: `undertiles.py <out dir>` cuts every passing under-deck shot (`under/` from the road crossings,
   `under_gaps/` from `imgcheck/under_gaps.py`, `under_covered*/` from the covered roads below; sidecar `pass`) to the
   drawn pixels lower than the top-level height tiles (`depth/`) by over 2 m and within 1 m of road at the shot's own
   height (road or gutter in the game files' surface raster, or the game's road links: no buildings' interiors), as
   transparent WebP tiles `under.pmtiles` (the lowest level on top) and their outline `under_footprint.geojson`; the page shows them in hybrid view with
   "Lowest level only". A shot passes when the part of its lower road it drew (at least a quarter) reads right; after
   changing the checks, `ortho_depth.py recheck <run dir> --depth <depth tiles>` redoes a run's sidecars from the saved
   heights. `under_gaps.py hidden`, then `plan`, plans shots (`ortho_depth.py under`) for the hidden road the shots so
   far don't show.

After recapturing some cells (`imgcheck/ortho.py run`; `--calm` for calm water, so the surf matches between
neighbouring shots; `imgcheck/ortho_edges.py` plans the uncaptured cells that still hold land or shallow seabed): put
the new tiles in the grid, delete their `full_filled` and water tiles, run the steps above (`compose_full.py --all`
only draws tiles without a water tile) and `satfull.py --redo-cells "i,j;..."`, which rebuilds the z16 blocks holding
them from the spool. Without the spool, `satfull.py` rebuilds the whole map. Reload an open map after `sat.pmtiles`
is rewritten.

Covered roads (tunnels, roads under buildings and sheds), for the labeller and the beneath-decks view: `imgcheck/covered/find.py`
finds road nodes the height tiles show covered, `classify.py` drops foliage and groups the rest, `plan_all.py` writes
an under-deck plan for `ortho_depth.py under`.
