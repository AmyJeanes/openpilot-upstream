"""Which lanes allow a move, by their turn:lanes arrows: generic (any OSM-tagged map), so navd's planner needs nothing of
the GTA map stack for it. tools/sim/bridge/gta5/map/osm_lanes.py re-exports these for the map code."""

LEFTS = frozenset({'left', 'slight_left', 'sharp_left'})
RIGHTS = frozenset({'right', 'slight_right', 'sharp_right'})


def turn_targets(turns: list[frozenset[str]], move: str, fork: bool = False) -> list[int]:
  """The lanes (of ours, from the left) whose turn:lanes arrows allow a move: 'left' or 'right' (a turn that way, or
  with `fork`, keeping to that side at a fork or exit, which mappers mark slight_left / slight_right), or 'through'. []
  where no lane's arrows say."""
  if not any(turns):
    return []
  if move == 'through':
    return [i for i, t in enumerate(turns) if 'through' in t or not t]
  want = {f'slight_{move}'} if fork else LEFTS if move == 'left' else RIGHTS
  return [i for i, t in enumerate(turns) if t & want]
