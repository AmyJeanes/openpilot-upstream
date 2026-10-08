"""The words for the navigation guidance, shared by the layouts that show it."""
from openpilot.selfdrive.ui.nav.nav_state import Guidance, format_distance
from openpilot.system.ui.lib.multilang import tr

THEN_SHOW = 1000.0  # m after the next maneuver that the one after it shows

MODIFIER_TEXT = {
  "left": tr("left"), "right": tr("right"), "slight left": tr("slight left"), "slight right": tr("slight right"),
  "sharp left": tr("sharp left"), "sharp right": tr("sharp right"), "straight": tr("straight on"), "uturn": tr("U-turn"),
}


def maneuver_phrase(kind: str, modifier: str) -> str:
  side = "left" if "left" in modifier else "right"
  if kind == "arrive":
    return tr("Arrive")
  if kind == "fork":
    return tr("Keep left") if side == "left" else tr("Keep right")
  if kind == "off ramp":
    return tr("Exit left") if side == "left" else tr("Exit right")
  if kind == "on ramp":
    return tr("Freeway entrance")
  if kind == "merge":
    return tr("Merge")
  if kind in ("roundabout", "exit roundabout"):
    return tr("Roundabout")
  if modifier == "uturn":
    return tr("Make a U-turn")
  if modifier == "straight":
    return tr("Continue")
  return tr("Turn {}").format(MODIFIER_TEXT.get(modifier, modifier))


def maneuver_road(g: Guidance) -> str:
  """The maneuver card's second line: what the maneuver is, and the road it leads onto."""
  m = g.maneuver
  if m is None:
    return ""
  if m.type == "arrive":
    return m.primary or g.destination or tr("Destination")
  label = g.secondary or (maneuver_phrase(m.type, m.modifier) if m.type in ("on ramp", "off ramp", "fork") else "")
  return ", ".join(t for t in (label, m.primary) if t) or maneuver_phrase(m.type, m.modifier)


def then_text(g: Guidance, metric: bool) -> str:
  if len(g.maneuvers) < 2:
    return ""
  first, nxt = g.maneuvers[0], g.maneuvers[1]
  gap = nxt.distance - first.distance
  if gap > THEN_SHOW:
    return ""
  if nxt.type == "arrive":
    return tr("Then arrive in {}").format(format_distance(gap, metric))
  phrase = maneuver_phrase(nxt.type, nxt.modifier)
  if nxt.type == "turn" and nxt.modifier in MODIFIER_TEXT and nxt.modifier not in ("straight", "uturn"):
    phrase = MODIFIER_TEXT[nxt.modifier]
  else:
    phrase = phrase[0].lower() + phrase[1:]
  return tr("Then {} in {}").format(phrase, format_distance(gap, metric))


def lane_caption(g: Guidance, metric: bool) -> tuple[str, str]:
  """The lane card's title and line under it."""
  lanes = g.lanes
  current = next((i for i, lane in enumerate(lanes) if lane.current), None)
  active = [i for i, lane in enumerate(lanes) if lane.active]
  m = g.maneuver
  if g.lane_open_distance > 0:
    title = tr("Exit lane opens") if m is not None and m.type == "off ramp" else tr("Turn lane opens")
    sub = tr("from {}").format(format_distance(g.lane_open_distance, metric))
    if g.lane_distance > 0:
      sub += " · " + tr("in by {}").format(format_distance(g.lane_distance, metric))
    return title, sub
  if current is not None and current in active:
    return tr("Stay in lane"), ""
  if current is not None and active:
    title = tr("Keep right") if min(active) > current else tr("Keep left")
  else:
    title = tr("Use the lane shown")
  return title, tr("in lane by {}").format(format_distance(g.lane_distance, metric)) if g.lane_distance > 0 else ""
