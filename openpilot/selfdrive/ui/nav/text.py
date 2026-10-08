"""The words for the navigation guidance."""
from openpilot.selfdrive.ui.nav.nav_state import Guidance
from openpilot.system.ui.lib.multilang import tr

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
  """The line under the maneuver's distance: the road it leads onto, after its exit or sign; the destination for the
  arrival."""
  m = g.maneuver
  if m is None:
    return ""
  if m.type == "arrive":
    return g.destination or m.primary or tr("Destination")
  label = g.secondary or (maneuver_phrase(m.type, m.modifier) if m.type in ("on ramp", "off ramp", "fork") else "")
  return " · ".join(t for t in (label, m.primary) if t) or maneuver_phrase(m.type, m.modifier)
