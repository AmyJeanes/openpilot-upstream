"""What the plugin plays and shows as the car engages and hands back, so the player knows what the driver is doing: one
of openpilot's sounds (selfdrive/assets/sounds, installed with the plugin) and a few seconds of text below the HUD's set
speed, sent as {"type": "event", "sound": ..., "text": ..., "secs": ...}, either of the first two left out.
- engage: the map driver or openpilot takes the car ("Map driver engaged", "Cruising" on a cruise route, "openpilot
  engaged").
- disengage: the player hands back, with the brake, gas, steering or the engage key ("Disengaged"), or the map driver
  arrives ("Arrived").
- warning: the driver gives up on its own: a map drive's abort, as it starts to stop ("Map driver gave up: collision"),
  or openpilot disengaging with no input from the player ("openpilot disengaged").
- refuse: the engage key engages nothing ("Can't engage: no route"; cruising, "no road ahead" once the brain gives up
  waiting for it), or openpilot doesn't engage within REFUSE_WAIT s.
While the engage key waits to cruise there's text alone ("Looking for the road ahead"), and pressing it again cancels
with the disengage sound ("Not cruising").
The arrow keys' prompts to the map driver (gta5_brain, Cruise.prompt, MapDrive.ask_lane) get text alone as they're
taken ("Next left: Vinewood Blvd", "Keeping right", "Changing lane left") and the refuse sound with why where they
can't be ("No left here", "Too late for that right").
Expert mode's drives are silent. GTA5_SOUNDS=0 leaves out the sounds (for recording runs), GTA5_HUD_EVENTS=0 the text."""
import os

SOUNDS = os.getenv("GTA5_SOUNDS", "1") != "0"
HUD_EVENTS = os.getenv("GTA5_HUD_EVENTS", "1") != "0"
SECS = 4.0  # s the text shows
USER_GRACE = 2.0  # s after the player's input or the engage key that openpilot disengaging is the player's doing
REFUSE_WAIT = 2.0  # s openpilot has to engage after the engage key asks it to
# the player's own hand-backs, as the drivers and the map drive word them (gta5_drivers.TOOK_OVER, MapDrive._abort)
USER = ("driver input", "the driver pressed the engage key")
# why a map drive aborts (MapDrive._abort, MapDriver.step) in short, by the first of these its words contain
ABORTS = (
  ("collision", "collision"), ("off the path", "off the path"), ("off the road", "off the road"),
  ("lateral acceleration", "too fast for a bend"), ("oncoming", "in the oncoming lanes"), ("no progress", "stuck"),
  ("deadlock", "deadlock"), ("pedestrian", "blocked by a pedestrian"), ("parked vehicle", "blocked by a parked car"),
  ("static obstacle", "blocked by an obstacle"), ("by a vehicle", "blocked by a car"),
  ("no lanes on the route", "no lanes on the route"), ("no lanes where it starts", "no lanes here"),
  ("from its lane line", "too far from its lane"), ("sharp corner", "a sharp corner ahead"), ("plan:", "no plan"),
  ("no route", "no route"), ("the road ends", "the road ends"),
)


def short(why: str) -> str:
  """A map drive's reason to stop, in a few words for the HUD."""
  why = why.removeprefix("abort: ")
  return next((s for words, s in ABORTS if words in why), why[:40])


def event(kind: str, who: str = "", why: str = "", cruise: bool = False) -> tuple[str, str]:
  """The (sound, text) for a transition: kind "engage" (who: map, openpilot), "handback" (who drove, and why, as the
  drivers have it: "arrived", "abort: <why>", the player's, or "" for openpilot's own), "refuse" (why), or the engage
  key's wait to cruise: "wait" (no sound) and "cancel"."""
  if kind == "wait":
    return "", "Looking for the road ahead"
  if kind == "cancel":
    return "disengage", "Not cruising"
  if kind == "engage":
    return "engage", "Cruising" if cruise else "Map driver engaged" if who == "map" else "openpilot engaged"
  if kind == "refuse":
    return "refuse", f"Can't engage: {why}"
  if why == "arrived":
    return "disengage", "Arrived"
  if why.removeprefix("abort: ") in USER:
    return "disengage", "Disengaged"
  if who == "openpilot":
    return "warning", "openpilot disengaged"
  return "warning", f"Map driver gave up: {short(why)}"


PROMPTS = {  # an arrow key's prompt: kind -> (sound, text), {side} and {road} filled in
  "turn": ("", "Next {side}{road}"), "keep": ("", "Keeping {side}{road}"), "cancel": ("", "Turn cancelled"),
  "late": ("refuse", "Too late for that {side}"), "forbidden": ("refuse", "Can't turn {side} there"),
  "router_busy": ("refuse", "Router busy: try {side} again"),
  "none": ("refuse", "No {side} here"), "no_lane": ("refuse", "No {side} here"),
  "lane": ("", "Changing lane {side}"), "wait": ("", "Waiting to change lane {side}"),
  "blocked": ("refuse", "Lane {side} not clear"), "busy": ("refuse", "Already changing lane"),
  "lane_no": ("refuse", "Can't change lane {side} here"), "lane_cancel": ("", "Lane change cancelled"),
  "hatched": ("refuse", "No {side} here: hatched"),
}


def prompt(kind: str, side: str, road: str = "") -> tuple[str, str]:
  """The (sound, text) for an arrow key's prompt to the map driver: what it does, or why it can't."""
  sound, text = PROMPTS[kind]
  return sound, text.format(side=side, road=f": {road}" if road else "")


class Announcer:
  """Follows who's engaged step by step and sends the plugin an event as that changes, with why from the brain."""
  def __init__(self, send):
    self.send = send
    self.was: dict[str, bool] = {"map": False, "openpilot": False}
    self.user_t = float("-inf")  # the player's last input or press of the engage key
    self.asked: float | None = None  # when the engage key asked openpilot to engage, until it does
    self.why: str | None = None  # why the map driver handed back, from the brain
    self.told = None  # the map drive whose abort was announced
    self.quiet = False

  def pressed(self, now: float):
    """The engage key, which counts as the player's input."""
    self.user_t = now

  def asked_openpilot(self, now: float):
    self.asked = now

  def handback(self, why: str):
    """Why the map driver hands back this step."""
    self.why = why

  def refused(self, why: str):
    if not self.quiet:
      self._send(*event("refuse", why=why))

  def prompt(self, kind: str, side: str, road: str = ""):
    """What an arrow key's prompt to the map driver does (events.prompt)."""
    if not self.quiet:
      self._send(*prompt(kind, side, road))

  def status(self, kind: str):
    """The engage key's wait to cruise: "wait" or "cancel"."""
    if not self.quiet:
      self._send(*event(kind))

  def step(self, now: float, driver, openpilot, user: bool, quiet: bool, route=None):
    """After the drivers' step: driver is the engage key's (a MapDriver, or openpilot itself), user the player's
    input now, quiet while expert mode is on."""
    if user:
      self.user_t = now
    self.quiet = quiet
    on = {"map": driver is not openpilot and driver.engaged, "openpilot": openpilot.engaged}
    md = getattr(driver, "md", None) if on["map"] else None
    if quiet:
      self.was, self.asked, self.why = on, None, None
      return
    if md is not None and md.phase == "abort" and self.told is not md:
      self.told = md
      if md.why not in USER:
        self._send(*event("handback", "map", "abort: " + md.why))
    for who in ("map", "openpilot"):
      if on[who] and not self.was[who]:
        self.asked = None if who == "openpilot" else self.asked
        self._send(*event("engage", who, cruise=who == "map" and bool(getattr(route, "cruise", False))))
      elif self.was[who] and not on[who]:
        why = self.why if who == "map" else None
        if why is None:
          why = USER[0] if now - self.user_t < USER_GRACE else ""
        mine = why.removeprefix("abort: ") not in USER
        if who == "map" and mine and why.startswith("abort: ") and self.told is getattr(driver, "md", None):
          continue  # announced as it started to stop
        self._send(*event("handback", who, why))
    if self.asked is not None and not on["openpilot"] and now - self.asked > REFUSE_WAIT:
      self.asked = None
      self._send(*event("refuse", why="openpilot didn't engage"))
    self.was, self.why = on, None

  def _send(self, sound: str, text: str):
    msg = {"type": "event", **({"sound": sound} if SOUNDS and sound else {}), **({"text": text, "secs": SECS} if HUD_EVENTS else {})}
    if len(msg) > 1:
      self.send(msg)
