"""Navigation's request to the driver in the UI (navInstruction.request, PLAN D): read from the message, worded by
stage, and the state the prompt is shown from (no rendering)."""
from openpilot.cereal import messaging
from openpilot.selfdrive.ui.nav.nav_state import NavRequest, RequestState, instruction_guidance, nav_request, request_prompt


def instruction(kind="laneChange", side="right", reason="exit", distance=180.0, changes=1, stage=1, rid=3, approved=False):
  msg = messaging.new_message("navInstruction")
  ni = msg.navInstruction
  ni.valid = True
  r = ni.request
  r.kind, r.side, r.reason, r.distance, r.laneChanges, r.stage, r.id, r.approved = kind, side, reason, distance, changes, stage, rid, approved
  return messaging.log_from_bytes(msg.to_bytes()).navInstruction


def test_read():
  g = instruction_guidance(instruction())
  assert g.request == NavRequest("laneChange", "right", "exit", 180.0, 1, 1, 3, False)
  assert nav_request(instruction(kind="none")) is None
  assert instruction_guidance(messaging.new_message("navInstruction").navInstruction).request is None  # none sent


def test_prompt_by_stage():
  assert request_prompt(nav_request(instruction())) == ("Change lanes right for the exit", "signal to confirm")
  assert request_prompt(nav_request(instruction(stage=2))) == ("Change lanes right for the exit", "nudge the wheel to confirm")
  assert request_prompt(nav_request(instruction(stage=3))) == ("Changing lanes right for the exit", "")
  assert request_prompt(nav_request(instruction(approved=True))) == ("Changing lanes right for the exit", "")
  assert request_prompt(nav_request(instruction(side="left", reason="turn"))) == ("Change lanes left for the turn", "signal to confirm")
  assert request_prompt(nav_request(instruction(kind="turn", side="left", reason="turn"))) == ("Turn left", "signal to confirm")
  assert request_prompt(nav_request(instruction(reason="other")))[0] == "Change lanes right"
  assert request_prompt(None) is None


def test_state():
  s = RequestState(on_s=0.3, off_s=1.0)
  a = nav_request(instruction(rid=1))
  assert s.update(a, 0.0) is None
  assert s.update(a, 0.35) == a  # held: shown
  approved = nav_request(instruction(rid=1, approved=True))
  assert s.update(approved, 0.4).approved  # the same request's approval at once
  assert s.update(None, 1.0) == approved  # a dropout
  assert s.update(None, 1.5) == approved
  assert s.update(None, 2.1) is None  # gone
  b = nav_request(instruction(rid=2, side="left"))
  assert s.update(b, 3.0) is None and s.update(b, 3.31) == b
  c = nav_request(instruction(rid=3))
  assert s.update(c, 3.4) == b  # a new one replaces it once held
  assert s.update(c, 3.71) == c
