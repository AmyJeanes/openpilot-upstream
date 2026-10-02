import os
import traceback
from collections import deque

import openpilot.cereal.messaging as messaging
from opendbc.can.packer import CANPacker
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.honda.values import CruiseButtons
from opendbc.car.tesla.values import CANBUS, TeslaSafetyFlags
from openpilot.common.params import Params
from openpilot.selfdrive.pandad.pandad_api_impl import can_list_to_can_capnp
from openpilot.tools.sim.lib.common import SimulatorState

# DI_state.DI_cruiseState
CRUISE_STANDBY = 1
CRUISE_ENABLED = 2
MIN_SET_SPEED = 30 * CV.KPH_TO_MS  # the lowest speed the car's cruise control sets


def is_tesla() -> bool:
  return os.getenv("FINGERPRINT", "").startswith("TESLA")


class SimulatedTesla:
  """Simulates a Tesla Model 3 (panda state + CAN messages) to openpilot: an angle-steering car with openpilot
  longitudinal control (an alpha feature, so AlphaLongitudinalEnabled must be set). openpilot engages with the car's
  cruise control, which this simulates from the bridge's cruise buttons: set/decel enables it at the current speed and
  steps the set speed down, resume/accel resumes or steps it up, and cancel, the brake or openpilot's cancel request
  disables it."""
  packer = CANPacker("tesla_model3_party")

  def __init__(self):
    self.pm = messaging.PubMaster(['can', 'pandaStates'])
    self.sm = messaging.SubMaster(['carControl', 'controlsState', 'carParams', 'selfdriveState'])
    self.idx = 0
    self.params = Params()
    self.obd_multiplexing = False
    self.metric = self.params.get_bool("IsMetric")
    self.cruise_enabled = False
    self.set_speed = 0.0  # m/s, 0 until first set
    self.presses: deque[int] = deque()

  def press(self, button: int):
    """A cruise button press, from the bridge's thread."""
    self.presses.append(button)

  def update_cruise(self, simulator_state: SimulatorState):
    unit = CV.KPH_TO_MS if self.metric else CV.MPH_TO_MS
    while self.presses:
      self.handle_press(self.presses.popleft(), simulator_state, unit)
    if simulator_state.user_brake > 0 or self.sm['carControl'].cruiseControl.cancel:
      self.cruise_enabled = False

  def handle_press(self, pressed: int, simulator_state: SimulatorState, unit: float):
    if pressed == CruiseButtons.DECEL_SET:
      if self.cruise_enabled:
        self.set_speed = max(self.set_speed - unit, MIN_SET_SPEED)
      else:
        self.set_speed = max(round(simulator_state.speed / unit) * unit, MIN_SET_SPEED)
        self.cruise_enabled = True
    elif pressed == CruiseButtons.RES_ACCEL:
      if self.cruise_enabled:
        self.set_speed += unit
      elif self.set_speed > 0:
        self.cruise_enabled = True
    elif pressed == CruiseButtons.CANCEL:
      self.cruise_enabled = False

  def send_can_messages(self, simulator_state: SimulatorState):
    if not simulator_state.valid:
      return
    self.sm.update(0)
    self.update_cruise(simulator_state)

    party, ap_party = CANBUS.party, CANBUS.autopilot_party
    speed_kph = simulator_state.speed * 3.6
    set_speed = self.set_speed * (3.6 if self.metric else CV.MS_TO_MPH)
    # the car's torque sensor, from the bridge's Honda-scaled driver torque (left positive; a lane change nudge is 2000)
    torque_nm = max(-3.0, min(3.0, simulator_state.user_torque / 1000))

    msg = []
    msg.append(self.packer.make_can_msg("DI_speed", party, {"DI_vehicleSpeed": speed_kph}))
    msg.append(self.packer.make_can_msg("DI_systemStatus", party, {"DI_gear": 4, "DI_accelPedalPos": simulator_state.user_gas * 100}))
    msg.append(self.packer.make_can_msg("ESP_status", party, {"ESP_driverBrakeApply": 2 if simulator_state.user_brake > 0 else 1}))
    msg.append(self.packer.make_can_msg("EPAS3S_sysStatus", party, {
      "EPAS3S_internalSAS": -simulator_state.steering_angle,
      "EPAS3S_torsionBarTorque": -torque_nm,
      "EPAS3S_handsOnLevel": 0,
      "EPAS3S_eacStatus": 2,  # EAC_ACTIVE
    }))
    msg.append(self.packer.make_can_msg("DI_state", party, {
      "DI_cruiseState": CRUISE_ENABLED if self.cruise_enabled else CRUISE_STANDBY,
      "DI_digitalSpeed": set_speed,
      "DI_speedUnits": 1 if self.metric else 0,
      "DI_autoparkState": 1,  # STANDBY
    }))
    msg.append(self.packer.make_can_msg("ESP_B", party, {"ESP_vehicleStandstillSts": 1 if simulator_state.speed < 0.1 else 0}))
    msg.append(self.packer.make_can_msg("UI_warning", party, {
      "anyDoorOpen": 0,
      "buckleStatus": 1,
      "leftBlinkerBlinking": 1 if simulator_state.left_blinker else 0,
      "rightBlinkerBlinking": 1 if simulator_state.right_blinker else 0,
    }))
    # only sent by firmware with the 3-bit DAS_steeringControlType that openpilot needs, or it's dashcam only
    msg.append(self.packer.make_can_msg("DI_autonomyHealth", party, {}))

    msg.append(self.packer.make_can_msg("SCCM_steeringAngleSensor", ap_party, {"SCCM_steeringAngleSpeed": 0}))
    msg.append(self.packer.make_can_msg("DAS_status", ap_party, {}))
    msg.append(self.packer.make_can_msg("DAS_control", ap_party, {}))
    msg.append(self.packer.make_can_msg("DAS_steeringControl", ap_party, {}))
    msg.append(self.packer.make_can_msg("DAS_settings", ap_party, {"DAS_autosteerEnabled": 0}))

    self.pm.send('can', can_list_to_can_capnp(msg))

  def send_panda_state(self, simulator_state):
    # fingerprinting waits for the panda to acknowledge switching OBD multiplexing
    if self.params.get_bool("ObdMultiplexingEnabled") != self.obd_multiplexing:
      self.obd_multiplexing = not self.obd_multiplexing
      self.params.put_bool("ObdMultiplexingChanged", True, block=True)

    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': simulator_state.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': 'tesla',
      'alternativeExperience': self.sm["carParams"].alternativeExperience,
      'safetyParam': TeslaSafetyFlags.LONG_CONTROL.value,
    }
    self.pm.send('pandaStates', dat)

  def update(self, simulator_state: SimulatorState):
    try:
      self.send_can_messages(simulator_state)

      if self.idx % 50 == 0:  # only send panda states at 2hz
        self.send_panda_state(simulator_state)

      self.idx += 1
    except Exception:
      traceback.print_exc()
      raise
