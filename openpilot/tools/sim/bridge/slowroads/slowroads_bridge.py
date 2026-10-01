from multiprocessing import Queue

from openpilot.common.params import Params
from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.slowroads.slowroads_world import SlowRoadsWorld
from openpilot.tools.sim.lib.common import World


class SlowRoadsBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, port: int = 8790, debug_port: int = 9339, inject: bool = True):
    super().__init__(dual_camera=True, high_quality=False)
    # a game collision or reset can trip openpilot's excessive-actuation check, which latches and blocks engaging until
    # acknowledged; in the sim that reflects game physics rather than a car, so clear it before selfdrived starts
    Params().remove("Offroad_ExcessiveActuation")
    self.port = port
    self.debug_port = debug_port
    self.inject = inject

  def spawn_world(self, q: Queue) -> World:
    return SlowRoadsWorld(self.simulator_state, q, self.port, self.debug_port, self.inject)
