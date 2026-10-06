from multiprocessing import Queue

from openpilot.common.params import Params
from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.gta5.gta5_world import GTA5World
from openpilot.tools.sim.lib.common import World


class GTA5Bridge(SimulatorBridge):
  TICKS_PER_FRAME = 5
  # the game paces its 20 Hz frames by the Windows clock, which runs ~4% faster than WSL's monotonic one, so a 20 Hz limit
  # would skip every 25th frame; the frames pace the camera thread (GTA5World.camera_yuv waits for each)
  CAMERA_HZ = 30

  def __init__(self, port: int = 8791):
    super().__init__(dual_camera=True, high_quality=False)
    # a game collision can trip openpilot's excessive-actuation check, which latches and blocks engaging until
    # acknowledged; in the sim that reflects game physics rather than a car, so clear it before selfdrived starts
    Params().remove("Offroad_ExcessiveActuation")
    self.port = port

  def spawn_world(self, q: Queue) -> World:
    return GTA5World(self.simulator_state, q, self.port)
