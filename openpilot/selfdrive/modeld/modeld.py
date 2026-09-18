#!/usr/bin/env python3
from collections.abc import Callable
import base64
import json
import sys
import ctypes
from functools import cached_property
import os
os.environ['GMMU'] = '0' # for chestnut fast loading, noop for qcom
from tinygrad.device import Buffer, Device
from tinygrad.dtype import DType, dtypes
from tinygrad.tensor import Tensor
from tinygrad.helpers import round_up
from tinygrad.uop.ops import UOp
import math
import pickle
import threading
import time
import numpy as np
import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.messaging import PubMaster, SubMaster
from openpilot.cereal.services import SERVICE_LIST
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, should_stop, smooth_value, get_curvature_from_plan
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_driving_model_data, fill_pose_msg, PublishState
from openpilot.common.file_chunker import open_file_chunked
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.helpers import MODELS_DIR, chestnut_present, chestnut_compiled, modeld_pkl_path, load_oob
from openpilot.selfdrive.modeld import reproject_c4 as RC

SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
REPROJECT_C4 = os.getenv('REPROJECT_C4', '1') != '0'  # 3X cameras -> comma 4 geometry on the QCOM, in front of the comma 4 big model
REPROJECT_C4_REFINE = os.getenv('REPROJECT_C4_REFINE', '1') != '0'  # monitor the rotation with the model's own wide_from_device output
C4_CAM = (1344, 760)

LAT_SMOOTH_SECONDS = 0.0
LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3
BIG_MODEL_TIMEOUT = 60


def get_action_from_model(model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                          lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
  if 'action' not in model_output:
    plan = model_output['plan'][0]
    desired_accel = get_accel_from_plan(plan[:,Plan.VELOCITY][:,0],
                                        plan[:,Plan.ACCELERATION][:,0],
                                        ModelConstants.T_IDXS,
                                        action_t=long_action_t)
    desired_curvature = get_curvature_from_plan(plan[:,Plan.T_FROM_CURRENT_EULER][:,2],
                                                plan[:,Plan.ORIENTATION_RATE][:,2],
                                                ModelConstants.T_IDXS,
                                                v_ego,
                                                lat_action_t)
  else:
    desired_accel = model_output['action'][0,1]
    desired_curvature = model_output['action'][0,0] / (max(1.0, v_ego))**2
  stop = should_stop(v_ego, desired_accel)
  desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)
  if v_ego > MIN_LAT_CONTROL_SPEED:
    desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, LAT_SMOOTH_SECONDS)
  else:
    desired_curvature = prev_action.desiredCurvature

  return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                desiredAcceleration=float(desired_accel),
                                shouldStop=bool(stop))


class ChestnutGpuState:
  # GPU metrics require modeld's GPU context
  def __init__(self, pm: PubMaster, big: bool):
    self.pm = pm
    self.big = big
    self.valid = True
    self.sends = 0
    self.metrics = {}

  @cached_property
  def power_limit(self) -> int:
    smu = Device["AMD"].iface.dev_impl.smu
    return smu._send_msg(smu.smu_mod.PPSMC_MSG_GetPptLimit, 0, read_back_arg=True, timeout=100)

  def send(self) -> None:
    msg = messaging.new_message('chestnutGpuState')
    state = msg.chestnutGpuState
    self.sends += 1
    if self.big and "AMD" in Device._opened_devices and self.sends % 100 == 1:
      try:
        smu = Device["AMD"].iface.dev_impl.smu
        metrics_t = smu.smu_mod.SmuMetricsExternal_t
        smu._send_msg(smu.smu_mod.PPSMC_MSG_TransferTableSmu2Dram, smu.smu_mod.TABLE_SMU_METRICS, timeout=100)
        metrics_buf = bytearray(smu.adev.vram.view(smu.driver_table_paddr, ctypes.sizeof(metrics_t))[:])
        metrics = metrics_t.from_buffer(metrics_buf).SmuMetrics
        self.metrics = {'tempC': metrics.AvgTemperature[smu.smu_mod.TEMP_HOTSPOT],
                        'memoryTempC': metrics.AvgTemperature[smu.smu_mod.TEMP_MEM],
                        'powerDrawW': metrics.AverageSocketPower,
                        'powerLimitW': self.power_limit,
                        'gpuUsagePercent': metrics.AverageGfxActivity,
                        'gpuClockMhz': metrics.AverageGfxclkFrequencyPostDs,
                        'fanSpeedRpm': metrics.AvgFanRpm}
        self.valid = True
      except Exception:
        if self.valid:
          cloudlog.exception("chestnut state read failed")
        self.valid = False
        self.metrics.clear()
    if self.big:
      for k, v in self.metrics.items():
        setattr(state, k, v)

    msg.valid = not self.big or (self.valid and bool(self.metrics))
    self.pm.send('chestnutGpuState', msg)


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


def input_view(buffer: Buffer, shape: tuple[int, ...], dtype: DType, offset: int) -> Tensor:
  view = buffer.view(math.prod(shape), dtype, offset).ensure_allocated()
  return Tensor(UOp.from_buffer(view)).reshape(shape)


class ModelState:
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, cam_w: int, cam_h: int, chestnut: bool, reproject: bool = False):
    self.rp = None
    self.src_size = get_nv12_info(cam_w, cam_h)[3]
    if reproject:
      self.src_wh, self.cache_dir = (cam_w, cam_h), os.environ.get('XDG_CACHE_HOME', '/data/tgcache')
      self.rotation = RC.load_rotation(); self.rot_file = RC.read_rotation_file()
      cloudlog.warning(f"reproject_c4: rotation {np.degrees(self.rotation).round(3)} deg from {'rotation.json' + (' (fitted)' if self.rot_file.get('fitted') else '') if self.rot_file else 'CalibrationParams seed / fleet median'}")
      self.calib = RC.calib_from_rotvec(self.rotation)
      self.rp = RC.Reprojector(self.src_wh, C4_CAM, device='QCOM', cache_dir=self.cache_dir, calib=self.calib)
      self.meter = RC.SeamMeter(self.src_wh, C4_CAM, calib=self.calib)
      self.refiner = RC.RotationRefiner(self.rotation)
      self.res_sum, self.res_n = np.zeros(3), 0  # window residuals against the applied rotation
      self.pending = None; self.loader: threading.Thread | None = None; self.rot_mtime = 0.0
      RC.save_applied(self.rotation, self.rot_file.get('fitted', False))
      self._src_tensors: dict[int, Tensor] = {}
      self.rp_time = 0.0; self.rp_enqueue = 0.0
      cam_w, cam_h = C4_CAM  # the warp pkl + model see a comma 4 camera
    jits = load_oob(open_file_chunked(modeld_pkl_path(chestnut)))
    self.model_device = jits['input_specs']['new_img'][2]
    self.input_shapes = {name: (shape, np.dtype(dtype)) for name, (shape, dtype, _) in jits['input_specs'].items()}
    self.state_pairs = {name: f'next_{name}' for name in self.input_shapes if f'next_{name}' in jits['metadata']['output_shapes']}
    self.vision_input_names = ('img', 'big_img')
    self.output_slices = pickle.loads(base64.b64decode(jits['metadata']['metadata']['output_slices']))

    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    self.chestnut = chestnut

    stride, y_height, uv_height, _ = get_nv12_info(cam_w, cam_h)
    self.frame_copy_size = stride * (y_height + uv_height)
    self.pack_inputs()
    if self.rp is not None:
      self.rp.bind(self.frames[1], self.frames[0])  # the kernel writes the comma 4 frames straight into the packed upload
    with open(MODELS_DIR / f'{"big_" if chestnut else ""}driving_warp_{cam_w}x{cam_h}_tinygrad.pkl', 'rb') as f:
      self.run_warp = pickle.load(f)['run']
    self.run_model = jits['run']
    self.outputs = {name: Tensor(np.zeros(shape, dtype=dtype), device=device).realize() for name, (shape, dtype, device) in jits['output_specs'].items()}
    for name, next_name in self.state_pairs.items():
      state = self.input_queues[name]
      self.outputs[next_name] = input_view(state._buffer(), state.shape, state.dtype, 0)
    self.parser = Parser()

  def pack_inputs(self) -> None:
    # Pack host inputs into one upload to reduce USB transfer overhead for the eGPU.
    self.input_queues = {name: Tensor(np.zeros(shape, dtype=dtype), device=self.model_device).realize()
                         for name, (shape, dtype) in self.input_shapes.items() if name in self.state_pairs}
    shapes = {'tfm': (2, 3, 3)} | {name: shape for name, (shape, _) in self.input_shapes.items()
                                   if name not in self.state_pairs and name != 'new_img'}
    npy_size = sum(round_up(math.prod(shape) * 4, 128) for shape in shapes.values())
    self.packed_input = np.zeros(npy_size + 2 * self.frame_copy_size, dtype=np.uint8)
    self.input_host = Tensor(self.packed_input, device='NPY')._buffer()
    self.input_device = Tensor(self.packed_input, device=self.model_device)._buffer()
    self.npy = {}
    offset = 0
    for name, shape in shapes.items():
      self.npy[name] = np.ndarray(shape, dtype=np.float32, buffer=self.packed_input, offset=offset)
      self.input_queues[name] = input_view(self.input_device, shape, dtypes.float32, offset)
      offset += round_up(self.npy[name].nbytes, 128)
    self.frames = self.packed_input[npy_size:].reshape(2, self.frame_copy_size)
    self.warp_inputs = {'input_frame': input_view(self.input_device, self.frames.shape, dtypes.uint8, npy_size), 'M_inv': self.input_queues.pop('tfm')}

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    return {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}

  def refine(self, model_output: dict[str, np.ndarray], v_ego: float) -> None:
    """The model's wide_from_device_euler is the residual of the applied rotation as the model sees it. Without a direct fit
    (reprojectd) the window residuals are averaged and written as the estimate for the next start; with one it is a monitor."""
    if 'wide_from_device_euler' not in model_output:
      return
    new = self.refiner.push(model_output['wide_from_device_euler'][0], model_output['wide_from_device_euler_stds'][0], v_ego)
    self.refiner.rotvec = np.array(self.rotation, np.float64)  # never compound: every window measures against what is applied
    if new is None or self.rot_file.get('fitted'):
      return
    self.res_sum += self.refiner.last_residual; self.res_n += 1
    est = RC.matrix_to_rotvec(RC.rotvec_to_matrix(self.rotation) @ RC.rotvec_to_matrix(self.res_sum / self.res_n * self.refiner.k))
    RC.save_rotation(est, applied=list(self.rotation), windows=self.res_n)
    cloudlog.warning(f"reproject_c4: rotation estimate {self.refiner.steps} ({self.res_n} windows): residual {np.degrees(self.refiner.last_residual).round(3)} deg, "
                     f"applied {np.degrees(self.rotation).round(3)} -> next start {np.degrees(est).round(3)} deg")

  def poll_fit(self) -> None:
    """Once a second: a new direct fit in rotation.json (reprojectd, tables already built) is loaded in a thread and swapped in
    between runs (one upload). This happens while calibrationd holds, so the car cannot be engaged."""
    if self.pending is not None:
      T, calib, meter, rot = self.pending; self.pending = None; self.loader = None
      t0 = time.perf_counter()
      self.rp.reload(T); self.calib, self.meter, self.rotation = calib, meter, rot
      self.rot_file = RC.read_rotation_file(); self.refiner.rotvec = np.array(rot, np.float64); self.res_sum[:] = 0; self.res_n = 0
      RC.save_applied(rot, True)
      cloudlog.warning(f"reproject_c4: fitted rotation {np.degrees(rot).round(3)} deg swapped in ({(time.perf_counter() - t0) * 1e3:.0f} ms)")
      return
    if self.loader is not None:
      return
    try:
      mtime = os.stat(RC.ROTATION_FILE).st_mtime
    except OSError:
      return
    if mtime == self.rot_mtime:
      return
    self.rot_mtime = mtime; d = RC.read_rotation_file()
    if not d.get('fitted') or np.allclose(d['rotvec'], self.rotation, atol=1e-6):
      return
    calib = RC.calib_from_rotvec(d['rotvec'])
    if not os.path.exists(RC.table_path(self.src_wh, C4_CAM, self.cache_dir, calib)):
      cloudlog.warning("reproject_c4: fitted rotation has no tables yet"); self.rot_mtime = 0.0; return
    def load():  # ~150 ms of npz decompression + the meter's ring: off the model loop
      T = RC.load_tables(self.src_wh, C4_CAM, self.cache_dir, calib)
      self.pending = (T, calib, RC.SeamMeter(self.src_wh, C4_CAM, calib=calib), tuple(float(v) for v in d['rotvec']))
    self.loader = threading.Thread(target=load, daemon=True); self.loader.start()

  def src_tensor(self, buf) -> Tensor:
    data = buf.data if hasattr(buf, 'data') else buf
    ptr = np.frombuffer(data, dtype=np.uint8).ctypes.data
    if ptr not in self._src_tensors:  # one mapping per VisionIPC ring slot
      self._src_tensors[ptr] = Tensor.from_blob(ptr, (self.src_size,), dtype='uint8', device='QCOM')
    return self._src_tensors[ptr]

  def run(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray], after_enqueue: Callable[[], None] | None = None) -> dict[str, np.ndarray]:
    if self.rp is not None:
      t0 = time.perf_counter()
      self.rp(self.src_tensor(bufs['big_img']), self.src_tensor(bufs['img']), **inputs.get('reproj_match', {}))
      self.rp_enqueue = time.perf_counter() - t0  # the Python side of the stage; the rest is the GPU
      Device['QCOM'].synchronize()
      self.rp_time = time.perf_counter() - t0
    else:
      for i, key in enumerate(self.vision_input_names):
        np.copyto(self.frames[i], np.frombuffer(bufs[key].data, dtype=np.uint8, count=self.frame_copy_size))
    for i, key in enumerate(self.vision_input_names):
      self.npy['tfm'][i] = transforms[key]

    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire_pulse'][0] = 0
    self.npy['desire'][:] = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
    self.prev_desire[:] = inputs['desire_pulse']
    self.npy['traffic_convention'][:] = inputs['traffic_convention']
    self.npy['action_t'][:] = inputs['action_t']

    self.input_device.copy_from(self.input_host)
    self.input_queues['new_img'] = self.run_warp(**self.warp_inputs)
    self.run_model(output_buffers=self.outputs, **self.input_queues)
    if after_enqueue is not None:
      after_enqueue()
    model_output = self.outputs['outputs'].numpy()[0]
    if self.chestnut and not np.all(np.isfinite(model_output)):
      raise RuntimeError("model output not finite")
    outputs_dict = self.parser.parse_outputs(self.slice_outputs(model_output, self.output_slices))

    if SEND_RAW_PRED:
      outputs_dict['raw_pred'] = model_output.copy()
    return outputs_dict

  def warmup(self) -> None:
    dummy_frames = {k: np.zeros(self.src_size if self.rp is not None else self.frame_copy_size, dtype=np.uint8) for k in self.vision_input_names}
    eye = np.eye(3, dtype=np.float32)
    dims = {'desire_pulse': ModelConstants.DESIRE_LEN, 'traffic_convention': 2, 'action_t': 2}
    self.run(dummy_frames, dict.fromkeys(self.vision_input_names, eye), {k: np.zeros(v, dtype=np.float32) for k, v in dims.items()})
    self.packed_input[:] = 0
    if self.rp is not None:
      self._src_tensors.clear()
    for key in self.state_pairs:
      self.input_queues[key].assign(0).realize()
    self.prev_desire[:] = 0


def main(demo=False):
  cloudlog.warning("modeld init")

  CHESTNUT = chestnut_present() and chestnut_compiled()
  if CHESTNUT:
    os.environ['HCQDEV_WAIT_TIMEOUT_MS'] = '3000'
  params = Params()
  params.put_bool("ChestnutLoading", CHESTNUT)
  params.remove("ChestnutActive")

  config_realtime_process(7, 54)

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_NARROW_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_NARROW_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_NARROW_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  st = time.monotonic()
  cloudlog.warning("loading model")
  model = None
  if CHESTNUT:
    big_model = None
    def load_big():
      nonlocal big_model
      try:
        m = ModelState(vipc_client_main.width, vipc_client_main.height, True,
                       reproject=bool(REPROJECT_C4) and use_extra_client and (vipc_client_main.width, vipc_client_main.height) == (1928, 1208))
        m.warmup()
        big_model = m
      except Exception:
        cloudlog.exception("big model load failed")
    loader = threading.Thread(target=load_big, daemon=True)
    loader.start()
    loader.join(BIG_MODEL_TIMEOUT)
    model = big_model
    params.put_bool("ChestnutActive", model is not None)

  small_model = ModelState(vipc_client_main.width, vipc_client_main.height, False) if model is None or CHESTNUT else None
  if model is None:
    model = small_model
  params.put_bool("ChestnutLoading", False)
  cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s, modeld starting")

  # messaging
  pub_socks = ["modelV2", "drivingModelData", "cameraOdometry"] + (["chestnutGpuState"] if CHESTNUT else [])
  pm = PubMaster(pub_socks)
  sm = SubMaster(["deviceState", "carState", "narrowRoadCameraState", "wideRoadCameraState", "extrinsicsCalibration", "driverMonitoringState", "carControl", "lateralDelay"])
  stage_times: list[float] = []; exec_times: list[float] = []; meter_times: list[float] = []; enq_times: list[float] = []

  def sys_stats() -> dict:
    """Core clock, GPU load and this thread's context switches: for the on-road 4.4 vs 7.9 ms stage mystery."""
    def rd(p):
      try:
        return open(p).read()
      except OSError:
        return ''
    st = rd('/proc/thread-self/status')
    sw = {k: int(st.split(k + ':')[1].split()[0]) for k in ('voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches') if k + ':' in st}
    return dict(cpu_mhz=int(rd('/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq').strip() or 0) // 1000,
                gpu_busy=rd('/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage').strip(), vcsw=sw.get('voluntary_ctxt_switches'), nivcsw=sw.get('nonvoluntary_ctxt_switches'))

  publish_state = PublishState()
  params = Params()
  chestnut_state = ChestnutGpuState(pm, model.chestnut) if CHESTNUT else None

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  extrinsics_calibration_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()

  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("modeld got CarParams: %s", CP.brand)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  # TODO Move smooth seconds to action function
  long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
  prev_action = log.ModelDataV2.Action()

  DH = DesireHelper()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                         extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["narrowRoadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    lat_delay = sm["lateralDelay"].lateralDelay + LAT_SMOOTH_SECONDS
    if sm.updated["extrinsicsCalibration"] and sm.seen['narrowRoadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["extrinsicsCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[("mici", "os04c10") if model.rp is not None else (str(sm['deviceState'].deviceType), str(sm['narrowRoadCameraState'].sensor))]
      main_intrinsics = dc.wide_road.intrinsics if main_wide_camera else dc.narrow_road.intrinsics
      model_transform_main = get_warp_matrix(device_from_calib_euler, main_intrinsics, False).astype(np.float32)
      has_wide_camera = use_extra_client or main_wide_camera
      extra_intrinsics = dc.wide_road.intrinsics if has_wide_camera else dc.narrow_road.intrinsics
      model_transform_extra = get_warp_matrix(device_from_calib_euler, extra_intrinsics, True).astype(np.float32)
      extrinsics_calibration_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
    frame_delay = DT_MDL # compensate for time passed since the frame was captured: current_time - timestamp_eof is 50ms on average
    action_delay = DT_MDL / 2 # middle of the interval between model output (current state) and next frame (expected state)
    lat_action_t = lat_delay + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay
    inputs: dict[str, np.ndarray] = {
      'desire_pulse': vec_desire,
      'traffic_convention': traffic_convention,
      'action_t': np.array([lat_action_t, long_action_t], dtype=np.float32),
    }
    if model.rp is not None:
      # match the wide surround to the narrow inset: measured in the seam ring of these frames, the sensors' exposure
      # settings as the fallback when the ring is unusable
      ncs, wcs = sm['narrowRoadCameraState'], sm['wideRoadCameraState']
      g = RC.exposure_gain(ncs.gain * ncs.integLines, wcs.gain * wcs.integLines) if sm.seen['wideRoadCameraState'] else 1.0
      mt0 = time.perf_counter()
      inputs['reproj_match'] = model.meter.update(np.frombuffer(bufs['big_img'].data, dtype=np.uint8), np.frombuffer(bufs['img'].data, dtype=np.uint8), g)
      meter_times.append(time.perf_counter() - mt0)

    mt1 = time.perf_counter()
    try:
      send_chestnut = (chestnut_state is not None and
                       run_count % round(ModelConstants.MODEL_RUN_FREQ / SERVICE_LIST['chestnutGpuState'].frequency) == 0)
      model_output = model.run(bufs, transforms, inputs, chestnut_state.send if send_chestnut else None)
    except Exception:
      if not params.get_bool("ChestnutActive"):
        raise
      # fallback to small model
      cloudlog.exception("big model failed, fall back to small")
      params.put_bool("ChestnutActive", False)
      model = small_model
      if chestnut_state is not None:
        chestnut_state.big = False
      run_count = 0
      model_output = None
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1
    exec_times.append(model_execution_time); stage_times.append(model.rp_time if model.rp is not None else 0.0); enq_times.append(model.rp_enqueue if model.rp is not None else 0.0)
    if len(exec_times) % 20 == 0 and model.rp is not None and inputs.get('reproj_match'):
      # live fit metrics for the on-screen debug view (ui reads this file); once a second
      m4 = inputs['reproj_match']; rf = model.refiner
      live = dict(model_ms=round(float(np.median(exec_times[-20:])) * 1e3, 1), stage_ms=round(float(np.median(stage_times[-20:])) * 1e3, 2),
                  stage_enq_ms=round(float(np.median(enq_times[-20:])) * 1e3, 2), **sys_stats(),
                  meter_ms=round(float(np.median(meter_times[-20:])) * 1e3, 2) if meter_times else None, drops=int(vipc_dropped_frames),
                  gain=round(m4['gain_y'], 3), model_gain=round(float(g), 3), u=round(m4['u_off'], 1), v=round(m4['v_off'], 1), gx=round(m4['gx'], 3), gy=round(m4['gy'], 3),
                  bands=[round(b, 3) for b in m4['bands']], rot_deg=[round(float(x), 3) for x in np.degrees(model.rotation)],
                  residual_deg=[round(float(x), 3) for x in np.degrees(rf.last_residual)] if rf.last_residual is not None else None,
                  steps=rf.steps, acc=rf.n, loading=model.loader is not None, fitted=bool(model.rot_file.get('fitted')), big=model is not small_model)
      try:
        tmp = '/data/reproject_c4/live.json.tmp'; json.dump(live, open(tmp, 'w')); os.replace(tmp, '/data/reproject_c4/live.json')
      except OSError:
        pass
    if len(exec_times) % 200 == 0:
      q = lambda t: f"{np.median(t[-200:]) * 1e3:.2f}/{np.percentile(t[-200:], 95) * 1e3:.2f}"
      m4 = inputs.get('reproj_match')
      print(f"run {len(exec_times)}: modelExecutionTime median/p95 {q(exec_times)} ms, of which reprojection {q(stage_times)} ms (enqueue {q(enq_times)}); {sys_stats()}; seam meter {q(meter_times) if meter_times else '-'} ms; "
            f"gain {m4['gain_y']:.3f} U {m4['u_off']:+.1f} V {m4['v_off']:+.1f} grad {m4['gx']:+.3f},{m4['gy']:+.3f} bands {np.round(m4['bands'], 3)} (model {g:.3f})"
            if m4 else f"run {len(exec_times)}: modelExecutionTime median/p95 {q(exec_times)} ms", flush=True)

    if model_output is not None and model.rp is not None:
      if REPROJECT_C4_REFINE:
        model.refine(model_output, v_ego)
      if len(exec_times) % 20 == 0:
        model.poll_fit()

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')

      action = get_action_from_model(model_output, prev_action, lat_action_t, long_action_t, v_ego)
      prev_action = action
      fill_model_msg(modelv2_send, model_output, action,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, extrinsics_calibration_seen)
      modelv2_send.modelV2.big = model.chestnut

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction

      fill_driving_model_data(drivingdata_send, modelv2_send)
      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, extrinsics_calibration_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
    last_vipc_frame_id = meta_main.frame_id

if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
