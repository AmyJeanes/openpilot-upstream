// gta5op_core.dll: the openpilot camera, vehicle state and controls, run once per game frame by the loader.
#include <winsock2.h>
#include <windows.h>
#include <share.h>
#include <Xinput.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

#include "capture.h"
#include "core_api.h"
#include "natives.h"
#include "net.h"
#include "present_hook.h"

#pragma comment(lib, "xinput.lib")

shv::Api shv::api;

namespace {

using namespace nat;

constexpr float PI = 3.14159265f;
constexpr float DEG = PI / 180.0f;

struct Config {
  std::string bridge;  // host:port; empty reads BRIDGE_FILE, which the bridge writes
  float vfov = 76.0f;
  bool lens = true;
  float mountHeight = 1.22f;   // above the ground, as the sim's calibration assumes
  float mountForward = NAN;    // from the vehicle origin; NAN = a quarter of the way to the front
  float curvGain = 1.55f;  // initial path curvature (1/m) per unit of steer bias, learned per car while driving
  float latKi = 1.0f;     // 1/s, integral on the yaw rate error, in units of the feed-forward
  // with no throttle the game slows a car hard, about -(coastAccel + coastPerSpeed * v); steertest with throttle=0 measures it
  float coastAccel = 3.1f;     // m/s^2
  float coastPerSpeed = 0.04f;  // m/s^2 per m/s
  float brakeGain = 8.0f;      // m/s^2 at full brake beyond coasting, roughly
  int keyEngage = VK_F6;
  int keyLeft = VK_LEFT;
  int keyRight = VK_RIGHT;
  int keySpeedUp = VK_UP;
  int keySpeedDown = VK_DOWN;
  bool interleave = false;  // render the openpilot camera only on the frames it captures, the player's camera otherwise
  int interleaveLag = 0;    // game frames from a camera switch to the frame it renders in
  bool presentHook = true;  // when interleaving, take frames from the game's presents and keep them off screen
  // with the present hook, a render for each openpilot camera, at these vertical fields of view: about the road lens's
  // own pixel scale, and enough to fill the wide lens
  bool splitViews = false;
  float roadVfov = 30.0f, wideVfov = 116.0f;
  float wideDelay = 0.025f;  // s from the road view to the wide view; 0 is the next frame
};

const char *BRIDGE_FILE = "C:\\Users\\Public\\gta5op-bridge.txt";

std::wstring g_dir;
Config g_cfg;
Link g_link;
Capture g_capture;
bool g_captureFailed = false;
enum class Hook { Off, On, Failed } g_hook = Hook::Off;
std::mutex g_logMutex;

std::mutex g_stateMutex;
std::string g_state = "{}";  // latest vehicle state as JSON, sent with each frame

std::atomic<int> g_engagePresses{0}, g_leftPresses{0}, g_rightPresses{0};

// cruise speed buttons: a press, then repeats while held, as a car's stalk steps the set speed
struct HeldKey {
  int presses = 0, presses5 = 0;
  bool down = false;
  double nextRepeat = 0;
} g_speedUp, g_speedDown;

struct VehicleInfo {
  Vehicle handle = 0;
  float wheelBase = 2.7f;
  float mountX = 0, mountY = 0, mountZ = 1.0f;
  int wheelLf = -1;
} g_veh;

struct Motion {
  bool valid = false;
  float v = 0, aMeas = 0, yawRate = 0, heading = 0, pitch = 0, roll = 0;
  float grade = 0, bank = 0;  // radians: nose up, right side down
  Vector3 pos{}, rotVel{}, steerBone{};
  int resets = 0;
} g_m;

struct Driver {
  float steer = 0;  // left positive
  bool gas = false, brake = false;
} g_user;

struct Control {
  bool active = false;
  float curvature = 0, accel = 0;  // left-positive 1/m, m/s^2
  double t = -1e9;
  float latI = 0, lonI = 0;
  bool braking = false, wasLive = false, holding = false;
  float steerOut = 0, throttleOut = 0, brakeOut = 0;
} g_ctl;

// The path curvature per unit of steer bias: the bias sets a wheel angle, which sets the curvature whatever the speed, by
// an amount that differs between cars with their steering lock and wheelbase. A least-squares fit of the measured
// curvature against the bias applied a moment before, forgetting over several seconds.
struct CurvGain {
  float gain = 1.55f;
  double num = 0, den = 0, nextLog = 0;
  std::deque<std::pair<double, float>> applied;  // time, bias
} g_curvGain;

double g_latLogUntil = 0;  // logs the steering loop each frame until then

// open-loop steering check: holds a steer bias and logs the motion it produces
struct SteerTest {
  double until = 0, nextLog = 0;
  float bias = 0, throttle = 0;
} g_test;

Cam g_cam = 0;  // the openpilot camera for both views, also standing for the set while it exists
Cam g_cams[3] = {};  // by view: both, road, wide, each at its own field of view
int g_activeView = 0;
bool g_rendering = false;  // whether the game renders the script camera rather than its own
double g_nextOp = 0;       // when interleaving, the time of the next openpilot camera frame
uint64_t g_frameViews = 0;  // recent frames' camera, 4 bits each, newest lowest: 0 the player's, else the view plus one
double g_wideAt = -1;       // when to render the wide view that completes a road view, or -1
int g_ticks = 0, g_opCount = 0;
double g_statsT = 0;
float g_camPitch = 0, g_camYaw = 0;  // degrees, for checks against known rotations
int g_indicator = 0;  // 0 off, 1 left, 2 right
bool g_engaged = false;

// openpilot's set speed, from the bridge, for the speed readout
struct Hud {
  float setKph = 0;
  bool metric = false, engaged = false;
  double t = -1e9;
} g_hud;

struct Setup {
  int step = 0;  // 0 idle, 1 model loading, 2 waiting for the world to load, 3 placing on the road
  float x = 0, y = 0, z = 0, speed = 0;
  int minLanes = 0;  // lanes in the direction of travel, e.g. 3 for a freeway
  Hash model = 0;
  double t = 0;
} g_setup;

// a test car placed ahead in the lane, whose true range the state reports for checking openpilot's lead estimate
struct Lead {
  Vehicle veh = 0;
  Ped driver = 0;
  Hash model = 0;
  float dist = 30, speed = 0, rearY = 0;  // rearY: the model's rear, from its origin
  bool loading = false;
  double t = 0;
} g_lead;
bool g_noTraffic = false;
double g_testGasUntil = 0;  // the gas command presses the pedal until then
bool LeadTruth(float &ahead, float &left, float &speed);
float CameraHeight(double now);
float VehicleAhead(double now);

void Log(const std::string &msg) {
  std::lock_guard lk(g_logMutex);
  // shared, so the log can be followed while the game runs
  FILE *f = _wfsopen((g_dir + L"\\gta5op.log").c_str(), L"a", _SH_DENYNO);
  if (f) {
    SYSTEMTIME t;
    GetLocalTime(&t);
    fprintf(f, "[%02d:%02d:%02d.%03d] core: %s\n", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds, msg.c_str());
    fclose(f);
  }
}

std::string IniPath() { return std::filesystem::path(g_dir + L"\\gta5op.ini").string(); }

void ReadConfig() {
  std::string ini = IniPath();
  auto str = [&](const char *key, const std::string &def) {
    char buf[256];
    GetPrivateProfileStringA("gta5op", key, def.c_str(), buf, sizeof(buf), ini.c_str());
    return std::string(buf);
  };
  auto num = [&](const char *key, float def) {
    std::string s = str(key, "");
    return s.empty() ? def : static_cast<float>(atof(s.c_str()));
  };
  auto key = [&](const char *k, int def) {
    std::string s = str(k, "");
    return s.empty() ? def : static_cast<int>(strtol(s.c_str(), nullptr, 0));
  };
  Config c;
  c.bridge = str("bridge", "");
  c.vfov = num("vfov", c.vfov);
  c.lens = num("lens", 1) != 0;
  c.mountHeight = num("mount_height", c.mountHeight);
  c.mountForward = num("mount_forward", NAN);
  c.curvGain = num("curv_gain", c.curvGain);
  c.latKi = num("lat_ki", c.latKi);
  c.coastAccel = num("coast_accel", c.coastAccel);
  c.coastPerSpeed = num("coast_per_speed", c.coastPerSpeed);
  c.brakeGain = num("brake_gain", c.brakeGain);
  c.keyEngage = key("key_engage", c.keyEngage);
  c.keyLeft = key("key_left", c.keyLeft);
  c.keyRight = key("key_right", c.keyRight);
  c.keySpeedUp = key("key_speed_up", c.keySpeedUp);
  c.keySpeedDown = key("key_speed_down", c.keySpeedDown);
  c.interleave = num("interleave", 0) != 0;
  c.interleaveLag = std::clamp(key("interleave_lag", 0), 0, 8);
  c.presentHook = num("present_hook", 1) != 0;
  c.splitViews = num("split_views", 0) != 0;
  c.roadVfov = num("road_vfov", c.roadVfov);
  c.wideVfov = num("wide_vfov", c.wideVfov);
  g_cfg = c;
}

std::string BridgeAddress() {
  if (!g_cfg.bridge.empty()) return g_cfg.bridge;
  std::ifstream f(BRIDGE_FILE);
  std::string addr;
  std::getline(f, addr);
  while (!addr.empty() && isspace(static_cast<unsigned char>(addr.back()))) addr.pop_back();
  return addr;
}

HWND FindGameWindow() {
  struct Search {
    DWORD pid;
    HWND best;
    long area;
  } s{GetCurrentProcessId(), nullptr, 0};
  EnumWindows([](HWND h, LPARAM p) -> BOOL {
    auto *s = reinterpret_cast<Search *>(p);
    DWORD pid = 0;
    GetWindowThreadProcessId(h, &pid);
    RECT r;
    if (pid == s->pid && IsWindowVisible(h) && GetClientRect(h, &r)) {
      long area = (r.right - r.left) * (r.bottom - r.top);
      if (area > s->area) {
        s->best = h;
        s->area = area;
      }
    }
    return TRUE;
  }, reinterpret_cast<LPARAM>(&s));
  return s.best;
}

std::string Num(double v) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%.6g", std::isfinite(v) ? v : 0.0);
  return buf;
}

std::string Vec(const Vector3 &v) { return "[" + Num(v.x) + "," + Num(v.y) + "," + Num(v.z) + "]"; }

float WrapDeg(float d) {
  while (d > 180) d -= 360;
  while (d < -180) d += 360;
  return d;
}

// *** camera ***

void AttachCamera() {
  for (Cam c : g_cams) HARD_ATTACH_CAM_TO_ENTITY(c, g_veh.handle, g_camPitch, 0.0f, g_camYaw, g_veh.mountX, g_veh.mountY, g_veh.mountZ, TRUE);
}

// A camera per view rather than one whose field of view changes: the game can apply a new field of view a frame late,
// even one set frames ahead, while switching the active camera takes effect on the frame.
void ActivateCamera() {
  const float fovs[3] = {g_cfg.vfov, g_cfg.roadVfov, g_cfg.wideVfov};
  for (int i = 0; i < 3; i++) {
    g_cams[i] = CREATE_CAM("DEFAULT_SCRIPTED_CAMERA", FALSE);
    SET_CAM_FOV(g_cams[i], fovs[i]);
    SET_CAM_NEAR_CLIP(g_cams[i], 0.05f);
  }
  g_cam = g_cams[HOOK_BOTH];
  AttachCamera();
  SET_CAM_ACTIVE(g_cam, TRUE);
  g_activeView = HOOK_BOTH;
  g_rendering = false;
  g_frameViews = 0;
  g_wideAt = -1;
  Log(std::string("camera on") + (g_cfg.interleave ? ", interleaved" : ""));
}

void RenderCamera(bool on) {
  if (on != g_rendering) RENDER_SCRIPT_CAMS(on, FALSE, 0, TRUE, FALSE, 0);
  g_rendering = on;
}

// Picks the frames to render the openpilot camera in, and returns the view the frame being drawn shows (a HOOK_ view), or
// -1 for the player's camera. Interleaved, that's one frame at each 20 Hz capture time, or with split views a road view
// then a wide view half a period later, so the player's frames are evenly spaced; the rest are the player's camera.
int UpdateCameraFrame(double now, float dt, bool split) {
  int view = HOOK_BOTH;
  if (g_cfg.interleave) {
    double t = now + 0.5 * dt;
    bool op = t >= g_nextOp;
    if (op) {
      view = split ? HOOK_ROAD : HOOK_BOTH;
      if (split) g_wideAt = t + g_cfg.wideDelay;
      g_nextOp = std::max(g_nextOp + 0.05, t);
    } else if (g_wideAt > 0 && t >= g_wideAt) {
      view = HOOK_WIDE;
      g_wideAt = -1;
    } else {
      view = -1;
    }
    g_ticks++;
    g_opCount += view >= 0;
    if (now - g_statsT > 10) {
      if (g_statsT) Log("interleave: " + std::to_string(g_ticks) + " ticks, " + std::to_string(g_opCount) + " openpilot frames in 10 s");
      g_ticks = g_opCount = 0;
      g_statsT = now;
    }
  }
  if (view >= 0 && view != g_activeView) {
    SET_CAM_ACTIVE(g_cams[g_activeView], FALSE);
    SET_CAM_ACTIVE(g_cams[view], TRUE);
    g_activeView = view;
  }
  RenderCamera(view >= 0);
  g_frameViews = (g_frameViews << 4) | uint64_t(view + 1);
  return int((g_frameViews >> (4 * g_cfg.interleaveLag)) & 0xF) - 1;
}

void ReleaseCamera() {
  if (!g_cam) return;
  RENDER_SCRIPT_CAMS(FALSE, FALSE, 0, TRUE, FALSE, 0);
  g_rendering = false;
  for (Cam &c : g_cams) {
    if (c && DOES_CAM_EXIST(c)) DESTROY_CAM(c, FALSE);
    c = 0;
  }
  g_cam = 0;
  Log("camera off");
}

// *** vehicle ***

std::string ModelKey(Vehicle v) {
  char key[16];
  snprintf(key, sizeof(key), "0x%08X", GET_ENTITY_MODEL(v));
  return key;
}

void ReleaseControls() {
  if (g_veh.handle && DOES_ENTITY_EXIST(g_veh.handle)) {
    if (g_ctl.holding) SET_VEHICLE_HANDBRAKE(g_veh.handle, FALSE);
    SET_VEHICLE_INDICATOR_LIGHTS(g_veh.handle, 0, FALSE);
    SET_VEHICLE_INDICATOR_LIGHTS(g_veh.handle, 1, FALSE);
  }
  g_ctl.holding = false;
}

void OnVehicleChanged(Vehicle v) {
  ReleaseControls();
  ReleaseCamera();
  g_veh = VehicleInfo{};
  g_veh.handle = v;
  g_curvGain = CurvGain{};
  g_curvGain.gain = g_cfg.curvGain;
  int resets = g_m.resets;
  g_m = Motion{};
  g_m.resets = resets + 1;
  g_indicator = 0;
  if (!v) return;
  Vector3 mn{}, mx{};
  GET_MODEL_DIMENSIONS(GET_ENTITY_MODEL(v), &mn, &mx);
  g_veh.mountX = 0;
  g_veh.mountY = std::isnan(g_cfg.mountForward) ? 0.25f * mx.y : g_cfg.mountForward;
  g_veh.mountZ = mn.z + g_cfg.mountHeight;
  // behind the windscreen, as a comma device is mounted, whatever the cabin's height; the bone sits low on the glass
  int ws = GET_ENTITY_BONE_INDEX_BY_NAME(v, "windscreen");
  if (ws >= 0 && std::isnan(g_cfg.mountForward)) {
    Vector3 w = GET_WORLD_POSITION_OF_ENTITY_BONE(v, ws);
    Vector3 l = GET_OFFSET_FROM_ENTITY_GIVEN_WORLD_COORDS(v, w.x, w.y, w.z);
    g_veh.mountY = l.y;
    // and at least as high as a comma device sits below the roof (a Model 3 calibrates to 1.20-1.23 m, 0.29 m below
    // its roof): the driving model judges scale from the camera's height, reading speeds 20% fast from 9 cm too low
    g_veh.mountZ = std::max(l.z + 0.08f, mx.z - 0.29f);
  }
  // a mount set with the camera command for this model
  char saved[32];
  GetPrivateProfileStringA("mount_forward", ModelKey(v).c_str(), "", saved, sizeof(saved), IniPath().c_str());
  if (saved[0]) g_veh.mountY = static_cast<float>(atof(saved));
  int lf = GET_ENTITY_BONE_INDEX_BY_NAME(v, "wheel_lf"), lr = GET_ENTITY_BONE_INDEX_BY_NAME(v, "wheel_lr");
  g_veh.wheelLf = lf;
  if (lf >= 0 && lr >= 0) {
    Vector3 a = GET_WORLD_POSITION_OF_ENTITY_BONE(v, lf), b = GET_WORLD_POSITION_OF_ENTITY_BONE(v, lr);
    Vector3 la = GET_OFFSET_FROM_ENTITY_GIVEN_WORLD_COORDS(v, a.x, a.y, a.z), lb = GET_OFFSET_FROM_ENTITY_GIVEN_WORLD_COORDS(v, b.x, b.y, b.z);
    float wb = la.y - lb.y;
    if (wb > 1.0f && wb < 8.0f) g_veh.wheelBase = wb;
  }
  Log("vehicle " + std::to_string(v) + ": wheelbase " + Num(g_veh.wheelBase) + " m, mount " + Num(g_veh.mountY) + " fwd, " +
      Num(g_veh.mountZ) + " up (model z " + Num(mn.z) + ".." + Num(mx.z) + ")");
}

bool GameFocused() {
  DWORD pid = 0;
  GetWindowThreadProcessId(GetForegroundWindow(), &pid);
  return pid == GetCurrentProcessId();
}

// the driver's own inputs, read from the devices: the game's control values include what the plugin injects
void PollHeldKey(HeldKey &k, int vk, bool focused, double now) {
  bool down = focused && (GetAsyncKeyState(vk) & 0x8000) != 0;
  // with shift, a step to the next multiple of 5, as a car's stalk pushed past its detent
  int &presses = (GetAsyncKeyState(VK_SHIFT) & 0x8000) ? k.presses5 : k.presses;
  if (down && !k.down) {
    presses++;
    k.nextRepeat = now + 0.5;
  } else if (down && now >= k.nextRepeat) {
    presses++;
    k.nextRepeat = now + 0.15;
  }
  k.down = down;
}

void ReadDriver(double now) {
  Driver d;
  bool focused = GameFocused();
  PollHeldKey(g_speedUp, g_cfg.keySpeedUp, focused, now);
  PollHeldKey(g_speedDown, g_cfg.keySpeedDown, focused, now);
  if (focused) {
    auto down = [](int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; };
    d.gas = down('W');
    d.brake = down('S');
    d.steer = (down('A') ? 1.0f : 0.0f) - (down('D') ? 1.0f : 0.0f);
  }
  XINPUT_STATE xs;
  for (DWORD i = 0; i < XUSER_MAX_COUNT; i++) {
    if (XInputGetState(i, &xs) != ERROR_SUCCESS) continue;
    d.gas |= xs.Gamepad.bRightTrigger > 40;
    d.brake |= xs.Gamepad.bLeftTrigger > 40;
    float lx = xs.Gamepad.sThumbLX / 32767.0f;
    if (std::fabs(lx) > 0.25f && std::fabs(d.steer) < std::fabs(lx)) d.steer = -lx;
  }
  if (now < g_testGasUntil) d.gas = true;
  g_user = d;
}

void Measure(float dt) {
  Vehicle v = g_veh.handle;
  Vector3 sv = GET_ENTITY_SPEED_VECTOR(v, TRUE);
  float speed = sv.y;
  float heading = GET_ENTITY_HEADING(v);
  Vector3 pos = GET_ENTITY_COORDS(v, TRUE);
  Vector3 rot = GET_ENTITY_ROTATION(v, 2);
  if (g_m.valid && dt > 0) {
    float dx = pos.x - g_m.pos.x, dy = pos.y - g_m.pos.y, dz = pos.z - g_m.pos.z;
    if (std::sqrt(dx * dx + dy * dy + dz * dz) > 10.0f + 3.0f * std::fabs(speed) * dt) {
      // teleported: no sensor sees that, so restart the derived signals
      g_m.resets++;
      g_m.valid = false;
    }
  }
  if (g_m.valid && dt > 0) {
    // the physics' own yaw rate (left-positive, like openpilot's); differencing the heading is noisy per frame
    g_m.yawRate = GET_ENTITY_ROTATION_VELOCITY(v).z;
    float raw = (speed - g_m.v) / dt;
    float alpha = 1.0f - std::exp(-dt / 0.1f);
    if (std::fabs(raw) < 30) g_m.aMeas = std::clamp(g_m.aMeas + alpha * (raw - g_m.aMeas), -15.0f, 15.0f);
  } else {
    g_m.yawRate = 0;
    g_m.aMeas = 0;
  }
  g_m.v = speed;
  g_m.heading = heading;
  g_m.pos = pos;
  g_m.pitch = rot.x;
  g_m.roll = rot.y;
  // from the car's own axes, which leaves no doubt about the rotation's sign conventions
  Vector3 front = GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(v, 0.0f, 1.0f, 0.0f), right = GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(v, 1.0f, 0.0f, 0.0f);
  g_m.grade = std::asin(std::clamp(front.z - pos.z, -1.0f, 1.0f));
  g_m.bank = std::asin(std::clamp(pos.z - right.z, -1.0f, 1.0f));
  g_m.rotVel = GET_ENTITY_ROTATION_VELOCITY(v);
  if (g_veh.wheelLf >= 0) g_m.steerBone = GET_ENTITY_BONE_OBJECT_ROTATION(v, g_veh.wheelLf);
  g_m.valid = true;
}

// The throttle that adds this much acceleration over coasting. The game ignores throttle below about 0.15 and gives most
// of its drive by 0.5 (measured with steertest on a sedan: 0.3 adds about 3.5 m/s^2, full throttle about 8).
float ThrottleFor(float drive) {
  static const float THROTTLE[] = {0.15f, 0.3f, 0.5f, 0.75f, 1.0f};
  static const float DRIVE[] = {0.0f, 3.5f, 6.3f, 7.5f, 8.1f};
  if (drive <= 0) return 0;
  for (int i = 1; i < 5; i++)
    if (drive < DRIVE[i]) return THROTTLE[i - 1] + (THROTTLE[i] - THROTTLE[i - 1]) * (drive - DRIVE[i - 1]) / (DRIVE[i] - DRIVE[i - 1]);
  return 1.0f;
}

void UpdateCurvGain(float dt, double now) {
  CurvGain &g = g_curvGain;
  constexpr double LAG = 0.15, TAU = 8.0;  // the game's steering response lag, s; forgetting time, s
  while (g.applied.size() > 1 && g.applied[1].first <= now - LAG) g.applied.pop_front();
  if (g.applied.empty() || g.applied.front().first > now - LAG || g_m.v < 5.0f || g_user.steer != 0) return;
  double bias = g.applied.front().second, a = std::min(1.0, dt / TAU);
  g.num += a * (g_m.yawRate / g_m.v * bias - g.num);
  g.den += a * (bias * bias - g.den);
  // highway steering is a bias of a few thousandths
  if (g.den > 0.001 * 0.001) g.gain = static_cast<float>(std::clamp(g.num / g.den, 0.3, 6.0));
  if (now >= g.nextLog) {
    g.nextLog = now + 10;
    Log("curvature gain " + Num(g.gain));
  }
}

void ApplyControls(float dt, double now) {
  Vehicle veh = g_veh.handle;
  if (now < g_test.until) {
    SET_VEHICLE_STEER_BIAS(veh, g_test.bias);
    SET_CONTROL_VALUE_NEXT_FRAME(0, INPUT_VEH_ACCELERATE, g_test.throttle);
    if (now >= g_test.nextLog) {
      g_test.nextLog = now + 0.1;
      float curv = std::fabs(g_m.v) > 1 ? g_m.yawRate / g_m.v : 0;
      Log("steertest bias " + Num(g_test.bias) + " v " + Num(g_m.v) + " yawRate " + Num(g_m.yawRate) + " curvature " + Num(curv) +
          " wheel angle " + Num(std::atan(curv * g_veh.wheelBase) / DEG) + " deg, bone " + Vec(g_m.steerBone));
    }
    return;
  }
  bool live = g_ctl.active && now - g_ctl.t < 0.3;
  if (!live) {
    if (g_ctl.wasLive) ReleaseControls();
    g_ctl.wasLive = false;
    g_curvGain.applied.clear();
    g_ctl.latI = g_ctl.lonI = 0;
    g_ctl.braking = false;
    g_ctl.steerOut = g_ctl.throttleOut = g_ctl.brakeOut = 0;
    return;
  }
  g_ctl.wasLive = true;
  float speed = g_m.v;

  // lateral: the game's steer bias sets a path curvature roughly proportional to it, so feed forward the bias for the
  // curvature, and integrate the yaw rate error
  float v = std::max(speed, 3.0f);
  float yawTarget = g_ctl.curvature * v;
  UpdateCurvGain(dt, now);
  float gain = g_curvGain.gain;
  g_ctl.latI = std::clamp(g_ctl.latI + g_cfg.latKi * (yawTarget - g_m.yawRate) / (gain * v) * dt, -0.05f, 0.05f);
  if (speed < 3.0f) g_ctl.latI *= std::max(0.0f, 1.0f - dt);
  float steer = std::clamp(g_ctl.curvature / gain + g_ctl.latI + 0.15f * g_user.steer, -0.3f, 0.3f);
  g_curvGain.applied.emplace_back(now, steer);
  if (now < g_latLogUntil)
    Log("lat v " + Num(speed) + " curv " + Num(g_ctl.curvature) + " yawTarget " + Num(yawTarget) + " yawRate " + Num(g_m.yawRate) +
        " steer " + Num(steer) + " latI " + Num(g_ctl.latI) + " gain " + Num(gain) + " dt " + Num(dt) + " cmdAge " + Num(now - g_ctl.t));
  // sets the wheel angle directly; the steering control input goes through the game's smoothing and speed scaling
  SET_VEHICLE_STEER_BIAS(g_veh.handle, steer);

  // longitudinal, as the Slow Roads bridge: requested acceleration plus a bounded integral on the measured one
  float accel = g_ctl.accel;
  bool stopped = std::fabs(speed) < 0.3f;
  if (stopped && accel <= 0) {
    g_ctl.lonI = 0;  // holding at a stop can't decelerate further; integrating that would block pulling away
  } else {
    g_ctl.lonI = std::clamp(g_ctl.lonI + 2.0f * (accel - g_m.aMeas) * dt, -1.5f, 1.5f);
    if (stopped) g_ctl.lonI = std::max(0.0f, g_ctl.lonI);
  }
  if (g_user.gas) g_ctl.lonI = 0;
  // a real Model 3 delivers the requested acceleration on any grade (openpilot sends it open loop), so add what gravity
  // takes away; the coasting model and throttle table are flat-road ones
  float grade = 9.81f * std::sin(g_m.grade);
  float u = std::clamp(accel + g_ctl.lonI, -4.0f, 2.5f) + grade;
  // the game's brake reverses a stopped car, so hold a stop with the handbrake like a car's brake hold
  bool hold = std::fabs(speed) < 0.5f && accel < 0 && !g_user.gas;
  if (hold != g_ctl.holding) SET_VEHICLE_HANDBRAKE(veh, hold);
  g_ctl.holding = hold;
  // throttle gives anything above coasting, including most slowing down; the brake only what's beyond it
  float coast = -(g_cfg.coastAccel + g_cfg.coastPerSpeed * std::fabs(speed));
  g_ctl.braking = !hold && u < coast + (g_ctl.braking ? 0.2f : 0.0f) && !g_user.gas;
  float throttle = 0, brake = 0;
  if (hold) {
  } else if (g_ctl.braking) {
    brake = std::clamp((coast - u) / g_cfg.brakeGain, 0.0f, 1.0f);
  } else {
    throttle = ThrottleFor(u - coast);
  }
  if (g_user.gas) throttle = 1.0f;
  SET_CONTROL_VALUE_NEXT_FRAME(0, INPUT_VEH_ACCELERATE, throttle);
  SET_CONTROL_VALUE_NEXT_FRAME(0, INPUT_VEH_BRAKE, brake);
  g_ctl.steerOut = steer;
  g_ctl.throttleOut = throttle;
  g_ctl.brakeOut = brake;
}

// what a steering angle sensor would show, as the curvature the car follows: the yaw rate's, which the bridge turns into
// an angle through openpilot's own learned vehicle model. Below walking pace that's undefined, so the commanded steering.
float SteerCurvature() {
  if (g_m.v > 2.0f) return g_m.yawRate / g_m.v;
  return g_ctl.wasLive ? g_ctl.steerOut * g_curvGain.gain : 0.0f;
}

void UpdateIndicator() {
  int left = g_leftPresses.exchange(0), right = g_rightPresses.exchange(0);
  int prev = g_indicator;
  if (left) g_indicator = g_indicator == 1 ? 0 : 1;
  if (right) g_indicator = g_indicator == 2 ? 0 : 2;
  if (g_indicator != prev && g_veh.handle) {
    // turnSignal 1 is the left light, 0 the right
    SET_VEHICLE_INDICATOR_LIGHTS(g_veh.handle, 1, g_indicator == 1);
    SET_VEHICLE_INDICATOR_LIGHTS(g_veh.handle, 0, g_indicator == 2);
  }
}

void DrawText(const std::string &text, float x, float y, float scale, int r, int g, int b) {
  SET_TEXT_FONT(4);
  SET_TEXT_SCALE(scale, scale);
  SET_TEXT_COLOUR(r, g, b, 255);
  SET_TEXT_OUTLINE();
  SET_TEXT_WRAP(0.0f, x);
  SET_TEXT_JUSTIFICATION(2);
  BEGIN_TEXT_COMMAND_DISPLAY_TEXT("STRING");
  ADD_TEXT_COMPONENT_SUBSTRING_PLAYER_NAME(text.c_str());
  END_TEXT_COMMAND_DISPLAY_TEXT(0.0f, y, 0);
}

// the car's speed and openpilot's set speed at the right of the screen, green while engaged
void DrawSpeed(double now) {
  bool fresh = now - g_hud.t < 1.0;
  float toUnit = g_hud.metric ? 3.6f : 2.23694f;
  int r = 255, g = 255, b = 255;
  if (fresh && g_hud.engaged) r = 80, g = 220, b = 100;
  DrawText(std::to_string(static_cast<int>(std::lround(g_m.v * toUnit))) + (g_hud.metric ? " km/h" : " mph"), 0.985f, 0.74f, 0.9f, r, g, b);
  std::string set = fresh && g_hud.setKph > 0 ? std::to_string(static_cast<int>(std::lround(g_hud.setKph * (g_hud.metric ? 1.0f : 0.621371f)))) : "-";
  DrawText("SET " + set, 0.985f, 0.795f, 0.55f, 200, 200, 200);
}

// the street the car is on, which the bridge guesses a speed limit from
std::string Street(double now) {
  static std::string street;
  static double next = 0;
  if (now < next) return street;
  next = now + 0.5;
  uint64_t hash = 0, crossing = 0;
  GET_STREET_NAME_AT_COORD(g_m.pos.x, g_m.pos.y, g_m.pos.z, &hash, &crossing);
  const char *name = hash ? GET_STREET_NAME_FROM_HASH_KEY(static_cast<Hash>(hash)) : nullptr;
  street.clear();
  for (const char *c = name; c && *c; c++)
    if (*c != '"' && *c != '\\' && static_cast<unsigned char>(*c) >= 0x20) street += *c;
  return street;
}

void Publish(double now, bool inVehicle) {
  std::ostringstream s;
  s << "{\"t\":" << Num(now) << ",\"inVehicle\":" << (inVehicle ? "true" : "false") << ",\"paused\":" << (IS_PAUSE_MENU_ACTIVE() ? "true" : "false")
    << ",\"engagePresses\":" << g_engagePresses.load() << ",\"speedUpPresses\":" << g_speedUp.presses
    << ",\"speedDownPresses\":" << g_speedDown.presses << ",\"speedUp5Presses\":" << g_speedUp.presses5
    << ",\"speedDown5Presses\":" << g_speedDown.presses5 << ",\"resets\":" << g_m.resets;
  if (inVehicle) {
    s << ",\"vEgo\":" << Num(g_m.v) << ",\"aMeas\":" << Num(g_m.aMeas) << ",\"yawRate\":" << Num(g_m.yawRate)
      << ",\"heading\":" << Num(g_m.heading) << ",\"pitch\":" << Num(g_m.pitch) << ",\"roll\":" << Num(g_m.roll) << ",\"grade\":" << Num(g_m.grade) << ",\"bank\":" << Num(g_m.bank)
      << ",\"wheelBase\":" << Num(g_veh.wheelBase) << ",\"steerCurvature\":" << Num(SteerCurvature()) << ",\"pos\":" << Vec(g_m.pos) << ",\"rotVel\":" << Vec(g_m.rotVel)
      << ",\"steerBone\":" << Vec(g_m.steerBone) << ",\"street\":\"" << Street(now) << "\""
      << ",\"indicator\":" << (g_indicator == 1 ? "\"left\"" : g_indicator == 2 ? "\"right\"" : "null")
      << ",\"user\":{\"steer\":" << Num(g_user.steer) << ",\"gas\":" << (g_user.gas ? "true" : "false") << ",\"brake\":" << (g_user.brake ? "true" : "false") << "}"
      << ",\"out\":{\"steer\":" << Num(g_ctl.steerOut) << ",\"throttle\":" << Num(g_ctl.throttleOut) << ",\"brake\":" << Num(g_ctl.brakeOut)
      << ",\"latI\":" << Num(g_ctl.latI) << ",\"curvGain\":" << Num(g_curvGain.gain) << ",\"lonI\":" << Num(g_ctl.lonI) << ",\"hold\":" << (g_ctl.holding ? "true" : "false") << "}"
      << ",\"camHeight\":" << Num(CameraHeight(now)) << ",\"vehicleAhead\":" << Num(VehicleAhead(now));
    float ahead = 0, left = 0, speed = 0;
    if (LeadTruth(ahead, left, speed)) s << ",\"lead\":{\"ahead\":" << Num(ahead) << ",\"left\":" << Num(left) << ",\"v\":" << Num(speed) << "}";
  }
  s << "}";
  std::lock_guard lk(g_stateMutex);
  g_state = s.str();
}

// *** commands from the bridge ***

void StepSetup(Ped ped, double now) {
  Setup &s = g_setup;
  if (s.step == 1) {
    if (!HAS_MODEL_LOADED(s.model)) {
      if (now - s.t > 10) {
        Log("setup: model didn't load");
        s.step = 0;
      }
      return;
    }
    Vector3 p = GET_ENTITY_COORDS(ped, TRUE);
    float h = GET_ENTITY_HEADING(ped);
    Vehicle v = CREATE_VEHICLE(s.model, p.x, p.y, p.z + 1.0f, h, FALSE, FALSE, FALSE);
    SET_MODEL_AS_NO_LONGER_NEEDED(s.model);
    if (!v) {
      Log("setup: vehicle creation failed");
      s.step = 0;
      return;
    }
    SET_PED_INTO_VEHICLE(ped, v, -1);
    SET_VEHICLE_ENGINE_ON(v, TRUE, TRUE, FALSE);
    s.step = std::isnan(s.x) ? 0 : 2;
    s.t = now;
  }
  Entity e = IS_PED_IN_ANY_VEHICLE(ped, FALSE) ? GET_VEHICLE_PED_IS_IN(ped, FALSE) : ped;
  if (s.step == 2) {
    if (now - s.t < 0.1) {
      SET_ENTITY_COORDS(e, s.x, s.y, s.z, FALSE, FALSE, FALSE, FALSE);
      FREEZE_ENTITY_POSITION(e, TRUE);
    }
    REQUEST_COLLISION_AT_COORD(s.x, s.y, s.z);
    LOAD_ALL_PATH_NODES(TRUE);
    if (now - s.t > 3) s.step = 3;
  } else if (s.step == 3) {
    Vector3 node{};
    float heading = 0;
    FREEZE_ENTITY_POSITION(e, FALSE);
    bool found = false;
    for (int n = 1; n <= 200 && !found; n++) {
      int lanes = 0;
      if (!GET_NTH_CLOSEST_VEHICLE_NODE_WITH_HEADING(s.x, s.y, s.z, n, &node, &heading, &lanes, 1, 3.0f, 0.0f)) break;
      found = lanes >= s.minLanes;
      if (found && s.minLanes) Log("setup: node " + std::to_string(n) + " has " + std::to_string(lanes) + " lanes");
    }
    if (found) {
      SET_ENTITY_COORDS(e, node.x, node.y, node.z + 0.5f, FALSE, FALSE, FALSE, FALSE);
      SET_ENTITY_HEADING(e, heading);
      if (e != ped) {
        SET_VEHICLE_ON_GROUND_PROPERLY(e, 5.0f);
        if (s.speed > 0) SET_VEHICLE_FORWARD_SPEED(e, s.speed);
      }
      Log("setup: placed at " + Num(node.x) + "," + Num(node.y) + "," + Num(node.z) + " heading " + Num(heading));
    } else {
      Log("setup: no road node near the target");
    }
    s.step = 0;
  }
}

void RemoveLead() {
  // the game deletes only entities a script owns
  if (g_lead.driver && DOES_ENTITY_EXIST(g_lead.driver)) {
    SET_ENTITY_AS_MISSION_ENTITY(g_lead.driver, TRUE, TRUE);
    DELETE_PED(&g_lead.driver);
  }
  if (g_lead.veh && DOES_ENTITY_EXIST(g_lead.veh)) {
    SET_ENTITY_AS_MISSION_ENTITY(g_lead.veh, TRUE, TRUE);
    DELETE_VEHICLE(&g_lead.veh);
  }
  g_lead.veh = g_lead.driver = 0;
}

// deletes cars of a model near the player's, other than the player's own: test leads left behind
void ClearModel(Hash model, float radius) {
  if (!g_veh.handle) return;
  Vector3 p = GET_ENTITY_COORDS(g_veh.handle, TRUE);
  for (int i = 0; i < 100; i++) {
    Vehicle v = GET_CLOSEST_VEHICLE(p.x, p.y, p.z, radius, model, 70);
    if (!v || v == g_veh.handle) break;
    SET_ENTITY_AS_MISSION_ENTITY(v, TRUE, TRUE);
    DELETE_VEHICLE(&v);
  }
}

void StepLead(double now) {
  Lead &l = g_lead;
  if (!l.loading || !g_veh.handle) return;
  if (!HAS_MODEL_LOADED(l.model)) {
    if (now - l.t > 10) {
      Log("lead: model didn't load");
      l.loading = false;
    }
    return;
  }
  l.loading = false;
  Vector3 mn{}, mx{};
  GET_MODEL_DIMENSIONS(l.model, &mn, &mx);
  l.rearY = mn.y;
  // its rear about the given distance ahead of the camera, moved onto the road there at the player's offset from the
  // road's nodes, so it's in the same lane where the road curves (the state reports its true range)
  float heading = GET_ENTITY_HEADING(g_veh.handle);
  Vector3 p = GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(g_veh.handle, 0.0f, g_veh.mountY + l.dist - mn.y, 0.0f);
  Vector3 egoPos = GET_ENTITY_COORDS(g_veh.handle, TRUE), egoNode{}, node{};
  float egoNodeHeading = 0, nodeHeading = 0;
  if (GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(egoPos.x, egoPos.y, egoPos.z, &egoNode, &egoNodeHeading, 1, 3.0f, 0.0f) &&
      GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(p.x, p.y, p.z, &node, &nodeHeading, 1, 3.0f, 0.0f)) {
    auto aligned = [heading](float h) { return std::fabs(std::remainder(h - heading, 360.0f)) > 90.0f ? h + 180.0f : h; };
    egoNodeHeading = aligned(egoNodeHeading) * DEG;
    nodeHeading = aligned(nodeHeading) * DEG;
    // headings are counterclockwise from north: right is (cos h, sin h)
    float offset = (egoPos.x - egoNode.x) * std::cos(egoNodeHeading) + (egoPos.y - egoNode.y) * std::sin(egoNodeHeading);
    p = node;
    p.x += offset * std::cos(nodeHeading);
    p.y += offset * std::sin(nodeHeading);
    heading = nodeHeading / DEG;
  }
  l.veh = CREATE_VEHICLE(l.model, p.x, p.y, p.z + 0.5f, heading, FALSE, FALSE, FALSE);
  SET_MODEL_AS_NO_LONGER_NEEDED(l.model);
  if (!l.veh) {
    Log("lead: vehicle creation failed");
    return;
  }
  SET_ENTITY_AS_MISSION_ENTITY(l.veh, TRUE, TRUE);
  SET_VEHICLE_ON_GROUND_PROPERLY(l.veh, 5.0f);
  if (l.speed > 0) {
    l.driver = CREATE_RANDOM_PED_AS_DRIVER(l.veh, TRUE);
    SET_BLOCKING_OF_NON_TEMPORARY_EVENTS(l.driver, TRUE);
    SET_VEHICLE_ENGINE_ON(l.veh, TRUE, TRUE, FALSE);
    SET_VEHICLE_FORWARD_SPEED(l.veh, l.speed);
    TASK_VEHICLE_DRIVE_WANDER(l.driver, l.veh, l.speed, 786603);  // normal driving: keeps to lanes, stops at lights
    SET_VEHICLE_MAX_SPEED(l.veh, l.speed);  // the task alone drives well over its speed
  } else {
    SET_VEHICLE_HANDBRAKE(l.veh, TRUE);
  }
  Log("lead: placed " + Num(l.dist) + " m ahead, speed " + Num(l.speed));
}

// the test lead's rear relative to the camera: ahead, left, and its speed along the ego car's heading
bool LeadTruth(float &ahead, float &left, float &speed) {
  if (!g_lead.veh || !DOES_ENTITY_EXIST(g_lead.veh) || !g_veh.handle) return false;
  Vector3 rear = GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(g_lead.veh, 0.0f, g_lead.rearY, 0.0f);
  Vector3 rel = GET_OFFSET_FROM_ENTITY_GIVEN_WORLD_COORDS(g_veh.handle, rear.x, rear.y, rear.z);
  ahead = rel.y - g_veh.mountY;
  left = -(rel.x - g_veh.mountX);
  float dh = (GET_ENTITY_HEADING(g_lead.veh) - GET_ENTITY_HEADING(g_veh.handle)) * DEG;
  speed = GET_ENTITY_SPEED_VECTOR(g_lead.veh, TRUE).y * std::cos(dh);
  return true;
}

// the range from the camera to the first vehicle straight ahead, within 100 m (0 for none): a check on openpilot's
// lead in ordinary traffic, where the road is straight
float VehicleAhead(double now) {
  static float range = 0;
  static double next = 0;
  if (now < next || !g_veh.handle) return range;
  next = now + 0.1;
  Vector3 a = GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(g_veh.handle, g_veh.mountX, g_veh.mountY + 0.5f, g_veh.mountZ - 0.5f);
  Vector3 b = GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(g_veh.handle, g_veh.mountX, g_veh.mountY + 100.0f, g_veh.mountZ - 0.5f);
  int test = START_EXPENSIVE_SYNCHRONOUS_SHAPE_TEST_LOS_PROBE(a.x, a.y, a.z, b.x, b.y, b.z, 2, g_veh.handle, 7);
  BOOL hit = FALSE;
  Vector3 end{}, normal{};
  Entity e = 0;
  range = 0;
  if (GET_SHAPE_TEST_RESULT(test, &hit, &end, &normal, &e) == 2 && hit) {
    Vector3 rel = GET_OFFSET_FROM_ENTITY_GIVEN_WORLD_COORDS(g_veh.handle, end.x, end.y, end.z);
    range = rel.y - g_veh.mountY;
  }
  return range;
}

// the camera's height above the ground under it
float CameraHeight(double now) {
  static float height = 0;
  static double next = 0;
  if (now < next || !g_veh.handle) return height;
  next = now + 0.5;
  Vector3 c = GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(g_veh.handle, g_veh.mountX, g_veh.mountY, g_veh.mountZ);
  float ground = 0;
  if (GET_GROUND_Z_FOR_3D_COORD(c.x, c.y, c.z, &ground, FALSE, FALSE)) height = c.z - ground;
  return height;
}

void HandleMessage(const Message &m, double now) {
  std::string type = MsgStr(m, "type");
  if (type == "control") {
    g_ctl.active = MsgBool(m, "active");
    g_ctl.curvature = static_cast<float>(MsgNum(m, "curvature"));
    g_ctl.accel = static_cast<float>(MsgNum(m, "accel"));
    g_ctl.t = now;
  } else if (type == "hud") {
    g_hud.setKph = static_cast<float>(MsgNum(m, "setSpeed"));
    g_hud.metric = MsgBool(m, "metric");
    g_hud.engaged = MsgBool(m, "engaged");
    g_hud.t = now;
  } else if (type == "engaged") {
    g_engaged = MsgBool(m, "on");
  } else if (type == "indicatorOff") {
    g_indicator = 0;
    if (g_veh.handle) {
      SET_VEHICLE_INDICATOR_LIGHTS(g_veh.handle, 0, FALSE);
      SET_VEHICLE_INDICATOR_LIGHTS(g_veh.handle, 1, FALSE);
    }
  } else if (type == "camera") {
    g_camPitch = static_cast<float>(MsgNum(m, "pitch", g_camPitch));
    g_camYaw = static_cast<float>(MsgNum(m, "yaw", g_camYaw));
    // mount position for this car, meters: forward of the vehicle origin, and up from the ground
    if (m.count("forward") && g_veh.handle) {
      g_veh.mountY = static_cast<float>(MsgNum(m, "forward"));
      WritePrivateProfileStringA("mount_forward", ModelKey(g_veh.handle).c_str(), Num(g_veh.mountY).c_str(), IniPath().c_str());
    }
    if (m.count("height")) {
      float height = static_cast<float>(MsgNum(m, "height"));
      g_veh.mountZ += height - g_cfg.mountHeight;
      g_cfg.mountHeight = height;
    }
    Log("camera: forward " + Num(g_veh.mountY) + ", up " + Num(g_veh.mountZ) + ", pitch " + Num(g_camPitch) + ", yaw " + Num(g_camYaw));
    if (g_cam) AttachCamera();
  } else if (type == "interleave") {
    g_cfg.interleave = MsgBool(m, "on", g_cfg.interleave);
    g_cfg.interleaveLag = std::clamp(static_cast<int>(MsgNum(m, "lag", g_cfg.interleaveLag)), 0, 8);
    g_cfg.presentHook = MsgBool(m, "hook", g_cfg.presentHook);
    g_cfg.splitViews = MsgBool(m, "split", g_cfg.splitViews);
    g_cfg.wideDelay = static_cast<float>(MsgNum(m, "wide_delay", g_cfg.wideDelay));
    Log("interleave " + std::string(g_cfg.interleave ? "on" : "off") + ", lag " + std::to_string(g_cfg.interleaveLag) +
        (g_cfg.splitViews ? ", split views" : ""));
  } else if (type == "reset") {
    // restarts one part of the openpilot camera pipeline (hook, capture or camera), each recreated next tick
    std::string part = MsgStr(m, "part");
    if (part == "hook" && g_hook == Hook::On) {
      present_hook::Uninstall();
      g_hook = Hook::Off;
      g_capture.Stop();
    } else if (part == "capture") {
      g_capture.Stop();
    } else if (part == "camera") {
      ReleaseCamera();
    }
    Log("reset " + part);
  } else if (type == "engage") {
    g_engagePresses++;  // as if the engage key were pressed
  } else if (type == "trim" && g_veh.handle) {
    if (m.count("interior")) SET_VEHICLE_EXTRA_COLOUR_5(g_veh.handle, static_cast<int>(MsgNum(m, "interior")));
    if (m.count("dashboard")) SET_VEHICLE_EXTRA_COLOUR_6(g_veh.handle, static_cast<int>(MsgNum(m, "dashboard")));
    Log("trim interior " + Num(MsgNum(m, "interior", -1)) + " dashboard " + Num(MsgNum(m, "dashboard", -1)));
  } else if (type == "paint" && g_veh.handle) {
    int r = static_cast<int>(MsgNum(m, "r")), g = static_cast<int>(MsgNum(m, "g")), b = static_cast<int>(MsgNum(m, "b"));
    int paintType = static_cast<int>(MsgNum(m, "finish", 1));  // metallic
    SET_VEHICLE_MOD_KIT(g_veh.handle, 0);
    SET_VEHICLE_MOD_COLOR_1(g_veh.handle, paintType, 0, 0);
    SET_VEHICLE_MOD_COLOR_2(g_veh.handle, paintType, 0);
    SET_VEHICLE_CUSTOM_PRIMARY_COLOUR(g_veh.handle, r, g, b);
    // the secondary colour, which many cars use for trim; the same as the primary unless given
    int sr = static_cast<int>(MsgNum(m, "sr", r)), sg = static_cast<int>(MsgNum(m, "sg", g)), sb = static_cast<int>(MsgNum(m, "sb", b));
    SET_VEHICLE_CUSTOM_SECONDARY_COLOUR(g_veh.handle, sr, sg, sb);
    Log("paint " + std::to_string(r) + "," + std::to_string(g) + "," + std::to_string(b) + " secondary " + std::to_string(sr) + "," +
        std::to_string(sg) + "," + std::to_string(sb) + " finish " + std::to_string(paintType));
  } else if (type == "latlog") {
    g_latLogUntil = now + MsgNum(m, "secs", 10);
  } else if (type == "indicator") {
    (MsgStr(m, "side") == "right" ? g_rightPresses : g_leftPresses)++;  // as if the indicator key were pressed
  } else if (type == "steertest") {
    g_test.bias = static_cast<float>(MsgNum(m, "bias", 0.1));
    g_test.throttle = static_cast<float>(MsgNum(m, "throttle", 0.3));
    g_test.until = now + MsgNum(m, "secs", 2.0);
    g_test.nextLog = now;
  } else if (type == "lead") {
    RemoveLead();
    if (m.count("clear")) ClearModel(GET_HASH_KEY(MsgStr(m, "clear").c_str()), 200.0f);
    if (!MsgBool(m, "remove")) {
      g_lead.dist = static_cast<float>(MsgNum(m, "dist", 30));
      g_lead.speed = static_cast<float>(MsgNum(m, "speed", 0));
      std::string model = MsgStr(m, "model");
      g_lead.model = GET_HASH_KEY(model.empty() ? "sultan" : model.c_str());
      REQUEST_MODEL(g_lead.model);
      g_lead.loading = true;
      g_lead.t = now;
    }
  } else if (type == "leadspeed" && g_lead.driver && DOES_ENTITY_EXIST(g_lead.veh)) {
    // a new speed for the moving lead; near 0 it pulls up and waits, as in a queue
    float v = static_cast<float>(MsgNum(m, "v", g_lead.speed));
    SET_VEHICLE_MAX_SPEED(g_lead.veh, std::max(v, 0.01f));
    SET_DRIVE_TASK_CRUISE_SPEED(g_lead.driver, std::max(v, 0.01f));
    SET_VEHICLE_HANDBRAKE(g_lead.veh, v < 0.5f);
  } else if (type == "gas") {
    g_testGasUntil = now + MsgNum(m, "secs", 1.0);  // as if the driver pressed the gas
  } else if (type == "traffic") {
    g_noTraffic = !MsgBool(m, "on", true);
    if (g_noTraffic) ClearModel(0, 300.0f);  // any model
    Log(std::string("traffic ") + (g_noTraffic ? "off" : "on"));
  } else if (type == "setup") {
    Setup s;
    s.x = static_cast<float>(MsgNum(m, "x", NAN));
    s.y = static_cast<float>(MsgNum(m, "y", NAN));
    s.z = static_cast<float>(MsgNum(m, "z", NAN));
    s.speed = static_cast<float>(MsgNum(m, "speed", 0));
    s.minLanes = static_cast<int>(MsgNum(m, "lanes", 0));
    s.t = now;
    std::string model = MsgStr(m, "model");
    if (!model.empty()) {
      // a name, or a hash as the ini's per-model keys write it (0x...), for add-on cars
      s.model = model.rfind("0x", 0) == 0 ? static_cast<Hash>(std::strtoul(model.c_str(), nullptr, 16)) : GET_HASH_KEY(model.c_str());
      REQUEST_MODEL(s.model);
      s.step = 1;
    } else if (!std::isnan(s.x)) {
      s.step = 2;
    }
    if (m.count("hour")) SET_CLOCK_TIME(static_cast<int>(MsgNum(m, "hour")), 0, 0);
    std::string weather = MsgStr(m, "weather");
    if (!weather.empty()) SET_WEATHER_TYPE_NOW_PERSIST(weather.c_str());
    if (m.count("fix") && g_veh.handle) SET_VEHICLE_FIXED(g_veh.handle);
    g_setup = s;
  }
}

void SendFrame(std::vector<uint8_t> &&frames, double t) {
  std::string state;
  {
    std::lock_guard lk(g_stateMutex);
    state = g_state;
  }
  std::string header = "{\"type\":\"frame\",\"t\":" + Num(t) + ",\"width\":" + std::to_string(CAM_W) + ",\"height\":" +
                       std::to_string(CAM_H) + ",\"views\":[\"road\",\"wide\"],\"state\":" + state + "}";
  g_link.SendFrame(header, std::move(frames));
}

}  // namespace

extern "C" __declspec(dllexport) void CoreInit(const CoreHost *host) {
  shv::api = *host->api;
  g_dir = host->dir;
  ReadConfig();
  Log("init: bridge " + (g_cfg.bridge.empty() ? std::string("from ") + BRIDGE_FILE : g_cfg.bridge) + ", vfov " + Num(g_cfg.vfov) +
      ", lens " + (g_cfg.lens ? "on" : "off"));
  g_link.Start(BridgeAddress, Log);
}

extern "C" __declspec(dllexport) void CoreTick() {
  double now = QpcSeconds();
  float dt = GET_FRAME_TIME();
  for (auto &m : g_link.TakeMessages()) HandleMessage(m, now);

  Ped ped = PLAYER_PED_ID();
  if (g_setup.step) StepSetup(ped, now);
  StepLead(now);
  if (g_noTraffic) {
    SET_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(0.0f);
    SET_RANDOM_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(0.0f);
    SET_PARKED_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(0.0f);
    SET_PED_DENSITY_MULTIPLIER_THIS_FRAME(0.0f);
    SET_SCENARIO_PED_DENSITY_MULTIPLIER_THIS_FRAME(0.0f, 0.0f);
  }
  Vehicle v = IS_PED_IN_ANY_VEHICLE(ped, FALSE) ? GET_VEHICLE_PED_IS_IN(ped, FALSE) : 0;
  if (v != g_veh.handle) OnVehicleChanged(v);

  bool connected = g_link.Connected();
  bool want = connected && v != 0;
  if (want && !g_cam) ActivateCamera();
  else if (!want && g_cam) ReleaseCamera();

  if (g_cfg.interleave && g_cfg.presentHook && g_hook == Hook::Off && want)
    g_hook = present_hook::Install(Log, [](const HookFrame &f) { g_capture.ProcessShared(f); }) ? Hook::On : Hook::Failed;
  bool hooked = g_cfg.interleave && g_cfg.presentHook && g_hook == Hook::On;
  CaptureConfig cc;
  cc.vfovDeg = g_cfg.vfov;
  cc.roadVfovDeg = g_cfg.roadVfov;
  cc.wideVfovDeg = g_cfg.wideVfov;
  cc.lens = g_cfg.lens;
  if (want && hooked && !g_capture.HookMode()) {
    g_capture.StartHook(cc, SendFrame, Log);
  } else if (want && !hooked && (!g_capture.Running() || g_capture.HookMode()) && !g_captureFailed) {
    HWND hwnd = FindGameWindow();
    if (hwnd) g_captureFailed = !g_capture.Start(hwnd, cc, SendFrame, Log);
  }
  g_capture.SetEnabled(want);
  g_capture.SetMarker(g_cfg.interleave && !hooked);
  present_hook::SetEnabled(want && hooked);

  int view = g_cam ? UpdateCameraFrame(now, dt, hooked && g_cfg.splitViews) : -1;
  if (view >= 0) {
    // help text would cover the marker, and the wide lens reaches the radar in the corner. Hiding the feed for a frame
    // restarts its animation, which flickers, so it's hidden only where the frames reach the screen.
    HIDE_HELP_TEXT_THIS_FRAME();
    HIDE_HUD_AND_RADAR_THIS_FRAME();
    if (!hooked) THEFEED_HIDE_THIS_FRAME();
    // tells the capture this frame is the openpilot camera's, and which view by its colour (magenta both, cyan road,
    // yellow wide); the lens resampling blacks it out
    if (g_cfg.interleave)
      DRAW_RECT(0.0025f, 0.0045f, 0.005f, 0.009f, view == HOOK_ROAD ? 0 : 255, view == HOOK_BOTH ? 0 : 255, view == HOOK_WIDE ? 0 : 255, 255, FALSE);
  }
  // on the player's frames only, which keeps it out of the openpilot camera's
  if (connected && v && view < 0) DrawSpeed(now);
  if (connected) {
    // police chases after a scrape with traffic would end any drive
    SET_MAX_WANTED_LEVEL(0);
    CLEAR_PLAYER_WANTED_LEVEL(PLAYER_ID());
  }
  // up arrow, the default speed key, otherwise takes out the phone
  if (connected && v) DISABLE_CONTROL_ACTION(0, INPUT_PHONE, TRUE);

  ReadDriver(now);
  if (g_engagePresses.load() && !connected) g_engagePresses = 0;
  if (v) {
    UpdateIndicator();
    Measure(dt);
    ApplyControls(dt, now);
  }
  Publish(now, v != 0);
}

extern "C" __declspec(dllexport) void CoreShutdown() {
  ReleaseControls();
  ReleaseCamera();
  RemoveLead();
  if (g_hook == Hook::On) present_hook::Uninstall();
  g_capture.Stop();
  g_link.Stop();
  Log("shutdown");
}

extern "C" __declspec(dllexport) void CoreKey(DWORD key) {
  if (static_cast<int>(key) == g_cfg.keyEngage) g_engagePresses++;
  else if (static_cast<int>(key) == g_cfg.keyLeft) g_leftPresses++;
  else if (static_cast<int>(key) == g_cfg.keyRight) g_rightPresses++;
}
