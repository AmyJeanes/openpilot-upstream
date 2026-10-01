// gta5op_core.dll: the openpilot camera, vehicle state and controls, run once per game frame by the loader.
#include <winsock2.h>
#include <windows.h>
#include <share.h>
#include <Xinput.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

#include "capture.h"
#include "core_api.h"
#include "natives.h"
#include "net.h"

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
  float yawGain = 15.5f;  // yaw rate (rad/s) per unit of steer bias, measured with steertest at 7-14 m/s
  float latKi = 1.0f;     // 1/s, integral on the yaw rate error, in units of the feed-forward
  float accelGain = 4.0f;      // m/s^2 at full throttle, roughly
  float brakeGain = 8.0f;      // m/s^2 at full brake, roughly
  int keyEngage = VK_F6;
  int keyLeft = VK_OEM_COMMA;
  int keyRight = VK_OEM_PERIOD;
};

const char *BRIDGE_FILE = "C:\\Users\\Public\\gta5op-bridge.txt";

std::wstring g_dir;
Config g_cfg;
Link g_link;
Capture g_capture;
bool g_captureFailed = false;
std::mutex g_logMutex;

std::mutex g_stateMutex;
std::string g_state = "{}";  // latest vehicle state as JSON, sent with each frame

std::atomic<int> g_engagePresses{0}, g_leftPresses{0}, g_rightPresses{0};

struct VehicleInfo {
  Vehicle handle = 0;
  float wheelBase = 2.7f;
  float mountX = 0, mountY = 0, mountZ = 1.0f;
  int wheelLf = -1;
} g_veh;

struct Motion {
  bool valid = false;
  float v = 0, aMeas = 0, yawRate = 0, heading = 0, pitch = 0, roll = 0;
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

// open-loop steering check: holds a steer bias and logs the motion it produces
struct SteerTest {
  double until = 0, nextLog = 0;
  float bias = 0, throttle = 0;
} g_test;

Cam g_cam = 0;
float g_camPitch = 0, g_camYaw = 0;  // degrees, for checks against known rotations
int g_indicator = 0;  // 0 off, 1 left, 2 right
bool g_engaged = false;

struct Setup {
  int step = 0;  // 0 idle, 1 model loading, 2 waiting for the world to load, 3 placing on the road
  float x = 0, y = 0, z = 0, speed = 0;
  int minLanes = 0;  // lanes in the direction of travel, e.g. 3 for a freeway
  Hash model = 0;
  double t = 0;
} g_setup;

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

void ReadConfig() {
  std::string ini = std::filesystem::path(g_dir + L"\\gta5op.ini").string();
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
  c.yawGain = num("yaw_gain", c.yawGain);
  c.latKi = num("lat_ki", c.latKi);
  c.accelGain = num("accel_gain", c.accelGain);
  c.brakeGain = num("brake_gain", c.brakeGain);
  c.keyEngage = key("key_engage", c.keyEngage);
  c.keyLeft = key("key_left", c.keyLeft);
  c.keyRight = key("key_right", c.keyRight);
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
  HARD_ATTACH_CAM_TO_ENTITY(g_cam, g_veh.handle, g_camPitch, 0.0f, g_camYaw, g_veh.mountX, g_veh.mountY, g_veh.mountZ, TRUE);
}

void ActivateCamera() {
  g_cam = CREATE_CAM("DEFAULT_SCRIPTED_CAMERA", TRUE);
  AttachCamera();
  SET_CAM_FOV(g_cam, g_cfg.vfov);
  SET_CAM_NEAR_CLIP(g_cam, 0.05f);
  SET_CAM_ACTIVE(g_cam, TRUE);
  RENDER_SCRIPT_CAMS(TRUE, FALSE, 0, TRUE, FALSE, 0);
  Log("camera on");
}

void ReleaseCamera() {
  if (!g_cam) return;
  RENDER_SCRIPT_CAMS(FALSE, FALSE, 0, TRUE, FALSE, 0);
  if (DOES_CAM_EXIST(g_cam)) DESTROY_CAM(g_cam, FALSE);
  g_cam = 0;
  Log("camera off");
}

// *** vehicle ***

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
void ReadDriver() {
  Driver d;
  if (GameFocused()) {
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
  g_m.rotVel = GET_ENTITY_ROTATION_VELOCITY(v);
  if (g_veh.wheelLf >= 0) g_m.steerBone = GET_ENTITY_BONE_OBJECT_ROTATION(v, g_veh.wheelLf);
  g_m.valid = true;
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
    g_ctl.latI = g_ctl.lonI = 0;
    g_ctl.braking = false;
    g_ctl.steerOut = g_ctl.throttleOut = g_ctl.brakeOut = 0;
    return;
  }
  g_ctl.wasLive = true;
  float speed = g_m.v;

  // lateral: the game's steer bias turns the car at a yaw rate roughly proportional to it, whatever the speed, so
  // feed forward the yaw rate the curvature needs, and integrate the yaw rate error
  float v = std::max(speed, 3.0f);
  float yawTarget = g_ctl.curvature * v;
  g_ctl.latI = std::clamp(g_ctl.latI + g_cfg.latKi * (yawTarget - g_m.yawRate) / g_cfg.yawGain * dt, -0.05f, 0.05f);
  if (speed < 3.0f) g_ctl.latI *= std::max(0.0f, 1.0f - dt);
  float steer = std::clamp(yawTarget / g_cfg.yawGain + g_ctl.latI + 0.15f * g_user.steer, -0.3f, 0.3f);
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
  float u = std::clamp(accel + g_ctl.lonI, -4.0f, 2.5f);
  // the game's brake reverses a stopped car, so hold a stop with the handbrake like a car's brake hold
  bool hold = std::fabs(speed) < 0.5f && accel < 0 && !g_user.gas;
  if (hold != g_ctl.holding) SET_VEHICLE_HANDBRAKE(veh, hold);
  g_ctl.holding = hold;
  g_ctl.braking = !hold && u < (g_ctl.braking ? -0.1f : -0.3f) && !g_user.gas;
  float throttle = 0, brake = 0;
  if (hold) {
  } else if (g_ctl.braking) {
    brake = std::clamp(-u / g_cfg.brakeGain, 0.0f, 1.0f);
  } else {
    throttle = std::clamp(u / g_cfg.accelGain, 0.0f, 1.0f);
  }
  if (g_user.gas) throttle = 1.0f;
  SET_CONTROL_VALUE_NEXT_FRAME(0, INPUT_VEH_ACCELERATE, throttle);
  SET_CONTROL_VALUE_NEXT_FRAME(0, INPUT_VEH_BRAKE, brake);
  g_ctl.steerOut = steer;
  g_ctl.throttleOut = throttle;
  g_ctl.brakeOut = brake;
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

void Publish(double now, bool inVehicle) {
  std::ostringstream s;
  s << "{\"t\":" << Num(now) << ",\"inVehicle\":" << (inVehicle ? "true" : "false") << ",\"paused\":" << (IS_PAUSE_MENU_ACTIVE() ? "true" : "false")
    << ",\"engagePresses\":" << g_engagePresses.load() << ",\"resets\":" << g_m.resets;
  if (inVehicle) {
    s << ",\"vEgo\":" << Num(g_m.v) << ",\"aMeas\":" << Num(g_m.aMeas) << ",\"yawRate\":" << Num(g_m.yawRate)
      << ",\"heading\":" << Num(g_m.heading) << ",\"pitch\":" << Num(g_m.pitch) << ",\"roll\":" << Num(g_m.roll)
      << ",\"wheelBase\":" << Num(g_veh.wheelBase) << ",\"pos\":" << Vec(g_m.pos) << ",\"rotVel\":" << Vec(g_m.rotVel)
      << ",\"steerBone\":" << Vec(g_m.steerBone)
      << ",\"indicator\":" << (g_indicator == 1 ? "\"left\"" : g_indicator == 2 ? "\"right\"" : "null")
      << ",\"user\":{\"steer\":" << Num(g_user.steer) << ",\"gas\":" << (g_user.gas ? "true" : "false") << ",\"brake\":" << (g_user.brake ? "true" : "false") << "}"
      << ",\"out\":{\"steer\":" << Num(g_ctl.steerOut) << ",\"throttle\":" << Num(g_ctl.throttleOut) << ",\"brake\":" << Num(g_ctl.brakeOut)
      << ",\"latI\":" << Num(g_ctl.latI) << ",\"lonI\":" << Num(g_ctl.lonI) << ",\"hold\":" << (g_ctl.holding ? "true" : "false") << "}";
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

void HandleMessage(const Message &m, double now) {
  std::string type = MsgStr(m, "type");
  if (type == "control") {
    g_ctl.active = MsgBool(m, "active");
    g_ctl.curvature = static_cast<float>(MsgNum(m, "curvature"));
    g_ctl.accel = static_cast<float>(MsgNum(m, "accel"));
    g_ctl.t = now;
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
    if (g_cam) AttachCamera();
  } else if (type == "engage") {
    g_engagePresses++;  // as if the engage key were pressed
  } else if (type == "steertest") {
    g_test.bias = static_cast<float>(MsgNum(m, "bias", 0.1));
    g_test.throttle = static_cast<float>(MsgNum(m, "throttle", 0.3));
    g_test.until = now + MsgNum(m, "secs", 2.0);
    g_test.nextLog = now;
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
      s.model = GET_HASH_KEY(model.c_str());
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
  Vehicle v = IS_PED_IN_ANY_VEHICLE(ped, FALSE) ? GET_VEHICLE_PED_IS_IN(ped, FALSE) : 0;
  if (v != g_veh.handle) OnVehicleChanged(v);

  bool connected = g_link.Connected();
  bool want = connected && v != 0;
  if (want && !g_cam) ActivateCamera();
  else if (!want && g_cam) ReleaseCamera();

  if (want && !g_capture.Running() && !g_captureFailed) {
    HWND hwnd = FindGameWindow();
    CaptureConfig cc;
    cc.vfovDeg = g_cfg.vfov;
    cc.lens = g_cfg.lens;
    if (hwnd) g_captureFailed = !g_capture.Start(hwnd, cc, SendFrame, Log);
  }
  g_capture.SetEnabled(want);

  if (g_cam) {
    HIDE_HUD_AND_RADAR_THIS_FRAME();
    HIDE_HELP_TEXT_THIS_FRAME();
    THEFEED_HIDE_THIS_FRAME();
    SET_ENTITY_LOCALLY_INVISIBLE(v);
    SET_ENTITY_LOCALLY_INVISIBLE(ped);
  }
  if (connected) {
    // police chases after a scrape with traffic would end any drive
    SET_MAX_WANTED_LEVEL(0);
    CLEAR_PLAYER_WANTED_LEVEL(PLAYER_ID());
  }

  ReadDriver();
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
  g_capture.Stop();
  g_link.Stop();
  Log("shutdown");
}

extern "C" __declspec(dllexport) void CoreKey(DWORD key) {
  if (static_cast<int>(key) == g_cfg.keyEngage) g_engagePresses++;
  else if (static_cast<int>(key) == g_cfg.keyLeft) g_leftPresses++;
  else if (static_cast<int>(key) == g_cfg.keyRight) g_rightPresses++;
}
