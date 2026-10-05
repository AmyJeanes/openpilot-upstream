// gta5op_core.dll: the openpilot camera, vehicle state and controls, run once per game frame by the loader.
#include <winsock2.h>
#include <windows.h>
#include <share.h>
#include <Xinput.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "capture.h"
#include "core_api.h"
#include "natives.h"
#include "net.h"
#include "present_hook.h"
#include "script_hook.h"

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
  int keyDebug = VK_F7;    // the map debug overlay on and off
  bool gpsRoute = false;   // our route on the game's minimap and map
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
  Hash model = 0;
  std::string name;  // the model's display name
  float wheelBase = 2.7f;
  float mountX = 0, mountY = 0, mountZ = 1.0f;  // the camera's position on the car, m right, forward and up of its origin
  float baseX = 0, baseY = 0, baseZ = 1.0f;     // the same before the mount's jitter
  std::string mountSource;  // how the base was found: bone, dims, ini (comma mount); probe, bones (dashcam)
  double dashcamAt = 0;     // when to find the dashcam mount: the car's collision isn't there on its first frames
  int wheelLf = -1;
} g_veh;

// Where the camera goes on each car. "comma" (the default): where a comma device mounts, low enough for openpilot's
// calibration; "dashcam": a dashcam's place for training data, high at the top centre of the windscreen, drop m below
// the roof and back m behind the glass. The jitter (m right, forward, up; deg pitch, yaw) is added to either.
struct Mount {
  std::string mode = "comma";
  float drop = 0.08f, back = 0.06f;
  float dx = 0, dy = 0, dz = 0, pitch = 0, yaw = 0;
} g_mount;

// held each frame while set; traffic off zeroes them all
struct Density {
  bool set = false;
  float vehicles = 1, random = 1, parked = 1, peds = 1, scenario = 1;
} g_density;

// what the world command last set, for the state (the clock's pause can't be read back)
struct World {
  std::string weather;
  float transition = 0, rain = -1;
  bool frozen = false;
} g_world;

// palette indices to apply to a car, -1 to leave one
struct Colours {
  int primary = -1, secondary = -1, pearl = -1, wheel = -1;
  float dirt = -1;
};

// a car swap: the player's car replaced with a new one of a model, where it is and at its speed
struct Swap {
  int step = 0;  // 0 idle, 1 model loading
  Hash model = 0;
  Colours colours;
  double t = 0;
  int swaps = 0;
  std::string error;
} g_swap;

struct Motion {
  bool valid = false;
  float v = 0, aMeas = 0, yawRate = 0, heading = 0, pitch = 0, roll = 0;
  float grade = 0, bank = 0;  // radians: nose up, right side down
  Vector3 pos{}, rotVel{}, steerBone{};
  int resets = 0;
  int collisions = 0;  // frames the car touched something, counted up so a 20 Hz reader misses none
  float bodyHealth = 0;
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
  float heading = NAN;  // the direction wanted, deg
  int lane = -1;  // the lane wanted, from the left
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

// The AI expert driver: the player's ped drives its own car with a vehicle task, to a target the bridge keeps moving along
// the route (or the map's waypoint, or wandering), while openpilot's controls and the player's driving inputs are ignored.
struct Ai {
  bool on = false;
  bool hasTarget = false;
  Vector3 target{};
  float speed = 12.0f;     // m/s, the task's cruise speed and its cap
  int style = 1076369579;  // the story-mode taxi's: stops for cars, peds and lights, no short cuts
  float ability = 1.0f, aggressiveness = 0.0f;
  float stopRange = 5.0f, straightLine = 10.0f;
  std::string task = "longrange";  // or coord, wander
  std::string mode;                // what it drives to: target, waypoint, wander
  // what the game was given, re-tasked only when it changes: re-tasking makes the AI hesitate
  bool tasked = false, speedPending = false, stylePending = false;
  std::string taskedKey;
  Vector3 taskedTarget{}, waypoint{};
  double taskedAt = -1e9, waypointAt = -1e9;
  int status = 7, retasks = 0, aborts = 0;
  std::string error;
} g_ai;
std::atomic<bool> g_aiOn{false};
std::atomic<int> g_aiAbortPresses{0};  // the engage key while the AI drives
bool LeadTruth(float &ahead, float &left, float &speed);
float CameraHeight(double now);
float VehicleAhead(double now);
std::string AiState(Vehicle v);

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
  c.keyDebug = key("key_debug", c.keyDebug);
  c.gpsRoute = num("gps_route", 0) != 0;
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
  for (Cam c : g_cams)
    HARD_ATTACH_CAM_TO_ENTITY(c, g_veh.handle, g_camPitch + g_mount.pitch, 0.0f, g_camYaw + g_mount.yaw, g_veh.mountX, g_veh.mountY, g_veh.mountZ, TRUE);
}

// the mount from its base and jitter, onto the camera
void ApplyMount() {
  g_veh.mountX = g_veh.baseX + g_mount.dx;
  g_veh.mountY = g_veh.baseY + g_mount.dy;
  g_veh.mountZ = g_veh.baseZ + g_mount.dz;
  if (g_cam) AttachCamera();
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

bool g_radarHidden = false;  // by the openpilot camera's last frame
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

std::string HashKey(Hash h) {
  char key[16];
  snprintf(key, sizeof(key), "0x%08X", h);
  return key;
}

std::string ModelKey(Vehicle v) { return HashKey(GET_ENTITY_MODEL(v)); }

void ReleaseControls() {
  if (g_veh.handle && DOES_ENTITY_EXIST(g_veh.handle)) {
    if (g_ctl.holding) SET_VEHICLE_HANDBRAKE(g_veh.handle, FALSE);
    SET_VEHICLE_INDICATOR_LIGHTS(g_veh.handle, 0, FALSE);
    SET_VEHICLE_INDICATOR_LIGHTS(g_veh.handle, 1, FALSE);
  }
  g_ctl.holding = false;
}

// A comma device's place: behind the windscreen, from the car's windscreen bone, else a quarter of the way to its front
void CommaMount() {
  Vehicle v = g_veh.handle;
  Vector3 mn{}, mx{};
  GET_MODEL_DIMENSIONS(g_veh.model, &mn, &mx);
  g_veh.baseX = 0;
  g_veh.baseY = std::isnan(g_cfg.mountForward) ? 0.25f * mx.y : g_cfg.mountForward;
  g_veh.baseZ = mn.z + g_cfg.mountHeight;
  g_veh.mountSource = "dims";
  // behind the windscreen, as a comma device is mounted, whatever the cabin's height; the bone sits low on the glass
  int ws = GET_ENTITY_BONE_INDEX_BY_NAME(v, "windscreen");
  if (ws >= 0 && std::isnan(g_cfg.mountForward)) {
    Vector3 w = GET_WORLD_POSITION_OF_ENTITY_BONE(v, ws);
    Vector3 l = GET_OFFSET_FROM_ENTITY_GIVEN_WORLD_COORDS(v, w.x, w.y, w.z);
    g_veh.baseY = l.y;
    // and at least as high as a comma device sits below the roof (a Model 3 calibrates to 1.20-1.23 m, 0.29 m below
    // its roof): the driving model judges scale from the camera's height, reading speeds 20% fast from 9 cm too low
    g_veh.baseZ = std::max(l.z + 0.08f, mx.z - 0.29f);
    g_veh.mountSource = "bone";
  }
  // a mount set with the camera command for this model
  char saved[32];
  GetPrivateProfileStringA("mount_forward", ModelKey(v).c_str(), "", saved, sizeof(saved), IniPath().c_str());
  if (saved[0]) {
    g_veh.baseY = static_cast<float>(atof(saved));
    g_veh.mountSource = "ini";
  }
}

// a bone's position in the car's own axes
bool BoneLocal(Vehicle v, const char *bone, Vector3 &out) {
  int i = GET_ENTITY_BONE_INDEX_BY_NAME(v, bone);
  if (i < 0) return false;
  Vector3 w = GET_WORLD_POSITION_OF_ENTITY_BONE(v, i);
  out = GET_OFFSET_FROM_ENTITY_GIVEN_WORLD_COORDS(v, w.x, w.y, w.z);
  return true;
}

// the top of the car's own body straight down through a point (its axes), from a probe of its collision
bool SurfaceZ(Vehicle v, float x, float y, float top, float bottom, float &z) {
  Vector3 a = GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(v, x, y, top), b = GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(v, x, y, bottom);
  int test = START_EXPENSIVE_SYNCHRONOUS_SHAPE_TEST_LOS_PROBE(a.x, a.y, a.z, b.x, b.y, b.z, 2, 0, 7);
  BOOL hit = FALSE;
  Vector3 end{}, normal{};
  Entity e = 0;
  if (GET_SHAPE_TEST_RESULT(test, &hit, &end, &normal, &e) != 2 || !hit || (e && e != v)) return false;
  z = GET_OFFSET_FROM_ENTITY_GIVEN_WORLD_COORDS(v, end.x, end.y, end.z).z;
  return true;
}

// A dashcam's place, at the top centre of the windscreen: probes down the car's centre line give the roof's height over
// the driver and where the windscreen falls through drop below it; the camera goes back behind the glass there. Without
// them, the windscreen bone and the model's top stand in, the glass taken as rising 1.2 m back per m up. A result
// outside the cabin keeps the comma mount.
void DashcamMount() {
  Vehicle v = g_veh.handle;
  Vector3 mn{}, mx{}, seat{}, ws{};
  GET_MODEL_DIMENSIONS(g_veh.model, &mn, &mx);
  bool hasSeat = BoneLocal(v, "seat_dside_f", seat), hasWs = BoneLocal(v, "windscreen", ws);
  float seatY = hasSeat ? seat.y : 0.0f;
  constexpr float STEP = 0.05f;
  float y0 = seatY - 0.4f, y1 = hasWs ? ws.y + 0.8f : 0.7f * mx.y;
  std::vector<float> zs;
  for (float y = y0; y <= y1 && zs.size() < 120; y += STEP) {
    float z = NAN;
    zs.push_back(SurfaceZ(v, 0.0f, y, mx.z + 0.5f, mn.z + 0.3f, z) ? z : NAN);
  }
  int hits = 0;
  float roof = -1e9f;
  for (size_t i = 0; i < zs.size(); i++) {
    hits += !std::isnan(zs[i]);
    if (y0 + i * STEP <= seatY + 0.3f && !std::isnan(zs[i])) roof = std::max(roof, zs[i]);
  }
  bool roofOk = roof > mn.z + 0.9f && roof < mx.z + 0.05f;
  float camY = NAN, camZ = NAN;
  std::string source;
  if (roofOk) {
    camZ = roof - g_mount.drop;
    // forward from over the driver to where the glass falls through the camera's height
    for (size_t i = 1; i < zs.size(); i++) {
      float y = y0 + i * STEP, a = zs[i - 1], b = zs[i];
      if (y - STEP < seatY || std::isnan(a) || std::isnan(b) || !(a > camZ && b <= camZ)) continue;
      camY = y - STEP + STEP * (a - camZ) / (a - b) - g_mount.back;
      source = "probe";
      break;
    }
  }
  if (source.empty() && hasWs) {
    camZ = (roofOk ? roof : mx.z - 0.03f) - g_mount.drop;
    camY = ws.y - 1.2f * std::max(0.0f, camZ - ws.z) - g_mount.back;
    source = "bones";
  }
  if (hasSeat && camY < seatY + 0.3f) camY = seatY + 0.3f;  // ahead of the driver's head
  bool ok = !source.empty() && camZ > mn.z + 0.8f && camZ < mx.z && camY > mn.y && camY < mx.y;
  Log("dashcam mount: " + (ok ? source : "none, comma mount kept") + ", " + std::to_string(hits) + "/" + std::to_string(zs.size()) +
      " probes hit, roof " + (roofOk ? Num(roof) : "?") + ", seat " + (hasSeat ? Num(seat.y) : "?") + ", windscreen " +
      (hasWs ? Vec(ws) : "?") + " -> " + Num(camY) + " fwd, " + Num(camZ) + " up");
  if (!ok) return;
  g_veh.baseX = 0;
  g_veh.baseY = camY;
  g_veh.baseZ = camZ;
  g_veh.mountSource = source;
  ApplyMount();
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
  g_veh.model = GET_ENTITY_MODEL(v);
  for (const char *c = GET_DISPLAY_NAME_FROM_VEHICLE_MODEL(g_veh.model); c && *c; c++)
    if (isalnum(static_cast<unsigned char>(*c)) || *c == '_' || *c == ' ') g_veh.name += *c;
  Vector3 mn{}, mx{};
  GET_MODEL_DIMENSIONS(g_veh.model, &mn, &mx);
  CommaMount();
  ApplyMount();
  if (g_mount.mode == "dashcam") g_veh.dashcamAt = QpcSeconds() + 0.3;
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
  g_m.collisions += HAS_ENTITY_COLLIDED_WITH_ANYTHING(v) ? 1 : 0;
  g_m.bodyHealth = GET_VEHICLE_BODY_HEALTH(v);
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

void SetIndicator(int indicator) {
  g_indicator = indicator;
  if (g_veh.handle) {
    // turnSignal 1 is the left light, 0 the right
    SET_VEHICLE_INDICATOR_LIGHTS(g_veh.handle, 1, g_indicator == 1);
    SET_VEHICLE_INDICATOR_LIGHTS(g_veh.handle, 0, g_indicator == 2);
  }
}

void UpdateIndicator() {
  int left = g_leftPresses.exchange(0), right = g_rightPresses.exchange(0);
  int indicator = g_indicator;
  if (left) indicator = indicator == 1 ? 0 : 1;
  if (right) indicator = indicator == 2 ? 0 : 2;
  if (indicator != g_indicator) SetIndicator(indicator);
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

// *** map debug overlay, GPS route, GTA's directions ***

struct P3 {
  float x, y, z;
};

struct DebugLine {
  char kind;  // gta5_overlay.py's: e edge, d divider, l/s stop line (light/sign), j junction, r/b route (ahead/behind),
              // n nav's lane plan, m next turn, g where its signal comes on
  std::vector<P3> pts;
};

// The map around the car from the bridge, drawn into the world every player frame while on (debug command, key_debug)
struct DebugOverlay {
  bool on = false, force = false, ground = false;
  std::string layers = "edsjrnm";  // e d s j r n m as above (s both stop lines, r both route parts, m both points), f fills
  float lift = 0.1f;               // m above the road
  std::vector<DebugLine> lines;
  int vertices = 0;
  bool recording = false;  // the bridge is recording: drawn only when forced
  double t = -1e9;         // when the lines came
} g_debug;
std::atomic<int> g_debugPresses{0};
constexpr double DEBUG_STALE = 3.0;    // s without new lines: the bridge stopped sending them
constexpr int DEBUG_MAX_SEGMENTS = 6000;  // drawn per frame

char DebugLayer(char kind) { return kind == 'l' ? 's' : kind == 'b' ? 'r' : kind == 'g' ? 'm' : kind; }

// the map preview's colours (gta5_train maprender.preview), but nav's plan, which runs on the route
void DebugColour(char kind, int &r, int &g, int &b) {
  switch (kind) {
    case 'e': r = 0, g = 255, b = 0; break;
    case 'd': r = 0, g = 255, b = 255; break;
    case 'l': r = 255, g = 230, b = 0; break;
    case 's': r = 255, g = 140, b = 0; break;
    case 'j': r = 40, g = 110, b = 255; break;
    case 'r': r = 255, g = 25, b = 25; break;
    case 'b': r = 140, g = 15, b = 15; break;
    case 'n': r = 255, g = 255, b = 255; break;  // dashed over the route's ribbon
    case 'g': r = 255, g = 190, b = 0; break;
    default: r = 255, g = 255, b = 255;
  }
}

// gta5_overlay.py's polylines: a kind letter, then decimetres from the origin, each point after the first as the change
// from the one before; polylines separated by ';'
void ParseDebugGeo(const std::string &s, float ox, float oy, float oz, std::vector<DebugLine> &out, int &vertices) {
  out.clear();
  vertices = 0;
  const char *c = s.c_str(), *end = c + s.size();
  while (c < end) {
    DebugLine line{*c++, {}};
    long acc[3] = {0, 0, 0};
    for (int n = 0; c < end && *c != ';'; n++) {
      char *next = nullptr;
      long v = strtol(c, &next, 10);
      if (next == c) break;
      c = next;
      acc[n % 3] += v;
      if (n % 3 == 2) line.pts.push_back({ox + acc[0] * 0.1f, oy + acc[1] * 0.1f, oz + acc[2] * 0.1f});
      if (c < end && *c == ',') c++;
    }
    while (c < end && *c != ';') c++;
    if (c < end) c++;
    vertices += static_cast<int>(line.pts.size());
    if (!line.pts.empty()) out.push_back(std::move(line));
  }
}

// onto the game's ground under each point, where it's near the map's height
void SnapToGround(std::vector<DebugLine> &lines) {
  for (DebugLine &l : lines)
    for (P3 &p : l.pts) {
      float z = 0;
      if (GET_GROUND_Z_FOR_3D_COORD(p.x, p.y, p.z + 1.5f, &z, FALSE, FALSE) && std::fabs(z - p.z) < 3.0f) p.z = z;
    }
}

// a triangle seen from above and below: DRAW_POLY shows from one side only
void DrawTriangle(const P3 &a, const P3 &b, const P3 &c, int r, int g, int bl, int alpha) {
  DRAW_POLY(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, r, g, bl, alpha);
  DRAW_POLY(a.x, a.y, a.z, c.x, c.y, c.z, b.x, b.y, b.z, r, g, bl, alpha);
}

// A band ROUTE_RIBBON m wide along a polyline, lying on the road: a quad per segment, square to it, and a bevel filling
// the outside of each corner (mitred corners spike at sharp turns). Returns the draw calls used.
constexpr float ROUTE_RIBBON = 1.75f;
int DrawRibbon(const std::vector<P3> &pts, float lift, int r, int g, int b, int budget) {
  constexpr int ALPHA = 90;
  constexpr float HALF = ROUTE_RIBBON / 2;
  int used = 0;
  bool havePrev = false;
  float pnx = 0, pny = 0;  // the previous segment's half-width offset to the right
  for (size_t i = 1; i < pts.size() && used + 6 <= budget; i++) {
    const P3 &a = pts[i - 1], &c = pts[i];
    float dx = c.x - a.x, dy = c.y - a.y, len = std::hypot(dx, dy);
    if (!(len > 0.05f)) continue;  // also NaN
    float nx = dy / len * HALF, ny = -dx / len * HALF;
    P3 aL{a.x - nx, a.y - ny, a.z + lift}, aR{a.x + nx, a.y + ny, a.z + lift}, cL{c.x - nx, c.y - ny, c.z + lift}, cR{c.x + nx, c.y + ny, c.z + lift};
    DrawTriangle(aL, aR, cR, r, g, b, ALPHA);
    DrawTriangle(aL, cR, cL, r, g, b, ALPHA);
    used += 4;
    if (havePrev) {
      // the outside of the corner at a: right of the line (+ the offsets) where it turns left
      float turn = pnx * ny - pny * nx, s = turn > 0 ? 1.0f : -1.0f;
      if (std::fabs(turn) > 1e-4f) {
        P3 o{a.x, a.y, a.z + lift}, p{a.x + s * pnx, a.y + s * pny, a.z + lift}, q{a.x + s * nx, a.y + s * ny, a.z + lift};
        DrawTriangle(o, p, q, r, g, b, ALPHA);
        used += 2;
      }
    }
    pnx = nx, pny = ny, havePrev = true;
  }
  return used;
}

// a line in DASH_ON m dashes with DASH_OFF m gaps, so it reads over the route's ribbon; returns the draw calls used
constexpr float DASH_ON = 2.0f, DASH_OFF = 1.5f;
// Dashes by their index along the whole line, so every pass of the inner loop moves on a dash (a float phase stepped
// by what's left of a dash can stop moving at rounding, which hung the game's script thread).
int DrawDashed(const std::vector<P3> &pts, float lift, int r, int g, int b, int budget) {
  constexpr double PERIOD = DASH_ON + DASH_OFF;
  int used = 0;
  double base = 0;  // m along the line to the segment's start
  for (size_t i = 1; i < pts.size() && used < budget; i++) {
    const P3 &a = pts[i - 1], &c = pts[i];
    double len = std::hypot(double(c.x) - a.x, double(c.y) - a.y);
    if (!(len > 1e-3) || !(len < 1e4)) {  // also NaN
      base += std::isfinite(len) ? len : 0;
      continue;
    }
    for (long k = long(std::floor(base / PERIOD)), last = long(std::floor((base + len) / PERIOD)); k <= last && used < budget; k++) {
      double t0 = std::max(0.0, (k * PERIOD - base) / len), t1 = std::min(1.0, (k * PERIOD + DASH_ON - base) / len);
      if (t1 - t0 < 1e-4) continue;
      DRAW_LINE(float(a.x + (c.x - a.x) * t0), float(a.y + (c.y - a.y) * t0), float(a.z + (c.z - a.z) * t0 + lift),
                float(a.x + (c.x - a.x) * t1), float(a.y + (c.y - a.y) * t1), float(a.z + (c.z - a.z) * t1 + lift), r, g, b, 255);
      used++;
    }
    base += len;
  }
  return used;
}

void DrawDebug(double now) {
  const DebugOverlay &d = g_debug;
  if (!d.on) return;
  bool stale = now - d.t > DEBUG_STALE, held = d.recording && !d.force;
  std::string status = "MAP DEBUG " + d.layers + (stale ? " (no map from the bridge)" : held ? " (off while recording)" : " " + std::to_string(d.vertices) + " pts");
  DrawText(status, 0.985f, 0.70f, 0.35f, 255, 255, 255);
  if (stale || held) return;
  auto has = [&](char layer) { return d.layers.find(layer) != std::string::npos; };
  bool fill = has('f');
  int budget = DEBUG_MAX_SEGMENTS;
  float lift = d.lift;
  for (const DebugLine &l : d.lines) {
    int r, g, b;
    DebugColour(l.kind, r, g, b);
    const P3 &p0 = l.pts[0];
    if (l.kind == 'm' || l.kind == 'g') {
      if (!has('m')) continue;
      float top = l.kind == 'm' ? 6.0f : 3.5f;
      DRAW_LINE(p0.x, p0.y, p0.z + lift, p0.x, p0.y, p0.z + top, r, g, b, 255);
      if (l.kind == 'm')
        DRAW_MARKER(1, p0.x, p0.y, p0.z + lift, 0, 0, 0, 0, 0, 0, 1.5f, 1.5f, 1.2f, r, g, b, 140, FALSE, FALSE, 2, FALSE, nullptr, nullptr, FALSE);
      else
        DRAW_MARKER(0, p0.x, p0.y, p0.z + top, 0, 0, 0, 0, 0, 0, 1.0f, 1.0f, 1.0f, r, g, b, 200, FALSE, FALSE, 2, FALSE, nullptr, nullptr, FALSE);
      continue;
    }
    if (l.kind == 'j' && fill && l.pts.size() >= 4) {
      // a fan from the first corner of the hull, both windings so it shows from above and below
      for (size_t i = 2; i + 1 < l.pts.size() && budget > 0; i++, budget -= 2) {
        const P3 &a = l.pts[i - 1], &c = l.pts[i];
        DRAW_POLY(p0.x, p0.y, p0.z + lift, a.x, a.y, a.z + lift, c.x, c.y, c.z + lift, r, g, b, 60);
        DRAW_POLY(p0.x, p0.y, p0.z + lift, c.x, c.y, c.z + lift, a.x, a.y, a.z + lift, r, g, b, 60);
      }
    }
    if (!has(DebugLayer(l.kind))) continue;
    if (l.kind == 'r' || l.kind == 'b') {
      budget -= DrawRibbon(l.pts, lift, r, g, b, budget);
      continue;
    }
    if (l.kind == 'n') {
      budget -= DrawDashed(l.pts, lift + 0.15f, r, g, b, budget);
      continue;
    }
    for (size_t i = 1; i < l.pts.size() && budget > 0; i++, budget--) {
      const P3 &a = l.pts[i - 1], &c = l.pts[i];
      DRAW_LINE(a.x, a.y, a.z + lift, c.x, c.y, c.z + lift, r, g, b, 255);
    }
  }
}

std::string DebugState(double now) {
  const DebugOverlay &d = g_debug;
  return "\"debug\":{\"on\":" + std::string(d.on ? "true" : "false") + ",\"layers\":\"" + d.layers + "\",\"force\":" + (d.force ? "true" : "false") +
         ",\"ground\":" + (d.ground ? "true" : "false") + ",\"lines\":" + std::to_string(d.lines.size()) + ",\"vertices\":" + std::to_string(d.vertices) +
         ",\"age\":" + Num(std::min(now - d.t, 1e6)) + "}";
}

// our route as a custom GPS route on the minimap and map (gpsroute command, gps_route in the ini), from the bridge's
// points; HUD colour 21 is purple, as GTA's own route
struct GpsRoute {
  bool on = false;
  int colour = 21, max = 100, radar = 16, map = 16;  // max: GTA's own limit on points isn't documented
  bool take = true;  // the map's waypoint off the map meanwhile (TakeWaypoint)
  std::vector<P3> pts;
  int shown = 0;  // points given to the game
} g_gps;

void ShowGpsRoute() {
  if (g_gps.shown) {
    CLEAR_GPS_CUSTOM_ROUTE();
    SET_GPS_CUSTOM_ROUTE_RENDER(FALSE, g_gps.radar, g_gps.map);
    g_gps.shown = 0;
  }
  if (!g_gps.on || g_gps.pts.size() < 2) return;
  START_GPS_CUSTOM_ROUTE(g_gps.colour, FALSE, TRUE);
  int n = std::min(static_cast<int>(g_gps.pts.size()), g_gps.max);
  for (int i = 0; i < n; i++) ADD_POINT_TO_GPS_CUSTOM_ROUTE(g_gps.pts[i].x, g_gps.pts[i].y, g_gps.pts[i].z);
  SET_GPS_CUSTOM_ROUTE_RENDER(TRUE, g_gps.radar, g_gps.map);
  g_gps.shown = n;
}

// "x,y,z;x,y,z;..." in metres
std::vector<P3> ParsePoints(const std::string &s) {
  std::vector<P3> out;
  const char *c = s.c_str(), *end = c + s.size();
  while (c < end) {
    float v[3];
    int n = 0;
    for (; n < 3 && c < end; n++) {
      char *next = nullptr;
      v[n] = strtof(c, &next);
      if (next == c) break;
      c = next;
      if (c < end && (*c == ',' || *c == ';')) c++;
    }
    if (n < 3) break;
    out.push_back({v[0], v[1], v[2]});
  }
  return out;
}

// While our route shows, the map's waypoint is taken off the map (in Enhanced nothing else hides GTA's own route line to
// it: SET_BLIP_ROUTE on its blip doesn't) and held here, reported in the state as the waypoint still, and given back
// with gpsroute off; a blip of our own marks it meanwhile. Within WAYPOINT_DONE m it's dropped, as GTA clears its own.
struct HeldWaypoint {
  bool held = false;
  float x = 0, y = 0, z = 0;
  int marker = 0;  // our blip
} g_waypoint;
constexpr float WAYPOINT_DONE = 20.0f;  // m

void RemoveWaypointMarker() {
  if (g_waypoint.marker && DOES_BLIP_EXIST(g_waypoint.marker)) REMOVE_BLIP(&g_waypoint.marker);
  g_waypoint.marker = 0;
}

void ReleaseWaypoint(bool restore) {
  if (!g_waypoint.held) return;
  g_waypoint.held = false;
  RemoveWaypointMarker();
  if (restore) SET_NEW_WAYPOINT(g_waypoint.x, g_waypoint.y);
  Log(std::string("waypoint ") + (restore ? "given back" : "reached") + " at " + Num(g_waypoint.x) + "," + Num(g_waypoint.y));
}

void TakeWaypoint(double now) {
  static double next = 0;
  if (now < next) return;
  next = now + 0.25;
  bool keep = g_gps.on && g_gps.take;
  if (!keep) {
    ReleaseWaypoint(true);
    return;
  }
  if (g_waypoint.held && g_veh.handle && std::hypot(g_m.pos.x - g_waypoint.x, g_m.pos.y - g_waypoint.y) < WAYPOINT_DONE) {
    ReleaseWaypoint(false);
    return;
  }
  // a new one set on the map replaces it; taken only once our route shows, which needs the bridge's router
  if (g_gps.shown <= 0 || !IS_WAYPOINT_ACTIVE()) return;
  Vector3 w = GET_BLIP_INFO_ID_COORD(GET_FIRST_BLIP_INFO_ID(GET_WAYPOINT_BLIP_ENUM_ID()));
  SET_WAYPOINT_OFF();
  RemoveWaypointMarker();
  g_waypoint = {true, w.x, w.y, w.z, ADD_BLIP_FOR_COORD(w.x, w.y, w.z)};
  if (g_waypoint.marker) {
    SET_BLIP_SPRITE(g_waypoint.marker, 8);  // the waypoint's own sprite
    SET_BLIP_COLOUR(g_waypoint.marker, 27);  // purple
  }
  Log("waypoint taken at " + Num(w.x) + "," + Num(w.y) + " while our route shows");
}

// the map's waypoint, or the one held while our route shows
bool WaypointAt(Vector3 &w) {
  if (g_waypoint.held) {
    w = Vector3{};
    w.x = g_waypoint.x, w.y = g_waypoint.y, w.z = g_waypoint.z;
    return true;
  }
  if (!IS_WAYPOINT_ACTIVE()) return false;
  w = GET_BLIP_INFO_ID_COORD(GET_FIRST_BLIP_INFO_ID(GET_WAYPOINT_BLIP_ENUM_ID()));
  return true;
}

std::string GpsState() {
  return "\"gpsRoute\":{\"on\":" + std::string(g_gps.on ? "true" : "false") + ",\"points\":" + std::to_string(g_gps.shown) + ",\"max\":" +
         std::to_string(g_gps.max) + ",\"waypointHeld\":" + (g_waypoint.held ? "true" : "false") + "}";
}

// GTA's own GPS directions to a point, while asked for (gtadirs command), to compare with our router's
struct GtaDirections {
  bool on = false;
  float x = 0, y = 0, z = 0;
  int direction = -1, result = 0;
  float p5 = 0, dist = 0;  // dist: decimetres to the next junction, as the game gives it
  double next = 0;
} g_dirs;

std::string DirectionsState(double now) {
  GtaDirections &d = g_dirs;
  if (!d.on) return "";
  if (now >= d.next) {
    d.next = now + 0.25;
    uint64_t dir = 0, p5 = 0, dist = 0;
    d.result = GENERATE_DIRECTIONS_TO_COORD(d.x, d.y, d.z, FALSE, &dir, &p5, &dist);
    d.direction = static_cast<int32_t>(static_cast<uint32_t>(dir));
    std::memcpy(&d.p5, &p5, sizeof(float));
    std::memcpy(&d.dist, &dist, sizeof(float));
  }
  return "\"gtaDirs\":{\"to\":[" + Num(d.x) + "," + Num(d.y) + "," + Num(d.z) + "],\"direction\":" + std::to_string(d.direction) + ",\"dist\":" +
         Num(d.dist / 10.0f) + ",\"rawDist\":" + Num(d.dist) + ",\"p5\":" + Num(d.p5) + ",\"result\":" + std::to_string(d.result) + "}";
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

// the map's waypoint and the GPS route to it: a point every 5 m from the car for up to 500 m, fewer where it ends
std::string Route(double now) {
  static std::string route;
  static double next = 0;
  if (now < next) return route;
  next = now + 0.2;
  route.clear();
  Vector3 w{};
  if (!WaypointAt(w)) return route;
  if (g_waypoint.held) return route = "\"waypoint\":[" + Num(w.x) + "," + Num(w.y) + "]";  // no GTA route to it meanwhile
  Vector3 last{};
  for (float d = 0; d <= 500.0f; d += 5.0f) {
    Vector3 p{};
    if (!GET_POS_ALONG_GPS_TYPE_ROUTE(&p, TRUE, d, 0)) break;
    if (d > 0 && p.x == last.x && p.y == last.y) break;  // past the end, it gives the end again
    route += std::string(route.empty() ? "[" : ",") + "[" + Num(std::round(p.x * 10) / 10) + "," + Num(std::round(p.y * 10) / 10) + "]";
    last = p;
  }
  if (!route.empty()) route = "\"waypoint\":[" + Num(w.x) + "," + Num(w.y) + "],\"route\":" + route + "]";
  return route;
}

constexpr float LANE_WIDTH = 5.4f;

// GTA's road at a point, for a car heading that way: its lanes that way and back, and the offsets right of the road's
// line of the point and of the lanes' left edge; false off the roads or across one
struct RoadLanes {
  float dirX = 0, dirY = 0;  // the road's direction the car's way, a unit vector
  int n = 0, back = 0;
  float median = 0, right = 0, leftEdge = 0;
};
bool RoadLanesAtPoint(const Vector3 &p, float heading, RoadLanes &r) {
  Vector3 a{}, b{};
  int toA = 0, toB = 0;
  if (!GET_CLOSEST_ROAD(p.x, p.y, p.z, 1.0f, 1, &a, &b, &toA, &toB, &r.median, FALSE)) return false;
  float dx = b.x - a.x, dy = b.y - a.y, len = std::hypot(dx, dy);
  if (len < 1.0f) return false;
  float off = std::remainder(heading - (std::atan2(dy, dx) / DEG - 90), 360.0f);
  bool towardsB = std::fabs(off) < 30;
  if (!towardsB && std::fabs(off) < 150) return false;  // turning, or on a crossing road
  float sign = towardsB ? 1.0f : -1.0f;
  r.dirX = dx / len * sign;
  r.dirY = dy / len * sign;
  r.right = ((p.x - a.x) * dy - (p.y - a.y) * dx) / len * sign;
  r.n = towardsB ? toB : toA;
  r.back = towardsB ? toA : toB;
  // lanes run out from the median's edge on a two-way road, and are centred on the road's line on a one-way one
  r.leftEdge = r.back > 0 ? r.median / 2 : -r.n * LANE_WIDTH / 2;
  return r.n > 0;
}
bool RoadLanesAt(Vector3 p, float heading, RoadLanes &r) {
  // GTA links some nodes to roads across, so look a little ahead or behind for the road this way too
  for (float d : {0.0f, 5.0f, 10.0f, -5.0f}) {
    Vector3 q = p;
    q.x -= std::sin(heading * DEG) * d;
    q.y += std::cos(heading * DEG) * d;
    if (RoadLanesAtPoint(q, heading, r)) return true;
  }
  return false;
}

// the car's lane, counted from the left, of the lanes its way, as "lane":[i,n]; negative in the oncoming lanes, and empty
// off GTA's roads or across them. Lanes are 5.4 m wide, outward from the median's edge on a two-way road and centred on
// its path nodes on a one-way one.
std::string Lane(double now) {
  static std::string lane;
  static double next = 0;
  if (now < next) return lane;
  next = now + 0.2;
  lane.clear();
  RoadLanes r;
  if (!RoadLanesAt(g_m.pos, g_m.heading, r)) return lane;
  float across = r.right - r.leftEdge;
  // the median counts as the inside lane, until the car is a quarter of a lane past it
  int i = across < -r.median - LANE_WIDTH / 4 ? std::max(static_cast<int>(std::floor((across + r.median) / LANE_WIDTH)), -r.back)
                                              : std::clamp(static_cast<int>(std::floor(across / LANE_WIDTH)), 0, r.n - 1);
  lane = "\"lane\":[" + std::to_string(i) + "," + std::to_string(r.n) + "]";
  return lane;
}

// what the traffic around shows of the light ahead, as "traffic":{red, crossing, peds}: AI cars going our way that wait at
// a red light (or queue behind one; they all clear when it turns green, before moving), cars crossing ahead, and
// pedestrians just in front
std::string Traffic(double now) {
  static std::string traffic;
  static double next = 0;
  if (now < next || !g_veh.handle) return traffic;
  next = now + 0.2;
  int red = 0, crossing = 0, peds = 0;
  int slots[2 + 2 * 32] = {32};
  int count = GET_PED_NEARBY_VEHICLES(PLAYER_PED_ID(), slots);
  for (int i = 0; i < std::min(count, 32); i++) {
    Vehicle v = slots[2 + 2 * i];
    if (v == g_veh.handle || !DOES_ENTITY_EXIST(v)) continue;
    Vector3 q = GET_ENTITY_COORDS(v, TRUE);
    Vector3 rel = GET_OFFSET_FROM_ENTITY_GIVEN_WORLD_COORDS(g_veh.handle, q.x, q.y, q.z);
    float dh = std::fabs(std::remainder(GET_ENTITY_HEADING(v) - g_m.heading, 360.0f));
    if (dh < 30 && std::fabs(rel.x) < 15 && rel.y > -40 && rel.y < 25 && IS_VEHICLE_STOPPED_AT_TRAFFIC_LIGHTS(v)) red++;
    else if (dh > 60 && dh < 120 && std::fabs(rel.x) < 30 && rel.y > 3 && rel.y < 40 && GET_ENTITY_SPEED(v) > 3) crossing++;
  }
  count = GET_PED_NEARBY_PEDS(PLAYER_PED_ID(), slots, -1);
  for (int i = 0; i < std::min(count, 32); i++) {
    Ped ped = slots[2 + 2 * i];
    if (!DOES_ENTITY_EXIST(ped) || IS_PED_DEAD_OR_DYING(ped, TRUE)) continue;
    Vector3 q = GET_ENTITY_COORDS(ped, TRUE);
    Vector3 rel = GET_OFFSET_FROM_ENTITY_GIVEN_WORLD_COORDS(g_veh.handle, q.x, q.y, q.z);
    if (std::fabs(rel.x) < 2.5f && rel.y > 1 && rel.y < 12) peds++;
  }
  traffic = "\"traffic\":{\"red\":" + std::to_string(red) + ",\"crossing\":" + std::to_string(crossing) + ",\"peds\":" + std::to_string(peds) + "}";
  return traffic;
}

std::string WeatherName(Hash h) {
  static const char *NAMES[] = {"EXTRASUNNY", "CLEAR", "CLOUDS", "SMOG", "FOGGY", "OVERCAST", "RAIN", "THUNDER", "CLEARING", "NEUTRAL",
                                "SNOW", "BLIZZARD", "SNOWLIGHT", "XMAS", "HALLOWEEN", "RAIN_HALLOWEEN", "SNOW_HALLOWEEN"};
  static Hash hashes[std::size(NAMES)] = {};
  if (!hashes[0])
    for (size_t i = 0; i < std::size(NAMES); i++) hashes[i] = GET_HASH_KEY(NAMES[i]);
  for (size_t i = 0; i < std::size(NAMES); i++)
    if (hashes[i] == h) return NAMES[i];
  return HashKey(h);
}

// "world":{hour, minute, weather (the type it's mostly), from, to, mix (the blend between them, 1 all to), rain (the rain
// and puddle level now), and what the world command last set: set (a weather), transition (s), rainSet, frozen}
std::string WorldState(double now) {
  static std::string out;
  static double next = 0;
  if (now < next) return out;
  next = now + 0.5;
  uint64_t from = 0, to = 0, mixSlot = 0;
  GET_CURR_WEATHER_STATE(&from, &to, &mixSlot);
  float mix = 0;
  std::memcpy(&mix, &mixSlot, sizeof(mix));
  std::string a = WeatherName(static_cast<Hash>(from)), b = WeatherName(static_cast<Hash>(to));
  out = "\"world\":{\"hour\":" + std::to_string(GET_CLOCK_HOURS()) + ",\"minute\":" + std::to_string(GET_CLOCK_MINUTES()) + ",\"weather\":\"" +
        (mix < 0.5f ? a : b) + "\",\"from\":\"" + a + "\",\"to\":\"" + b + "\",\"mix\":" + Num(mix) + ",\"rain\":" + Num(GET_RAIN_LEVEL()) +
        ",\"set\":\"" + g_world.weather + "\",\"transition\":" + Num(g_world.transition) + ",\"rainSet\":" + Num(g_world.rain) +
        ",\"frozen\":" + (g_world.frozen ? "true" : "false") + "}";
  return out;
}

// "density":{off (traffic off), set (multipliers held), vehicles, random, parked, peds, scenario}
std::string DensityState() {
  const Density &d = g_density;
  return "\"density\":{\"off\":" + std::string(g_noTraffic ? "true" : "false") + ",\"set\":" + (d.set ? "true" : "false") +
         ",\"vehicles\":" + Num(d.vehicles) + ",\"random\":" + Num(d.random) + ",\"parked\":" + Num(d.parked) + ",\"peds\":" + Num(d.peds) +
         ",\"scenario\":" + Num(d.scenario) + "}";
}

// "vehicle":{model (its hash, as gta5op.ini's keys), name, colours [primary, secondary, pearlescent, wheel], dirt, swaps,
// loading (a swap waiting for its model), error (the last swap's)}
std::string VehicleState(double now) {
  static std::string out;
  static double next = 0;
  static Vehicle last = 0;
  static int lastSwaps = -1, lastStep = -1;
  if (now < next && last == g_veh.handle && lastSwaps == g_swap.swaps && lastStep == g_swap.step) return out;
  next = now + 0.5;
  last = g_veh.handle;
  lastSwaps = g_swap.swaps;
  lastStep = g_swap.step;
  uint64_t c[4] = {};
  GET_VEHICLE_COLOURS(g_veh.handle, &c[0], &c[1]);
  GET_VEHICLE_EXTRA_COLOURS(g_veh.handle, &c[2], &c[3]);
  auto colour = [&](int i) { return std::to_string(static_cast<int32_t>(static_cast<uint32_t>(c[i]))); };
  out = "\"vehicle\":{\"model\":\"" + HashKey(g_veh.model) + "\",\"name\":\"" + g_veh.name + "\",\"colours\":[" + colour(0) + "," + colour(1) +
        "," + colour(2) + "," + colour(3) + "],\"dirt\":" + Num(GET_VEHICLE_DIRT_LEVEL(g_veh.handle)) + ",\"swaps\":" + std::to_string(g_swap.swaps) +
        ",\"loading\":" + (g_swap.step ? "true" : "false");
  if (!g_swap.error.empty()) out += ",\"error\":\"" + g_swap.error + "\"";
  out += "}";
  return out;
}

// "mount":{mode, source, pending (the dashcam place still to find), x, y, z (m right, forward and up of the car's
// origin), pitch, yaw (deg, up and left), base [x, y, z] before the jitter, jitter [x, y, z, pitch, yaw]}
std::string MountState() {
  const Mount &m = g_mount;
  return "\"mount\":{\"mode\":\"" + m.mode + "\",\"source\":\"" + g_veh.mountSource + "\",\"pending\":" + (g_veh.dashcamAt > 0 ? "true" : "false") +
         ",\"x\":" + Num(g_veh.mountX) + ",\"y\":" + Num(g_veh.mountY) + ",\"z\":" + Num(g_veh.mountZ) + ",\"pitch\":" + Num(g_camPitch + m.pitch) +
         ",\"yaw\":" + Num(g_camYaw + m.yaw) + ",\"base\":[" + Num(g_veh.baseX) + "," + Num(g_veh.baseY) + "," + Num(g_veh.baseZ) + "],\"jitter\":[" +
         Num(m.dx) + "," + Num(m.dy) + "," + Num(m.dz) + "," + Num(m.pitch) + "," + Num(m.yaw) + "]}";
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
      << ",\"collisions\":" << g_m.collisions << ",\"bodyHealth\":" << Num(g_m.bodyHealth)
      << ",\"camHeight\":" << Num(CameraHeight(now)) << ",\"vehicleAhead\":" << Num(VehicleAhead(now));
    float ahead = 0, left = 0, speed = 0;
    for (const std::string &part : {Route(now), Lane(now), Traffic(now), AiState(g_veh.handle), VehicleState(now), MountState(), DebugState(now),
                                    GpsState(), DirectionsState(now)})
      if (!part.empty()) s << "," << part;
    if (LeadTruth(ahead, left, speed)) s << ",\"lead\":{\"ahead\":" << Num(ahead) << ",\"left\":" << Num(left) << ",\"v\":" << Num(speed) << "}";
  }
  s << "," << WorldState(now) << "," << DensityState() << "}";
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
      // the road's direction nearer the one asked for, and the middle of lane= (from the left) that way
      if (!std::isnan(s.heading) && std::fabs(std::remainder(heading - s.heading, 360.0f)) > 90) heading += 180;
      RoadLanes r;
      if (s.lane >= 0 && RoadLanesAt(node, heading, r)) {
        float shift = r.leftEdge + (std::min(s.lane, r.n - 1) + 0.5f) * LANE_WIDTH - r.right;
        node.x += r.dirY * shift;  // the right of the road's direction (x, y) is (y, -x)
        node.y -= r.dirX * shift;
      }
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

void ApplyColours(Vehicle v, const Colours &c) {
  uint64_t slot[4] = {};
  if (c.primary >= 0 || c.secondary >= 0) {
    // the paint command's custom colours would cover the palette's
    CLEAR_VEHICLE_CUSTOM_PRIMARY_COLOUR(v);
    CLEAR_VEHICLE_CUSTOM_SECONDARY_COLOUR(v);
    GET_VEHICLE_COLOURS(v, &slot[0], &slot[1]);
    SET_VEHICLE_COLOURS(v, c.primary >= 0 ? c.primary : static_cast<int>(slot[0] & 0xFFFFFFFF), c.secondary >= 0 ? c.secondary : static_cast<int>(slot[1] & 0xFFFFFFFF));
  }
  if (c.pearl >= 0 || c.wheel >= 0) {
    GET_VEHICLE_EXTRA_COLOURS(v, &slot[2], &slot[3]);
    SET_VEHICLE_EXTRA_COLOURS(v, c.pearl >= 0 ? c.pearl : static_cast<int>(slot[2] & 0xFFFFFFFF), c.wheel >= 0 ? c.wheel : static_cast<int>(slot[3] & 0xFFFFFFFF));
  }
  if (c.dirt >= 0) SET_VEHICLE_DIRT_LEVEL(v, std::min(c.dirt, 15.0f));
}

// Replaces the player's car with one of the swap's model, where it was, facing its way at its speed (or spawns one at the
// player on foot). The old car is deleted.
void StepSwap(Ped ped, double now) {
  Swap &s = g_swap;
  if (s.step != 1) return;
  if (!HAS_MODEL_LOADED(s.model)) {
    if (now - s.t > 10) {
      s.error = "model didn't load";
      Log("vehicle: " + HashKey(s.model) + " didn't load");
      s.step = 0;
    }
    return;
  }
  s.step = 0;
  Vehicle old = IS_PED_IN_ANY_VEHICLE(ped, FALSE) ? GET_VEHICLE_PED_IS_IN(ped, FALSE) : 0;
  Entity at = old ? old : ped;
  Vector3 p = GET_ENTITY_COORDS(at, TRUE);
  float heading = GET_ENTITY_HEADING(at), speed = old ? GET_ENTITY_SPEED_VECTOR(old, TRUE).y : 0.0f;
  // above the old one until it's gone
  Vehicle v = CREATE_VEHICLE(s.model, p.x, p.y, p.z + 4.0f, heading, FALSE, FALSE, FALSE);
  SET_MODEL_AS_NO_LONGER_NEEDED(s.model);
  if (!v) {
    s.error = "vehicle creation failed";
    Log("vehicle: creation failed");
    return;
  }
  SET_ENTITY_AS_MISSION_ENTITY(v, TRUE, TRUE);  // so a later swap can delete it
  SET_VEHICLE_HAS_BEEN_OWNED_BY_PLAYER(v, TRUE);
  SET_PED_INTO_VEHICLE(ped, v, -1);
  if (old && DOES_ENTITY_EXIST(old)) {
    SET_ENTITY_AS_MISSION_ENTITY(old, TRUE, TRUE);
    DELETE_VEHICLE(&old);
  }
  SET_ENTITY_COORDS(v, p.x, p.y, p.z, FALSE, FALSE, FALSE, FALSE);
  SET_ENTITY_HEADING(v, heading);
  SET_VEHICLE_ON_GROUND_PROPERLY(v, 5.0f);
  SET_VEHICLE_ENGINE_ON(v, TRUE, TRUE, FALSE);
  if (speed > 1.0f) SET_VEHICLE_FORWARD_SPEED(v, speed);
  ApplyColours(v, s.colours);
  // the AI driver's task was for the old car: give it again for this one
  g_ai.tasked = false;
  g_ai.taskedKey.clear();
  s.swaps++;
  s.error.clear();
  Log("vehicle: swapped to " + HashKey(s.model) + " at " + Vec(p) + ", speed " + Num(speed));
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

// *** AI driver ***

void AiOn(Ped ped) {
  g_ai.on = true;
  g_aiOn = true;
  g_ai.hasTarget = g_ai.tasked = false;
  g_ai.error.clear();
  g_aiAbortPresses = 0;
  // openpilot's last command lets go: the handbrake it may hold, and its outputs in the state
  if (g_ctl.wasLive) ReleaseControls();
  g_ctl.wasLive = false;
  g_ctl.steerOut = g_ctl.throttleOut = g_ctl.brakeOut = 0;
  SET_BLOCKING_OF_NON_TEMPORARY_EVENTS(ped, TRUE);
  Log("ai on: " + g_ai.task + ", speed " + Num(g_ai.speed) + ", style " + std::to_string(g_ai.style) + ", ability " + Num(g_ai.ability) +
      ", aggressiveness " + Num(g_ai.aggressiveness));
}

void AiOff(Ped ped, const std::string &why) {
  if (!g_ai.on) return;
  if (g_ai.tasked) CLEAR_PED_TASKS(ped);  // leaves a driver in its seat, unlike the _IMMEDIATELY version
  SET_PED_KEEP_TASK(ped, FALSE);
  SET_BLOCKING_OF_NON_TEMPORARY_EVENTS(ped, FALSE);
  g_ai.on = g_ai.tasked = false;
  g_aiOn = false;
  g_ai.status = 7;
  g_ai.mode.clear();
  Log("ai off (" + why + ")");
}

float Dist2d(const Vector3 &a, const Vector3 &b) { return std::hypot(a.x - b.x, a.y - b.y); }

void StepAi(Ped ped, Vehicle v, double now) {
  if (g_aiAbortPresses.exchange(0) && g_ai.on) {
    g_ai.aborts++;
    AiOff(ped, "engage key");
  }
  if (!g_ai.on) return;
  if (!v || GET_PED_IN_VEHICLE_SEAT(v, -1, FALSE) != ped) {
    g_ai.error = v ? "not the driver" : "not in a vehicle";
    AiOff(ped, g_ai.error);
    return;
  }
  for (int input : {INPUT_VEH_MOVE_LR, INPUT_VEH_ACCELERATE, INPUT_VEH_BRAKE, INPUT_VEH_HANDBRAKE, INPUT_VEH_EXIT})
    DISABLE_CONTROL_ACTION(0, input, TRUE);
  // the task's own indicators come and go; the bridge's stay on, so the frames match its desire labels
  SET_VEHICLE_INDICATOR_LIGHTS(v, 1, g_indicator == 1);
  SET_VEHICLE_INDICATOR_LIGHTS(v, 0, g_indicator == 2);

  Vector3 w{};
  bool hasWaypoint = WaypointAt(w);
  bool wander = g_ai.task == "wander" || (!g_ai.hasTarget && !hasWaypoint);
  g_ai.mode = wander ? "wander" : g_ai.hasTarget ? "target" : "waypoint";
  Vector3 target = g_ai.target;
  if (g_ai.mode == "waypoint") {
    if (now - g_ai.waypointAt > 1.0) {
      // the road node nearest the waypoint, whose blip has no height
      Vector3 node{};
      float heading = 0;
      g_ai.waypoint = GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(w.x, w.y, w.z, &node, &heading, 1, 3.0f, 0.0f) ? node : w;
      g_ai.waypointAt = now;
    }
    target = g_ai.waypoint;
  }
  std::string task = wander ? "wander" : g_ai.task == "coord" ? "coord" : "longrange";
  static const Hash taskHashes[3] = {GET_HASH_KEY("SCRIPT_TASK_VEHICLE_DRIVE_WANDER"), GET_HASH_KEY("SCRIPT_TASK_VEHICLE_DRIVE_TO_COORD"),
                                     GET_HASH_KEY("SCRIPT_TASK_VEHICLE_DRIVE_TO_COORD_LONGRANGE")};
  Hash hash = taskHashes[task == "wander" ? 0 : task == "coord" ? 1 : 2];
  g_ai.status = g_ai.tasked ? GET_SCRIPT_TASK_STATUS(ped, hash) : 7;

  std::string key = task + ":" + g_ai.mode;
  bool changed = key != g_ai.taskedKey || (!wander && Dist2d(target, g_ai.taskedTarget) + std::fabs(target.z - g_ai.taskedTarget.z) > 1.0f);
  // a finished task arrived or gave up: again, now and then, unless it's there
  bool retry = g_ai.tasked && g_ai.status == 7 && now - g_ai.taskedAt > 2.0 && (wander || Dist2d(g_m.pos, target) > g_ai.stopRange + 5.0f);
  if (!g_ai.tasked || (changed && (key != g_ai.taskedKey || now - g_ai.taskedAt > 0.5)) || retry) {
    SET_DRIVER_ABILITY(ped, g_ai.ability);
    SET_DRIVER_AGGRESSIVENESS(ped, g_ai.aggressiveness);
    if (task == "wander") TASK_VEHICLE_DRIVE_WANDER(ped, v, g_ai.speed, g_ai.style);
    else if (task == "coord")
      TASK_VEHICLE_DRIVE_TO_COORD(ped, v, target.x, target.y, target.z, g_ai.speed, 0, GET_ENTITY_MODEL(v), g_ai.style, g_ai.stopRange, g_ai.straightLine);
    else TASK_VEHICLE_DRIVE_TO_COORD_LONGRANGE(ped, v, target.x, target.y, target.z, g_ai.speed, g_ai.style, g_ai.stopRange);
    SET_PED_KEEP_TASK(ped, TRUE);
    if (!g_ai.tasked || retry || key != g_ai.taskedKey)
      Log("ai: " + key + " to " + Vec(target) + (retry ? " (again)" : ""));
    g_ai.tasked = true;
    g_ai.taskedKey = key;
    g_ai.taskedTarget = target;
    g_ai.taskedAt = now;
    g_ai.retasks++;
    g_ai.speedPending = true;  // once the task runs: it may not take them before
    g_ai.status = 0;
  }
  if (g_ai.status == 1 && (g_ai.speedPending || g_ai.stylePending)) {
    SET_DRIVE_TASK_CRUISE_SPEED(ped, g_ai.speed);
    SET_DRIVE_TASK_MAX_CRUISE_SPEED(ped, g_ai.speed, TRUE);
    if (g_ai.stylePending) SET_DRIVE_TASK_DRIVING_STYLE(ped, g_ai.style);
    g_ai.speedPending = g_ai.stylePending = false;
  }
}

// "ai":{on, mode, task, status (the game's: 0 starting, 1 running, 7 finished), target, speed, style, retasks, aborts (the
// engage key), stoppedAtLight, error}
std::string AiState(Vehicle v) {
  std::string s = "\"ai\":{\"on\":" + std::string(g_ai.on ? "true" : "false");
  if (g_ai.on) {
    s += ",\"mode\":\"" + g_ai.mode + "\",\"task\":\"" + g_ai.task + "\",\"status\":" + std::to_string(g_ai.status) + ",\"speed\":" + Num(g_ai.speed) +
         ",\"style\":" + std::to_string(g_ai.style) + ",\"retasks\":" + std::to_string(g_ai.retasks) +
         ",\"stoppedAtLight\":" + (IS_VEHICLE_STOPPED_AT_TRAFFIC_LIGHTS(v) ? "true" : "false");
    if (g_ai.tasked && g_ai.mode != "wander") s += ",\"target\":" + Vec(g_ai.taskedTarget);
  }
  s += ",\"aborts\":" + std::to_string(g_ai.aborts);
  if (!g_ai.error.empty()) s += ",\"error\":\"" + g_ai.error + "\"";
  return s + "}";
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
    SetIndicator(0);
  } else if (type == "setIndicator") {
    std::string side = MsgStr(m, "side");
    SetIndicator(side == "left" ? 1 : side == "right" ? 2 : 0);
  } else if (type == "camera") {
    g_camPitch = static_cast<float>(MsgNum(m, "pitch", g_camPitch));
    g_camYaw = static_cast<float>(MsgNum(m, "yaw", g_camYaw));
    // mount position for this car, meters: forward of the vehicle origin, and up from the ground
    if (m.count("forward") && g_veh.handle) {
      g_veh.baseY = static_cast<float>(MsgNum(m, "forward"));
      WritePrivateProfileStringA("mount_forward", ModelKey(g_veh.handle).c_str(), Num(g_veh.baseY).c_str(), IniPath().c_str());
    }
    if (m.count("height")) {
      float height = static_cast<float>(MsgNum(m, "height"));
      g_veh.baseZ += height - g_cfg.mountHeight;
      g_cfg.mountHeight = height;
    }
    ApplyMount();
    Log("camera: forward " + Num(g_veh.mountY) + ", up " + Num(g_veh.mountZ) + ", pitch " + Num(g_camPitch) + ", yaw " + Num(g_camYaw));
    if (g_cam) AttachCamera();
  } else if (type == "mount") {
    // where the camera goes on each car: mode=comma|dashcam; drop, back (m, dashcam: below the roof, behind the glass);
    // the jitter added to it: dx, dy, dz (m right, forward, up), pitch, yaw (deg); jitter=0 clears it
    std::string mode = MsgStr(m, "mode", g_mount.mode);
    if (mode == "comma" || mode == "dashcam") g_mount.mode = mode;
    g_mount.drop = std::clamp(static_cast<float>(MsgNum(m, "drop", g_mount.drop)), 0.0f, 0.5f);
    g_mount.back = std::clamp(static_cast<float>(MsgNum(m, "back", g_mount.back)), 0.0f, 0.5f);
    if (m.count("jitter") && !MsgBool(m, "jitter")) g_mount.dx = g_mount.dy = g_mount.dz = g_mount.pitch = g_mount.yaw = 0;
    g_mount.dx = std::clamp(static_cast<float>(MsgNum(m, "dx", g_mount.dx)), -0.2f, 0.2f);
    g_mount.dy = std::clamp(static_cast<float>(MsgNum(m, "dy", g_mount.dy)), -0.2f, 0.2f);
    g_mount.dz = std::clamp(static_cast<float>(MsgNum(m, "dz", g_mount.dz)), -0.2f, 0.2f);
    g_mount.pitch = std::clamp(static_cast<float>(MsgNum(m, "pitch", g_mount.pitch)), -10.0f, 10.0f);
    g_mount.yaw = std::clamp(static_cast<float>(MsgNum(m, "yaw", g_mount.yaw)), -10.0f, 10.0f);
    if (g_veh.handle) {
      CommaMount();
      ApplyMount();
      g_veh.dashcamAt = g_mount.mode == "dashcam" ? now : 0;
    }
    Log("mount " + g_mount.mode + ", drop " + Num(g_mount.drop) + ", back " + Num(g_mount.back) + ", jitter " + Num(g_mount.dx) + "," +
        Num(g_mount.dy) + "," + Num(g_mount.dz) + " m, " + Num(g_mount.pitch) + "," + Num(g_mount.yaw) + " deg");
  } else if (type == "vehicle") {
    // model= (a name, or a hash as 0x...) replaces the player's car with a new one of it, where it is and at its speed,
    // unless it's that model already (force=1 replaces it anyway); primary, secondary, pearl, wheel (palette indices) and
    // dirt (0-15) colour the new car, or with no model the current one
    Colours c;
    c.primary = static_cast<int>(MsgNum(m, "primary", -1));
    c.secondary = static_cast<int>(MsgNum(m, "secondary", -1));
    c.pearl = static_cast<int>(MsgNum(m, "pearl", -1));
    c.wheel = static_cast<int>(MsgNum(m, "wheel", -1));
    c.dirt = static_cast<float>(MsgNum(m, "dirt", -1));
    std::string model = MsgStr(m, "model");
    Hash hash = model.empty() ? 0 : model.rfind("0x", 0) == 0 ? static_cast<Hash>(std::strtoul(model.c_str(), nullptr, 16)) : GET_HASH_KEY(model.c_str());
    if (!hash || (g_veh.handle && hash == g_veh.model && !MsgBool(m, "force"))) {
      if (g_veh.handle) ApplyColours(g_veh.handle, c);
      g_swap.error.clear();
    } else if (!IS_MODEL_IN_CDIMAGE(hash) || !IS_MODEL_A_VEHICLE(hash)) {
      g_swap.error = "no vehicle model " + HashKey(hash);
      Log("vehicle: " + g_swap.error);
    } else {
      REQUEST_MODEL(hash);
      g_swap.step = 1;
      g_swap.model = hash;
      g_swap.colours = c;
      g_swap.t = now;
    }
  } else if (type == "interleave") {
    g_cfg.interleave = MsgBool(m, "on", g_cfg.interleave);
    g_cfg.interleaveLag = std::clamp(static_cast<int>(MsgNum(m, "lag", g_cfg.interleaveLag)), 0, 8);
    g_cfg.presentHook = MsgBool(m, "hook", g_cfg.presentHook);
    g_cfg.splitViews = MsgBool(m, "split", g_cfg.splitViews);
    g_cfg.wideDelay = static_cast<float>(MsgNum(m, "wide_delay", g_cfg.wideDelay));
    Log("interleave " + std::string(g_cfg.interleave ? "on" : "off") + ", lag " + std::to_string(g_cfg.interleaveLag) +
        (g_cfg.splitViews ? ", split views" : ""));
  } else if (type == "debug") {
    // the map debug overlay: on, layers (gta5_overlay.py's letters), force (also while recording), ground (on the
    // game's ground), lift (m above the road)
    g_debug.on = MsgBool(m, "on", g_debug.on);
    std::string layers = MsgStr(m, "layers");
    if (!layers.empty()) g_debug.layers = layers;
    g_debug.force = MsgBool(m, "force", g_debug.force);
    g_debug.ground = MsgBool(m, "ground", g_debug.ground);
    g_debug.lift = std::clamp(static_cast<float>(MsgNum(m, "lift", g_debug.lift)), -1.0f, 3.0f);
    if (!g_debug.on) g_debug.lines.clear(), g_debug.vertices = 0;
    Log("debug " + std::string(g_debug.on ? "on" : "off") + ", layers " + g_debug.layers + (g_debug.force ? ", forced" : "") +
        (g_debug.ground ? ", on the ground" : ""));
  } else if (type == "debugGeo") {
    if (!g_debug.on) return;
    ParseDebugGeo(MsgStr(m, "g"), static_cast<float>(MsgNum(m, "ox")), static_cast<float>(MsgNum(m, "oy")), static_cast<float>(MsgNum(m, "oz")),
                  g_debug.lines, g_debug.vertices);
    if (g_debug.ground) SnapToGround(g_debug.lines);
    g_debug.recording = MsgBool(m, "rec");
    g_debug.t = now;
  } else if (type == "gpsroute") {
    // our route on the game's map: on, colour (HUD colour), max (points), radar and map (line widths), take (the map's
    // waypoint held off the map meanwhile, as GTA's own route line to it can't be hidden; take=0 leaves both lines)
    g_gps.on = MsgBool(m, "on", g_gps.on);
    g_gps.colour = static_cast<int>(MsgNum(m, "colour", g_gps.colour));
    g_gps.max = std::clamp(static_cast<int>(MsgNum(m, "max", g_gps.max)), 2, 2000);
    g_gps.take = MsgBool(m, "take", g_gps.take);
    g_gps.radar = static_cast<int>(MsgNum(m, "radar", g_gps.radar));
    g_gps.map = static_cast<int>(MsgNum(m, "map", g_gps.map));
    if (!g_gps.on) g_gps.pts.clear();
    ShowGpsRoute();
    Log("gps route " + std::string(g_gps.on ? "on" : "off") + ", colour " + std::to_string(g_gps.colour) + ", max " + std::to_string(g_gps.max));
  } else if (type == "gpsPoints") {
    if (!g_gps.on) return;
    g_gps.pts = ParsePoints(MsgStr(m, "p"));
    ShowGpsRoute();
  } else if (type == "gtadirs") {
    g_dirs.on = MsgBool(m, "on", true);
    g_dirs.x = static_cast<float>(MsgNum(m, "x", g_dirs.x));
    g_dirs.y = static_cast<float>(MsgNum(m, "y", g_dirs.y));
    g_dirs.z = static_cast<float>(MsgNum(m, "z", g_dirs.z));
    g_dirs.next = 0;
    g_dirs.direction = -1;
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
  } else if (type == "cruise") {
    // as if the cruise speed keys were pressed: up or down, by 5 with five=1
    auto &k = MsgStr(m, "dir") == "up" ? g_speedUp : g_speedDown;
    (MsgBool(m, "five") ? k.presses5 : k.presses) += static_cast<int>(MsgNum(m, "times", 1));
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
  } else if (type == "waypoint") {
    ReleaseWaypoint(false);  // a held one is replaced or cleared too
    if (MsgBool(m, "off")) SET_WAYPOINT_OFF();
    else SET_NEW_WAYPOINT(static_cast<float>(MsgNum(m, "x")), static_cast<float>(MsgNum(m, "y")));
  } else if (type == "ai") {
    // the AI driver: on=1|0; x, y, z its target (clear=1 drops it: the waypoint, else wandering); speed (m/s cap), style,
    // ability, aggr, stop (m, the target's stop range), task (longrange, coord or wander), straight (m, coord's straight
    // line distance), indicator (left, right or off)
    Ped ped = PLAYER_PED_ID();
    if (m.count("on")) {
      if (!MsgBool(m, "on")) AiOff(ped, "asked");
      else if (!g_ai.on) AiOn(ped);
    }
    if (m.count("speed")) {
      g_ai.speed = std::clamp(static_cast<float>(MsgNum(m, "speed")), 0.0f, 70.0f);
      g_ai.speedPending = true;
    }
    if (m.count("style")) {
      g_ai.style = static_cast<int>(static_cast<int64_t>(MsgNum(m, "style")));
      g_ai.stylePending = true;
    }
    g_ai.ability = std::clamp(static_cast<float>(MsgNum(m, "ability", g_ai.ability)), 0.0f, 1.0f);
    g_ai.aggressiveness = std::clamp(static_cast<float>(MsgNum(m, "aggr", g_ai.aggressiveness)), 0.0f, 1.0f);
    g_ai.stopRange = static_cast<float>(MsgNum(m, "stop", g_ai.stopRange));
    g_ai.straightLine = static_cast<float>(MsgNum(m, "straight", g_ai.straightLine));
    if (m.count("task")) g_ai.task = MsgStr(m, "task");
    if (MsgBool(m, "clear")) g_ai.hasTarget = false;
    if (m.count("x") && m.count("y")) {
      g_ai.target.x = static_cast<float>(MsgNum(m, "x"));
      g_ai.target.y = static_cast<float>(MsgNum(m, "y"));
      g_ai.target.z = static_cast<float>(MsgNum(m, "z", g_m.pos.z));
      g_ai.hasTarget = true;
    }
    if (m.count("indicator")) {
      std::string side = MsgStr(m, "indicator");
      SetIndicator(side == "left" ? 1 : side == "right" ? 2 : 0);
    }
  } else if (type == "gas") {
    g_testGasUntil = now + MsgNum(m, "secs", 1.0);  // as if the driver pressed the gas
  } else if (type == "traffic") {
    // on=0 clears and stops traffic; density multipliers, held each frame until changed: vehicles (random= follows it
    // unless given), parked, peds (scenario= follows it unless given); reset=1 lets the game choose again
    g_noTraffic = !MsgBool(m, "on", true);
    if (g_noTraffic) ClearModel(0, 300.0f);  // any model
    Density &d = g_density;
    auto mult = [&](const char *key, float fallback) { return std::clamp(static_cast<float>(MsgNum(m, key, fallback)), 0.0f, 3.0f); };
    if (MsgBool(m, "reset")) d = Density{};
    if (m.count("vehicles") || m.count("random") || m.count("parked") || m.count("peds") || m.count("scenario")) {
      d.set = true;
      d.vehicles = mult("vehicles", d.vehicles);
      d.random = mult("random", m.count("vehicles") ? d.vehicles : d.random);
      d.parked = mult("parked", d.parked);
      d.peds = mult("peds", d.peds);
      d.scenario = mult("scenario", m.count("peds") ? d.peds : d.scenario);
    }
    Log(std::string("traffic ") + (g_noTraffic ? "off" : "on") + (d.set ? ", vehicles " + Num(d.vehicles) + ", random " + Num(d.random) +
        ", parked " + Num(d.parked) + ", peds " + Num(d.peds) + ", scenario " + Num(d.scenario) : ""));
  } else if (type == "world") {
    // a repeatable scene for tests: the time of day (hour, minute), the weather held (weather=, at once or over
    // transition= s; clear=1 lets the game's weather cycle again), the rain and puddles (rain=0-1, -1 the weather's
    // own), and the clock stopped with freeze=1
    if (m.count("hour"))
      SET_CLOCK_TIME(std::clamp(static_cast<int>(MsgNum(m, "hour")), 0, 23), std::clamp(static_cast<int>(MsgNum(m, "minute", 0)), 0, 59), 0);
    if (MsgBool(m, "clear")) {
      CLEAR_OVERRIDE_WEATHER();
      CLEAR_WEATHER_TYPE_PERSIST();
      g_world.weather.clear();
    }
    std::string weather;
    for (char c : MsgStr(m, "weather"))
      if (isalnum(static_cast<unsigned char>(c)) || c == '_') weather += static_cast<char>(toupper(static_cast<unsigned char>(c)));
    if (!weather.empty()) {
      g_world.weather = weather;
      g_world.transition = std::max(0.0f, static_cast<float>(MsgNum(m, "transition", 0)));
      if (g_world.transition > 0) {
        CLEAR_OVERRIDE_WEATHER();  // an override holds the weather it names, at once
        SET_WEATHER_TYPE_OVERTIME_PERSIST(weather.c_str(), g_world.transition);
      } else {
        SET_WEATHER_TYPE_NOW_PERSIST(weather.c_str());
        SET_OVERRIDE_WEATHER(weather.c_str());
      }
    }
    if (m.count("rain")) {
      g_world.rain = std::clamp(static_cast<float>(MsgNum(m, "rain")), -1.0f, 1.0f);
      SET_RAIN(g_world.rain < 0 ? -1.0f : g_world.rain);
    }
    if (m.count("freeze")) {
      g_world.frozen = MsgBool(m, "freeze");
      PAUSE_CLOCK(g_world.frozen);
    }
    Log("world: hour " + Num(MsgNum(m, "hour", -1)) + ", weather " + weather + ", transition " + Num(g_world.transition) + ", rain " +
        Num(MsgNum(m, "rain", -2)) + ", freeze " + Num(MsgNum(m, "freeze", -1)));
  } else if (type == "setup") {
    Setup s;
    s.x = static_cast<float>(MsgNum(m, "x", NAN));
    s.y = static_cast<float>(MsgNum(m, "y", NAN));
    s.z = static_cast<float>(MsgNum(m, "z", NAN));
    s.speed = static_cast<float>(MsgNum(m, "speed", 0));
    s.minLanes = static_cast<int>(MsgNum(m, "lanes", 0));
    s.heading = static_cast<float>(MsgNum(m, "heading", NAN));
    s.lane = static_cast<int>(MsgNum(m, "lane", -1));
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
  g_gps.on = g_cfg.gpsRoute;
  Log("init: bridge " + (g_cfg.bridge.empty() ? std::string("from ") + BRIDGE_FILE : g_cfg.bridge) + ", vfov " + Num(g_cfg.vfov) +
      ", lens " + (g_cfg.lens ? "on" : "off"));
  g_link.Start(BridgeAddress, Log);
}

extern "C" __declspec(dllexport) void CoreTick() {
  double now = QpcSeconds();
  float dt = GET_FRAME_TIME();
  for (auto &m : g_link.TakeMessages()) HandleMessage(m, now);

  static bool scriptHook = false;
  if (!scriptHook) scriptHook = (script_hook::Install(Log), true);
  script_hook::Update();

  Ped ped = PLAYER_PED_ID();
  if (g_setup.step) StepSetup(ped, now);
  StepLead(now);
  StepSwap(ped, now);
  if (g_noTraffic || g_density.set) {
    const Density &d = g_noTraffic ? Density{true, 0, 0, 0, 0, 0} : g_density;
    SET_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(d.vehicles);
    SET_RANDOM_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(d.random);
    SET_PARKED_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(d.parked);
    SET_PED_DENSITY_MULTIPLIER_THIS_FRAME(d.peds);
    SET_SCENARIO_PED_DENSITY_MULTIPLIER_THIS_FRAME(d.scenario, d.scenario);
  }
  Vehicle v = IS_PED_IN_ANY_VEHICLE(ped, FALSE) ? GET_VEHICLE_PED_IS_IN(ped, FALSE) : 0;
  if (v != g_veh.handle) OnVehicleChanged(v);
  if (v && g_veh.dashcamAt > 0 && now >= g_veh.dashcamAt) {
    g_veh.dashcamAt = 0;
    if (g_mount.mode == "dashcam") DashcamMount();
  }

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

  script_hook::SetOverride(g_cam != 0);
  int view = g_cam ? UpdateCameraFrame(now, dt, hooked && g_cfg.splitViews) : -1;
  if (view >= 0) {
    // help text would cover the marker, and the wide lens reaches the radar in the corner. Hiding the rest of the HUD
    // or the feed for a frame restarts their animations (the radio station's name never shows), so the feed is hidden
    // only where the frames reach the screen.
    HIDE_HELP_TEXT_THIS_FRAME();
    if (!IS_RADAR_HIDDEN()) {
      DISPLAY_RADAR(FALSE);
      g_radarHidden = true;
    }
    if (!hooked) THEFEED_HIDE_THIS_FRAME();
    // tells the capture this frame is the openpilot camera's, and which view by its colour (magenta both, cyan road,
    // yellow wide); the lens resampling blacks it out
    if (g_cfg.interleave)
      DRAW_RECT(0.0025f, 0.0045f, 0.005f, 0.009f, view == HOOK_ROAD ? 0 : 255, view == HOOK_BOTH ? 0 : 255, view == HOOK_WIDE ? 0 : 255, 255, FALSE);
  } else if (g_radarHidden) {
    DISPLAY_RADAR(TRUE);
    g_radarHidden = false;
  }
  // sounds are heard from the rendering camera, as the game sees it a frame late
  static bool lastOp = false;
  if (view >= 0 || lastOp) FREEZE_MICROPHONE();
  lastOp = view >= 0;
  // on the player's frames only, which keeps it out of the openpilot camera's
  if (connected && v && view < 0) DrawSpeed(now);
  if (g_debugPresses.exchange(0) % 2) {
    g_debug.on = !g_debug.on;
    if (!g_debug.on) g_debug.lines.clear(), g_debug.vertices = 0;
    Log(std::string("debug ") + (g_debug.on ? "on" : "off") + " (key)");
  }
  // never with the marker: openpilot's frames are the ones the capture finds the marker in, so these never reach them
  if (view < 0) DrawDebug(now);
  TakeWaypoint(now);
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
  }
  StepAi(ped, v, now);
  // the AI drives alone: openpilot's commands, even a stale one still arriving, wait until it's off
  if (v && !g_ai.on) ApplyControls(dt, now);
  Publish(now, v != 0);
}

extern "C" __declspec(dllexport) void CoreShutdown() {
  AiOff(PLAYER_PED_ID(), "shutdown");  // a task left running would keep driving under the reloaded core
  ReleaseControls();
  ReleaseCamera();
  RemoveLead();
  script_hook::Uninstall();
  if (g_radarHidden) DISPLAY_RADAR(TRUE);
  g_gps.on = false;  // the route goes with this core; a reloaded one starts from gps_route in the ini
  ShowGpsRoute();
  ReleaseWaypoint(true);  // and gives the map its waypoint back
  if (g_hook == Hook::On) present_hook::Uninstall();
  g_capture.Stop();
  g_link.Stop();
  Log("shutdown");
}

extern "C" __declspec(dllexport) void CoreKey(DWORD key) {
  if (static_cast<int>(key) == g_cfg.keyEngage) (g_aiOn ? g_aiAbortPresses : g_engagePresses)++;
  else if (static_cast<int>(key) == g_cfg.keyLeft) g_leftPresses++;
  else if (static_cast<int>(key) == g_cfg.keyRight) g_rightPresses++;
  else if (static_cast<int>(key) == g_cfg.keyDebug) g_debugPresses++;
}
