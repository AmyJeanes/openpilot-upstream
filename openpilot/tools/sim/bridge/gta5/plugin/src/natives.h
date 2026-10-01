// The game script natives the plugin uses (hashes and names from alloc8or/gta5-nativedb-data).
#pragma once
#include "shv.h"

namespace nat {
using shv::Invoke;
using shv::Vector3;
using Entity = int;
using Ped = int;
using Vehicle = int;
using Cam = int;
using Hash = uint32_t;
using BOOL = int;

// player and entities
inline Ped PLAYER_PED_ID() { return Invoke<Ped>(0xD80958FC74E988A6); }
inline int PLAYER_ID() { return Invoke<int>(0x4F8644AF03D0E0D6); }
inline BOOL IS_PED_IN_ANY_VEHICLE(Ped p, BOOL atGetIn) { return Invoke<BOOL>(0x997ABD671D25CA0B, p, atGetIn); }
inline Vehicle GET_VEHICLE_PED_IS_IN(Ped p, BOOL includeEntering) { return Invoke<Vehicle>(0x9A9112A0FE9A4713, p, includeEntering); }
inline BOOL DOES_ENTITY_EXIST(Entity e) { return Invoke<BOOL>(0x7239B21A38F536BA, e); }
inline Vector3 GET_ENTITY_COORDS(Entity e, BOOL alive) { return Invoke<Vector3>(0x3FEF770D40960D5A, e, alive); }
inline Vector3 GET_ENTITY_ROTATION(Entity e, int order) { return Invoke<Vector3>(0xAFBD61CC738D9EB9, e, order); }
inline Vector3 GET_ENTITY_SPEED_VECTOR(Entity e, BOOL relative) { return Invoke<Vector3>(0x9A8D700A51CB7B0D, e, relative); }
inline Vector3 GET_ENTITY_ROTATION_VELOCITY(Entity e) { return Invoke<Vector3>(0x213B91045D09B983, e); }
inline float GET_ENTITY_HEADING(Entity e) { return Invoke<float>(0xE83D4F9BA2A38914, e); }
inline Hash GET_ENTITY_MODEL(Entity e) { return Invoke<Hash>(0x9F47B058362C84B5, e); }
inline void GET_MODEL_DIMENSIONS(Hash model, Vector3 *mn, Vector3 *mx) { Invoke(0x03E8D3D5F549087A, model, mn, mx); }
inline int GET_ENTITY_BONE_INDEX_BY_NAME(Entity e, const char *bone) { return Invoke<int>(0xFB71170B7E76ACBA, e, bone); }
inline Vector3 GET_WORLD_POSITION_OF_ENTITY_BONE(Entity e, int bone) { return Invoke<Vector3>(0x44A8FCB8ED227738, e, bone); }
inline Vector3 GET_ENTITY_BONE_OBJECT_ROTATION(Entity e, int bone) { return Invoke<Vector3>(0xBD8D32550E5CEBFE, e, bone); }
inline Vector3 GET_OFFSET_FROM_ENTITY_GIVEN_WORLD_COORDS(Entity e, float x, float y, float z) { return Invoke<Vector3>(0x2274BC1C4885E333, e, x, y, z); }
inline void SET_ENTITY_LOCALLY_INVISIBLE(Entity e) { Invoke(0xE135A9FF3F5D05D8, e); }
inline void SET_ENTITY_COORDS(Entity e, float x, float y, float z, BOOL xa, BOOL ya, BOOL za, BOOL clear) { Invoke(0x06843DA7060A026B, e, x, y, z, xa, ya, za, clear); }
inline void SET_ENTITY_HEADING(Entity e, float h) { Invoke(0x8E2530AA8ADA980E, e, h); }
inline void SET_ENTITY_AS_MISSION_ENTITY(Entity e, BOOL a, BOOL b) { Invoke(0xAD738C3085FE7E11, e, a, b); }
inline void FREEZE_ENTITY_POSITION(Entity e, BOOL t) { Invoke(0x428CA6DBD1094446, e, t); }

// vehicles
inline Hash GET_HASH_KEY(const char *s) { return Invoke<Hash>(0xD24D37CC275948CC, s); }
inline void REQUEST_MODEL(Hash m) { Invoke(0x963D27A58DF860AC, m); }
inline BOOL HAS_MODEL_LOADED(Hash m) { return Invoke<BOOL>(0x98A4EB5D89A0C952, m); }
inline void SET_MODEL_AS_NO_LONGER_NEEDED(Hash m) { Invoke(0xE532F5D78798DAAB, m); }
inline Vehicle CREATE_VEHICLE(Hash m, float x, float y, float z, float heading, BOOL net, BOOL host, BOOL p7) { return Invoke<Vehicle>(0xAF35D0D2583051B0, m, x, y, z, heading, net, host, p7); }
inline void SET_PED_INTO_VEHICLE(Ped p, Vehicle v, int seat) { Invoke(0xF75B0D629E1C063D, p, v, seat); }
inline BOOL SET_VEHICLE_ON_GROUND_PROPERLY(Vehicle v, float p1) { return Invoke<BOOL>(0x49733E92263139D1, v, p1); }
inline void SET_VEHICLE_ENGINE_ON(Vehicle v, BOOL on, BOOL instantly, BOOL noAutoStart) { Invoke(0x2497C4717C8B881E, v, on, instantly, noAutoStart); }
inline void SET_VEHICLE_HANDBRAKE(Vehicle v, BOOL on) { Invoke(0x684785568EF26A22, v, on); }
inline void SET_VEHICLE_INDICATOR_LIGHTS(Vehicle v, int turnSignal, BOOL on) { Invoke(0xB5D45264751B7DF0, v, turnSignal, on); }
// holds the steering at a fraction of full lock (left positive) for this frame
inline void SET_VEHICLE_STEER_BIAS(Vehicle v, float bias) { Invoke(0x42A8EC77D5150CBE, v, bias); }
inline void SET_VEHICLE_FIXED(Vehicle v) { Invoke(0x115722B1B9C14C1C, v); }
inline void SET_VEHICLE_FORWARD_SPEED(Vehicle v, float s) { Invoke(0xAB54A438726D25D5, v, s); }

// world and paths
inline BOOL LOAD_ALL_PATH_NODES(BOOL all) { return Invoke<BOOL>(0xC2AB6BFE34E92F8B, all); }
inline void REQUEST_COLLISION_AT_COORD(float x, float y, float z) { Invoke(0x07503F7948F491A7, x, y, z); }
inline BOOL GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(float x, float y, float z, Vector3 *pos, float *heading, int nodeType, float p6, float p7) {
  return Invoke<BOOL>(0xFF071FB798B803B0, x, y, z, pos, heading, nodeType, p6, p7);
}
inline BOOL GET_NTH_CLOSEST_VEHICLE_NODE_WITH_HEADING(float x, float y, float z, int n, Vector3 *pos, float *heading, int *lanes, int flags, float p8, float p9) {
  return Invoke<BOOL>(0x80CA6A8B6C094CC4, x, y, z, n, pos, heading, lanes, flags, p8, p9);
}
inline void SET_CLOCK_TIME(int h, int m, int s) { Invoke(0x47C3B5848C3E45D8, h, m, s); }
inline void SET_WEATHER_TYPE_NOW_PERSIST(const char *w) { Invoke(0xED712CA327900C8A, w); }
inline void SET_MAX_WANTED_LEVEL(int lvl) { Invoke(0xAA5F02DB48D704B9, lvl); }
inline void CLEAR_PLAYER_WANTED_LEVEL(int player) { Invoke(0xB302540597885499, player); }
inline BOOL IS_PAUSE_MENU_ACTIVE() { return Invoke<BOOL>(0xB0034A223497FFCB); }
inline float GET_FRAME_TIME() { return Invoke<float>(0x15C40837039FFAF7); }

// camera and HUD
inline Cam CREATE_CAM(const char *name, BOOL p1) { return Invoke<Cam>(0xC3981DCE61D9E13F, name, p1); }
inline void DESTROY_CAM(Cam c, BOOL p1) { Invoke(0x865908C81A2C22E9, c, p1); }
inline BOOL DOES_CAM_EXIST(Cam c) { return Invoke<BOOL>(0xA7A932170592B50E, c); }
inline void SET_CAM_ACTIVE(Cam c, BOOL a) { Invoke(0x026FB97D0A425F84, c, a); }
inline void SET_CAM_FOV(Cam c, float fov) { Invoke(0xB13C14F66A00D047, c, fov); }
inline void SET_CAM_NEAR_CLIP(Cam c, float d) { Invoke(0xC7848EFCCC545182, c, d); }
inline void HARD_ATTACH_CAM_TO_ENTITY(Cam c, Entity e, float xr, float yr, float zr, float x, float y, float z, BOOL relative) {
  Invoke(0x202A5ED9CE01D6E7, c, e, xr, yr, zr, x, y, z, relative);
}
inline void RENDER_SCRIPT_CAMS(BOOL render, BOOL ease, int easeTime, BOOL p3, BOOL p4, int p5) { Invoke(0x07E5B515DB0636FC, render, ease, easeTime, p3, p4, p5); }
inline void HIDE_HUD_AND_RADAR_THIS_FRAME() { Invoke(0x719FF505F097FD20); }
inline void HIDE_HELP_TEXT_THIS_FRAME() { Invoke(0xD46923FC481CA285); }
inline void THEFEED_HIDE_THIS_FRAME() { Invoke(0x25F87B30C382FCA7); }
inline void CLEAR_PRINTS() { Invoke(0xCC33FA791322B9D9); }

// controls
inline BOOL SET_CONTROL_VALUE_NEXT_FRAME(int control, int action, float v) { return Invoke<BOOL>(0xE8A25867FBA3B05E, control, action, v); }
inline void DISABLE_CONTROL_ACTION(int control, int action, BOOL related) { Invoke(0xFE99B66D079CF6BC, control, action, related); }
inline float GET_DISABLED_CONTROL_NORMAL(int control, int action) { return Invoke<float>(0x11E65974A982637C, control, action); }
inline float GET_CONTROL_NORMAL(int control, int action) { return Invoke<float>(0xEC3C9B8D5327B563, control, action); }

enum Input {
  INPUT_VEH_MOVE_LR = 59,
  INPUT_VEH_ACCELERATE = 71,
  INPUT_VEH_BRAKE = 72,
  INPUT_VEH_HANDBRAKE = 76,
};

}  // namespace nat
