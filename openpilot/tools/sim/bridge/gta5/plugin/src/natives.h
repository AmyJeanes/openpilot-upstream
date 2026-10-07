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
inline Vector3 GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(Entity e, float x, float y, float z) { return Invoke<Vector3>(0x1899F328B0E12848, e, x, y, z); }
inline void DELETE_VEHICLE(Vehicle *v) { Invoke(0xEA386986E786A54F, v); }
inline Vehicle GET_CLOSEST_VEHICLE(float x, float y, float z, float radius, Hash model, int flags) { return Invoke<Vehicle>(0xF73EB622C4F1689B, x, y, z, radius, model, flags); }
inline void DELETE_PED(Ped *p) { Invoke(0x9614299DCB53E54B, p); }
inline Ped CREATE_RANDOM_PED_AS_DRIVER(Vehicle v, BOOL returnHandle) { return Invoke<Ped>(0x9B62392B474F44A0, v, returnHandle); }
inline void SET_BLOCKING_OF_NON_TEMPORARY_EVENTS(Ped p, BOOL on) { Invoke(0x9F8AA94D6D97DBF4, p, on); }
inline void TASK_VEHICLE_DRIVE_WANDER(Ped p, Vehicle v, float speed, int style) { Invoke(0x480142959D337D00, p, v, speed, style); }
inline void TASK_VEHICLE_DRIVE_TO_COORD(Ped p, Vehicle v, float x, float y, float z, float speed, int p6, Hash model, int style, float stopRange, float straightLine) {
  Invoke(0xE2A2AA2F659D77A7, p, v, x, y, z, speed, p6, model, style, stopRange, straightLine);
}
inline void TASK_VEHICLE_DRIVE_TO_COORD_LONGRANGE(Ped p, Vehicle v, float x, float y, float z, float speed, int style, float stopRange) {
  Invoke(0x158BB33F920D360C, p, v, x, y, z, speed, style, stopRange);
}
inline void SET_DRIVE_TASK_DRIVING_STYLE(Ped p, int style) { Invoke(0xDACE1BE37D88AF67, p, style); }
inline void SET_DRIVE_TASK_MAX_CRUISE_SPEED(Ped p, float speed, BOOL updateBaseTask) { Invoke(0x404A5AA9B9F0B746, p, speed, updateBaseTask); }
inline void SET_DRIVER_ABILITY(Ped p, float ability) { Invoke(0xB195FFA8042FC5C3, p, ability); }
inline void SET_DRIVER_AGGRESSIVENESS(Ped p, float aggressiveness) { Invoke(0xA731F608CA104E3C, p, aggressiveness); }
inline void SET_PED_KEEP_TASK(Ped p, BOOL on) { Invoke(0x971D38760FBC02EF, p, on); }
inline void CLEAR_PED_TASKS(Ped p) { Invoke(0xE1EF3C1216AFF2CD, p); }
// taskHash: GET_HASH_KEY("SCRIPT_TASK_<name>"); 0 waiting to start, 1 running, 7 finished or never given
inline int GET_SCRIPT_TASK_STATUS(Ped p, Hash taskHash) { return Invoke<int>(0x77F1BEB8863288D5, p, taskHash); }
inline Ped GET_PED_IN_VEHICLE_SEAT(Vehicle v, int seat, BOOL p2) { return Invoke<Ped>(0xBB40DD2270B65366, v, seat, p2); }
// flags: 2 vehicles; result 2 when ready
inline int START_EXPENSIVE_SYNCHRONOUS_SHAPE_TEST_LOS_PROBE(float x1, float y1, float z1, float x2, float y2, float z2, int flags, Entity ignore, int p8) {
  return Invoke<int>(0x377906D8A31E5586, x1, y1, z1, x2, y2, z2, flags, ignore, p8);
}
inline int GET_SHAPE_TEST_RESULT(int handle, BOOL *hit, Vector3 *end, Vector3 *normal, Entity *entity) {
  return Invoke<int>(0x3D87450E15D98694, handle, hit, end, normal, entity);
}
inline void SET_VEHICLE_MAX_SPEED(Vehicle v, float speed) { Invoke(0xBAA045B4E42F3C06, v, speed); }
inline float GET_ENTITY_SPEED(Entity e) { return Invoke<float>(0xD5037BA82E12416F, e); }
// arrays of 8-byte slots: the capacity first, then a handle every other slot from index 2
inline int GET_PED_NEARBY_VEHICLES(Ped p, int *slots) { return Invoke<int>(0xCFF869CBFA210D82, p, slots); }
inline int GET_PED_NEARBY_PEDS(Ped p, int *slots, int ignoreType) { return Invoke<int>(0x23F8F5FC7E8C4A6B, p, slots, ignoreType); }
inline BOOL IS_PED_DEAD_OR_DYING(Ped p, BOOL melee) { return Invoke<BOOL>(0x3317DEDB88C95038, p, melee); }
// for AI drivers: waiting at a red light, or queued behind one
inline BOOL IS_VEHICLE_STOPPED_AT_TRAFFIC_LIGHTS(Vehicle v) { return Invoke<BOOL>(0x2959F696AE390A99, v); }
inline void SET_DRIVE_TASK_CRUISE_SPEED(Ped p, float speed) { Invoke(0x5C9B84BD7D31D908, p, speed); }
inline void SET_ENTITY_COORDS(Entity e, float x, float y, float z, BOOL xa, BOOL ya, BOOL za, BOOL clear) { Invoke(0x06843DA7060A026B, e, x, y, z, xa, ya, za, clear); }
inline void SET_ENTITY_HEADING(Entity e, float h) { Invoke(0x8E2530AA8ADA980E, e, h); }
inline void SET_ENTITY_AS_MISSION_ENTITY(Entity e, BOOL a, BOOL b) { Invoke(0xAD738C3085FE7E11, e, a, b); }
inline void FREEZE_ENTITY_POSITION(Entity e, BOOL t) { Invoke(0x428CA6DBD1094446, e, t); }

// vehicles
inline Hash GET_HASH_KEY(const char *s) { return Invoke<Hash>(0xD24D37CC275948CC, s); }
// street hashes are written to 8-byte slots
inline void GET_STREET_NAME_AT_COORD(float x, float y, float z, uint64_t *street, uint64_t *crossing) { Invoke(0x2EB41072B4C1E4C0, x, y, z, street, crossing); }
inline const char *GET_STREET_NAME_FROM_HASH_KEY(Hash h) { return Invoke<const char *>(0xD0EF8A959B8A4CB9, h); }
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
inline void SET_ENTITY_INVINCIBLE(Entity e, BOOL on) { Invoke(0x3882114BDE571AD4, e, on); }
inline void SET_VEHICLE_CAN_BE_VISIBLY_DAMAGED(Vehicle v, BOOL on) { Invoke(0x4C7028F78FFD3681, v, on); }
inline void SET_VEHICLE_CAN_BREAK(Vehicle v, BOOL on) { Invoke(0xC5F0A8EBD3F361CE, v, on); }
inline void SET_VEHICLE_TYRES_CAN_BURST(Vehicle v, BOOL on) { Invoke(0xEB9DC3C7D8596C46, v, on); }
inline void SET_VEHICLE_WHEELS_CAN_BREAK(Vehicle v, BOOL on) { Invoke(0x29B18B4FD460CA8F, v, on); }
inline void SET_VEHICLE_ENGINE_CAN_DEGRADE(Vehicle v, BOOL on) { Invoke(0x983765856F2564F9, v, on); }
inline BOOL HAS_ENTITY_COLLIDED_WITH_ANYTHING(Entity e) { return Invoke<BOOL>(0x8BAD02F0368D9E14, e); }
inline float GET_VEHICLE_BODY_HEALTH(Vehicle v) { return Invoke<float>(0xF271147EB7B40F12, v); }
inline void SET_VEHICLE_MOD_KIT(Vehicle v, int kit) { Invoke(0x1F2AA07F00B3217A, v, kit); }
// paintType: 0 normal, 1 metallic, 2 pearl, 3 matte, 4 metal, 5 chrome
inline void SET_VEHICLE_MOD_COLOR_1(Vehicle v, int paintType, int color, int pearl) { Invoke(0x43FEB945EE7F85B8, v, paintType, color, pearl); }
inline void SET_VEHICLE_MOD_COLOR_2(Vehicle v, int paintType, int color) { Invoke(0x816562BADFDEC83E, v, paintType, color); }
inline void SET_VEHICLE_CUSTOM_PRIMARY_COLOUR(Vehicle v, int r, int g, int b) { Invoke(0x7141766F91D15BEA, v, r, g, b); }
// interior trim and dashboard, as palette indices (0 metallic black, 12 matte black, 111 white)
inline void SET_VEHICLE_EXTRA_COLOUR_5(Vehicle v, int color) { Invoke(0xF40DD601A65F7F19, v, color); }
inline void SET_VEHICLE_EXTRA_COLOUR_6(Vehicle v, int color) { Invoke(0x6089CDF6A57F326C, v, color); }
inline void SET_VEHICLE_CUSTOM_SECONDARY_COLOUR(Vehicle v, int r, int g, int b) { Invoke(0x36CED73BFED89754, v, r, g, b); }
inline void SET_VEHICLE_FORWARD_SPEED(Vehicle v, float s) { Invoke(0xAB54A438726D25D5, v, s); }
// palette indices (DurtyFree's vehicleColors.json): primary and secondary, then pearlescent and wheel
inline void SET_VEHICLE_COLOURS(Vehicle v, int primary, int secondary) { Invoke(0x4F1D4BE3A7F24601, v, primary, secondary); }
inline void GET_VEHICLE_COLOURS(Vehicle v, uint64_t *primary, uint64_t *secondary) { Invoke(0xA19435F193E081AC, v, primary, secondary); }
inline void SET_VEHICLE_EXTRA_COLOURS(Vehicle v, int pearl, int wheel) { Invoke(0x2036F561ADD12E33, v, pearl, wheel); }
inline void GET_VEHICLE_EXTRA_COLOURS(Vehicle v, uint64_t *pearl, uint64_t *wheel) { Invoke(0x3BC4245933A166F7, v, pearl, wheel); }
inline void CLEAR_VEHICLE_CUSTOM_PRIMARY_COLOUR(Vehicle v) { Invoke(0x55E1D2758F34E437, v); }
inline void CLEAR_VEHICLE_CUSTOM_SECONDARY_COLOUR(Vehicle v) { Invoke(0x5FFBDEEC3E8E2009, v); }
inline void SET_VEHICLE_DIRT_LEVEL(Vehicle v, float dirt) { Invoke(0x79D3B596FE44EE8B, v, dirt); }  // 0-15
inline float GET_VEHICLE_DIRT_LEVEL(Vehicle v) { return Invoke<float>(0x8F17BC8BA08DA62B, v); }
inline BOOL IS_MODEL_IN_CDIMAGE(Hash m) { return Invoke<BOOL>(0x35B9E0803292B641, m); }
inline BOOL IS_MODEL_A_VEHICLE(Hash m) { return Invoke<BOOL>(0x19AAC8F07BFEC53E, m); }
inline const char *GET_DISPLAY_NAME_FROM_VEHICLE_MODEL(Hash m) { return Invoke<const char *>(0xB215AAC32D25D019, m); }
inline void SET_VEHICLE_HAS_BEEN_OWNED_BY_PLAYER(Vehicle v, BOOL owned) { Invoke(0x2B5F9D2AF1F1722D, v, owned); }

// world and paths
inline BOOL LOAD_ALL_PATH_NODES(BOOL all) { return Invoke<BOOL>(0xC2AB6BFE34E92F8B, all); }
inline void REQUEST_COLLISION_AT_COORD(float x, float y, float z) { Invoke(0x07503F7948F491A7, x, y, z); }
inline BOOL GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(float x, float y, float z, Vector3 *pos, float *heading, int nodeType, float p6, float p7) {
  return Invoke<BOOL>(0xFF071FB798B803B0, x, y, z, pos, heading, nodeType, p6, p7);
}
inline BOOL GET_NTH_CLOSEST_VEHICLE_NODE_WITH_HEADING(float x, float y, float z, int n, Vector3 *pos, float *heading, int *lanes, int flags, float p8, float p9) {
  return Invoke<BOOL>(0x80CA6A8B6C094CC4, x, y, z, n, pos, heading, lanes, flags, p8, p9);
}
// the road nearest a point, between path nodes a and b, with its lanes towards each and the median's width
inline BOOL GET_CLOSEST_ROAD(float x, float y, float z, float p3, int p4, Vector3 *a, Vector3 *b, int *lanesToA, int *lanesToB, float *median, BOOL major) {
  return Invoke<BOOL>(0x132F52BBA570FE92, x, y, z, p3, p4, a, b, lanesToA, lanesToB, median, major);
}
inline BOOL GET_GROUND_Z_FOR_3D_COORD(float x, float y, float z, float *groundZ, BOOL ignoreWater, BOOL p5) {
  return Invoke<BOOL>(0xC906A7DAB05C8D2B, x, y, z, groundZ, ignoreWater, p5);
}
inline void SET_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(float m) { Invoke(0x245A6883D966D537, m); }
inline void SET_RANDOM_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(float m) { Invoke(0xB3B3359379FE77D3, m); }
inline void SET_PARKED_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(float m) { Invoke(0xEAE6DCC7EEE3DB1D, m); }
inline void SET_PED_DENSITY_MULTIPLIER_THIS_FRAME(float m) { Invoke(0x95E3D6257B166CF2, m); }
inline void SET_SCENARIO_PED_DENSITY_MULTIPLIER_THIS_FRAME(float a, float b) { Invoke(0x7A556143A1C03898, a, b); }
inline void SET_CLOCK_TIME(int h, int m, int s) { Invoke(0x47C3B5848C3E45D8, h, m, s); }
inline void SET_WEATHER_TYPE_NOW_PERSIST(const char *w) { Invoke(0xED712CA327900C8A, w); }
inline void SET_OVERRIDE_WEATHER(const char *w) { Invoke(0xA43D5C6FE51ADBEF, w); }
inline void PAUSE_CLOCK(BOOL toggle) { Invoke(0x4055E40BD2DBEC1D, toggle); }
inline int GET_CLOCK_HOURS() { return Invoke<int>(0x25223CA6B4D20B7F); }
inline int GET_CLOCK_MINUTES() { return Invoke<int>(0x13D2B8ADD79640F2); }
inline void SET_WEATHER_TYPE_OVERTIME_PERSIST(const char *w, float secs) { Invoke(0xFB5045B7C42B75BF, w, secs); }
inline void CLEAR_OVERRIDE_WEATHER() { Invoke(0x338D2E3477711050); }
inline void CLEAR_WEATHER_TYPE_PERSIST() { Invoke(0xCCC39339BEF76CF5); }
// the weather blending from type 1 to type 2; hashes and the float are written to 8-byte slots
inline void GET_CURR_WEATHER_STATE(uint64_t *from, uint64_t *to, uint64_t *mix) { Invoke(0xF3BBE884A14BB413, from, to, mix); }
// rain and puddles, 0-1 (above 0.5 only puddles form faster); -1 goes back to the weather's own
inline void SET_RAIN(float level) { Invoke(0x643E26EA6E024D92, level); }
inline float GET_RAIN_LEVEL() { return Invoke<float>(0x96695E368AD855F3); }
inline void SET_MAX_WANTED_LEVEL(int lvl) { Invoke(0xAA5F02DB48D704B9, lvl); }
inline void CLEAR_PLAYER_WANTED_LEVEL(int player) { Invoke(0xB302540597885499, player); }
inline BOOL IS_PAUSE_MENU_ACTIVE() { return Invoke<BOOL>(0xB0034A223497FFCB); }
inline float GET_FRAME_TIME() { return Invoke<float>(0x15C40837039FFAF7); }

// GPS
inline BOOL IS_WAYPOINT_ACTIVE() { return Invoke<BOOL>(0x1DD1F58F493F1DA5); }
inline int GET_WAYPOINT_BLIP_ENUM_ID() { return Invoke<int>(0x186E5D252FA50E7D); }
inline int GET_FIRST_BLIP_INFO_ID(int sprite) { return Invoke<int>(0x1BEDE233E6CD2A1F, sprite); }
inline Vector3 GET_BLIP_INFO_ID_COORD(int blip) { return Invoke<Vector3>(0xFA7C7F0AADF25D09, blip); }
inline void SET_WAYPOINT_OFF() { Invoke(0xA7E4E2D361C2627F); }
inline void SET_NEW_WAYPOINT(float x, float y) { Invoke(0xFE43368D2AA4F2FC, x, y); }
inline BOOL GET_POS_ALONG_GPS_TYPE_ROUTE(Vector3 *result, BOOL p1, float dist, int type) { return Invoke<BOOL>(0xF3162836C28F9DA5, result, p1, dist, type); }
// blips: sprite 8 is the waypoint's; REMOVE_BLIP zeroes the handle
inline int ADD_BLIP_FOR_COORD(float x, float y, float z) { return Invoke<int>(0x5A039BB0BCA604B6, x, y, z); }
inline BOOL DOES_BLIP_EXIST(int blip) { return Invoke<BOOL>(0xA6DB27D19ECBB7DA, blip); }
inline void REMOVE_BLIP(int *blip) { Invoke(0x86A652570E5F25DD, blip); }
inline void SET_BLIP_SPRITE(int blip, int sprite) { Invoke(0xDF735600A4696DAF, blip, sprite); }
inline void SET_BLIP_COLOUR(int blip, int colour) { Invoke(0x03D7FB09E75D6B7E, blip, colour); }
// a script's own line on the minimap and map, straight between its points
inline void CLEAR_GPS_CUSTOM_ROUTE() { Invoke(0xE6DE0561D9232A64); }
inline void START_GPS_CUSTOM_ROUTE(int hudColour, BOOL displayOnFoot, BOOL followPlayer) { Invoke(0xDB34E8D56FC13B08, hudColour, displayOnFoot, followPlayer); }
inline void ADD_POINT_TO_GPS_CUSTOM_ROUTE(float x, float y, float z) { Invoke(0x311438A071DD9B1A, x, y, z); }
inline void SET_GPS_CUSTOM_ROUTE_RENDER(BOOL on, int radarThickness, int mapThickness) { Invoke(0x900086F371220B6F, on, radarThickness, mapThickness); }
// GTA's own GPS directions from the player to a point: direction (3 left, 4 right, 5 straight, 6/7 sharp left/right; 1
// still working it out) and the distance to the next junction in decimetres, written to 8-byte slots
inline int GENERATE_DIRECTIONS_TO_COORD(float x, float y, float z, BOOL p3, uint64_t *direction, uint64_t *p5, uint64_t *dist) {
  return Invoke<int>(0xF90125F1F79ECDF8, x, y, z, p3, direction, p5, dist);
}

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
inline void SET_CAM_COORD(Cam c, float x, float y, float z) { Invoke(0x4D41783FB745E42E, c, x, y, z); }
inline void SET_CAM_ROT(Cam c, float x, float y, float z, int order) { Invoke(0x85973643155D0B07, c, x, y, z, order); }
inline void HIDE_HUD_AND_RADAR_THIS_FRAME() { Invoke(0x719FF505F097FD20); }
inline void SET_FOCUS_POS_AND_VEL(float x, float y, float z, float vx, float vy, float vz) { Invoke(0xBB7454BAFF08FE25, x, y, z, vx, vy, vz); }
inline void CLEAR_FOCUS() { Invoke(0x31B73D1EA9F01DA2); }
inline void RENDER_SCRIPT_CAMS(BOOL render, BOOL ease, int easeTime, BOOL p3, BOOL p4, int p5) { Invoke(0x07E5B515DB0636FC, render, ease, easeTime, p3, p4, p5); }
inline void FREEZE_MICROPHONE() { Invoke(0xD57AAAE0E2214D11); }
inline void DISPLAY_RADAR(BOOL on) { Invoke(0xA0EBB943C300E693, on); }
inline BOOL IS_RADAR_HIDDEN() { return Invoke<BOOL>(0x157F93B036700462); }
inline void HIDE_HELP_TEXT_THIS_FRAME() { Invoke(0xD46923FC481CA285); }
inline void THEFEED_HIDE_THIS_FRAME() { Invoke(0x25F87B30C382FCA7); }
inline void CLEAR_PRINTS() { Invoke(0xCC33FA791322B9D9); }
// screen coordinates 0-1, the rectangle's center and size
inline void DRAW_RECT(float x, float y, float w, float h, int r, int g, int b, int a, BOOL p8) { Invoke(0x3A618A217E5154F0, x, y, w, h, r, g, b, a, p8); }
// in the world, for one frame: a depth-tested line, a one-sided triangle (seen from where its corners go counterclockwise)
inline void DRAW_LINE(float x1, float y1, float z1, float x2, float y2, float z2, int r, int g, int b, int a) {
  Invoke(0x6B7256074AE34680, x1, y1, z1, x2, y2, z2, r, g, b, a);
}
inline void DRAW_POLY(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, int r, int g, int b, int a) {
  Invoke(0xAC26716048436851, x1, y1, z1, x2, y2, z2, x3, y3, z3, r, g, b, a);
}
inline void DRAW_MARKER(int type, float x, float y, float z, float dirX, float dirY, float dirZ, float rotX, float rotY, float rotZ, float scaleX,
                        float scaleY, float scaleZ, int r, int g, int b, int a, BOOL bob, BOOL faceCamera, int rotOrder, BOOL rotate,
                        const char *dict, const char *name, BOOL invert) {
  Invoke(0x28477EC23D892089, type, x, y, z, dirX, dirY, dirZ, rotX, rotY, rotZ, scaleX, scaleY, scaleZ, r, g, b, a, bob, faceCamera, rotOrder,
         rotate, dict, name, invert);
}
inline void SET_TEXT_FONT(int font) { Invoke(0x66E0276CC5F6B9DA, font); }
inline void SET_TEXT_SCALE(float scale, float size) { Invoke(0x07C837F9A01C34C9, scale, size); }
inline void SET_TEXT_COLOUR(int r, int g, int b, int a) { Invoke(0xBE6B23FFA53FB442, r, g, b, a); }
inline void SET_TEXT_OUTLINE() { Invoke(0x2513DFB0FB8400FE); }
inline void SET_TEXT_JUSTIFICATION(int justify) { Invoke(0x4E096588B13FFECA, justify); }  // 0 centre, 1 left, 2 right
inline void SET_TEXT_WRAP(float start, float end) { Invoke(0x63145D9C883A1A70, start, end); }
inline void BEGIN_TEXT_COMMAND_DISPLAY_TEXT(const char *fmt) { Invoke(0x25FBB336DF1804CB, fmt); }
inline void ADD_TEXT_COMPONENT_SUBSTRING_PLAYER_NAME(const char *s) { Invoke(0x6C188BE134E074AA, s); }
inline void END_TEXT_COMMAND_DISPLAY_TEXT(float x, float y, int p2) { Invoke(0xCD015E5BB0D96A57, x, y, p2); }

// controls
inline BOOL SET_CONTROL_VALUE_NEXT_FRAME(int control, int action, float v) { return Invoke<BOOL>(0xE8A25867FBA3B05E, control, action, v); }
inline void DISABLE_CONTROL_ACTION(int control, int action, BOOL related) { Invoke(0xFE99B66D079CF6BC, control, action, related); }
inline float GET_DISABLED_CONTROL_NORMAL(int control, int action) { return Invoke<float>(0x11E65974A982637C, control, action); }
inline float GET_CONTROL_NORMAL(int control, int action) { return Invoke<float>(0xEC3C9B8D5327B563, control, action); }

enum Input {
  INPUT_PHONE = 27,
  INPUT_VEH_MOVE_LR = 59,
  INPUT_VEH_ACCELERATE = 71,
  INPUT_VEH_BRAKE = 72,
  INPUT_VEH_EXIT = 75,
  INPUT_VEH_HANDBRAKE = 76,
};

}  // namespace nat
