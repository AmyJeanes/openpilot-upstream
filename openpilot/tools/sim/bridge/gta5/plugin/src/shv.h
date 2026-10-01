// ScriptHookV's exports, linked through an import library built from ScriptHookV.def, so building needs no SDK.
#pragma once
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace shv {

using KeyboardHandler = void (*)(DWORD key, WORD repeats, BYTE scanCode, BOOL isExtended, BOOL isWithAlt, BOOL wasDownBefore, BOOL isUpNow);

struct Api {
  void (*scriptWait)(DWORD);
  void (*scriptRegister)(HMODULE, void (*)());
  void (*scriptUnregister)(HMODULE);
  void (*keyboardHandlerRegister)(KeyboardHandler);
  void (*keyboardHandlerUnregister)(KeyboardHandler);
  void (*nativeInit)(uint64_t);
  void (*nativePush64)(uint64_t);
  uint64_t *(*nativeCall)();
};

}  // namespace shv

// ScriptHookV's own declarations, so the names mangle to its exports. Linking them, rather than looking them up, makes the
// ASI loader load ScriptHookV.dll before this plugin, whatever order it finds the plugins in.
__declspec(dllimport) void scriptWait(DWORD time);
__declspec(dllimport) void scriptRegister(HMODULE module, void (*main)());
__declspec(dllimport) void scriptUnregister(HMODULE module);
__declspec(dllimport) void keyboardHandlerRegister(shv::KeyboardHandler handler);
__declspec(dllimport) void keyboardHandlerUnregister(shv::KeyboardHandler handler);
__declspec(dllimport) void nativeInit(UINT64 hash);
__declspec(dllimport) void nativePush64(UINT64 val);
__declspec(dllimport) PUINT64 nativeCall();

namespace shv {

inline Api Imports() {
  return Api{::scriptWait, ::scriptRegister, ::scriptUnregister, ::keyboardHandlerRegister, ::keyboardHandlerUnregister,
             ::nativeInit, ::nativePush64, ::nativeCall};
}

// script natives pass every argument and return value in 8-byte slots
#pragma pack(push, 1)
struct Vector3 {
  float x; uint32_t _px;
  float y; uint32_t _py;
  float z; uint32_t _pz;
};
#pragma pack(pop)

extern Api api;

template <typename T>
inline uint64_t Pack(T v) {
  static_assert(sizeof(T) <= 8);
  uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(T));
  return out;
}

template <typename R = void, typename... Args>
inline R Invoke(uint64_t hash, Args... args) {
  api.nativeInit(hash);
  (api.nativePush64(Pack(args)), ...);
  uint64_t *ret = api.nativeCall();
  if constexpr (!std::is_void_v<R>) return *reinterpret_cast<R *>(ret);
}

}  // namespace shv
