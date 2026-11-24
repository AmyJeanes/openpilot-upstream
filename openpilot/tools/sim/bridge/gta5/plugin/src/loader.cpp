// gta5op.asi: a thin script that runs the newest core in gta5op\core (gta5op_core_<build time>.dll) and swaps to a newer
// one as soon as it appears there, so a rebuilt core takes effect without restarting the game. The core runs inside this
// script's fiber, where natives may be called. The loader itself only changes at a game start: keep it small.
#include <windows.h>
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "core_api.h"
#include "shv.h"

namespace fs = std::filesystem;

shv::Api shv::api;

namespace {

constexpr size_t KEEP_CORES = 3, KEEP_LOGS = 20;
constexpr int LOAD_TRIES = 10;  // of a core that will not load, 0.5 s apart: it may still be being copied in

fs::path g_dir;  // <game>\gta5op

struct Core {
  HMODULE module = nullptr;
  std::wstring name;
  CoreInitFn init = nullptr;
  CoreTickFn tick = nullptr;
  CoreShutdownFn shutdown = nullptr;
  CoreKeyFn key = nullptr;
} g_core;

std::vector<std::wstring> g_bad;  // cores given up on
int g_tries = 0;
SRWLOCK g_keyLock = SRWLOCK_INIT;  // OnKey runs on the window thread while the script fiber may swap the core

void Log(const std::string &msg) {
  OutputDebugStringA(("gta5op: " + msg + "\n").c_str());
  WriteLog(g_dir.wstring(), "loader", msg);
}

// The files of dir named <prefix>*<ext>, newest first: the names carry a time, so name order is age order.
std::vector<std::wstring> Newest(const fs::path &dir, const wchar_t *prefix, const wchar_t *ext) {
  std::vector<std::wstring> names;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
    std::wstring name = it->path().filename().wstring();
    if (name.rfind(prefix, 0) == 0 && it->path().extension() == ext) names.push_back(name);
  }
  std::sort(names.rbegin(), names.rend());
  return names;
}

// Removes all but the newest few. A file in use (the loaded core, or one the game holds) stays until a later try.
void Prune(const fs::path &dir, const wchar_t *prefix, const wchar_t *ext, size_t keep) {
  std::vector<std::wstring> names = Newest(dir, prefix, ext);
  std::error_code ec;
  for (size_t i = keep; i < names.size(); i++) fs::remove(dir / names[i], ec);
}

void UnloadCore() {
  if (!g_core.module) return;
  AcquireSRWLockExclusive(&g_keyLock);
  g_core.key = nullptr;
  ReleaseSRWLockExclusive(&g_keyLock);
  g_core.shutdown();
  FreeLibrary(g_core.module);
  g_core = Core{};
}

// Loads the new core before the running one is shut down, so a file that will not load leaves the old core running.
bool LoadCore(const std::wstring &name) {
  std::string shown = fs::path(name).string();
  DWORD mode = 0;
  SetThreadErrorMode(SEM_FAILCRITICALERRORS, &mode);
  HMODULE m = LoadLibraryW((g_dir / L"core" / name).c_str());
  DWORD error = GetLastError();
  SetThreadErrorMode(mode, nullptr);
  if (!m) {
    if (!g_tries) Log("loading " + shown + " failed: error " + std::to_string(error));
    return false;
  }
  Core next;
  next.module = m;
  next.name = name;
  next.init = reinterpret_cast<CoreInitFn>(GetProcAddress(m, "CoreInit"));
  next.tick = reinterpret_cast<CoreTickFn>(GetProcAddress(m, "CoreTick"));
  next.shutdown = reinterpret_cast<CoreShutdownFn>(GetProcAddress(m, "CoreShutdown"));
  if (!next.init || !next.tick || !next.shutdown) {
    if (!g_tries) Log(shown + " lacks the core's exports");
    FreeLibrary(m);
    return false;
  }
  UnloadCore();
  g_core = next;
  CoreHost host{CORE_API_VERSION, &shv::api, g_dir.c_str()};
  g_core.init(&host);
  AcquireSRWLockExclusive(&g_keyLock);
  g_core.key = reinterpret_cast<CoreKeyFn>(GetProcAddress(m, "CoreKey"));
  ReleaseSRWLockExclusive(&g_keyLock);
  Log("core loaded: " + shown);
  return true;
}

void CheckCore() {
  static DWORD lastCheck = 0;
  DWORD now = GetTickCount() | 1;
  if (lastCheck && now - lastCheck < 500) return;
  lastCheck = now;
  for (const std::wstring &name : Newest(g_dir / L"core", L"gta5op_core_", L".dll")) {
    if (name == g_core.name) return;  // the newest that loads is running
    if (std::find(g_bad.begin(), g_bad.end(), name) != g_bad.end()) continue;
    if (LoadCore(name)) {
      g_tries = 0;
      Prune(g_dir / L"core", L"gta5op_core_", L".dll", KEEP_CORES);
    } else if (++g_tries >= LOAD_TRIES) {
      g_tries = 0;
      g_bad.push_back(name);
      Log("giving up on " + fs::path(name).string());
    }
    return;
  }
  static bool said = false;
  if (!g_core.module && !said) said = (Log("no core to load in gta5op\\core"), true);
}

void ScriptMain() {
  for (;;) {
    CheckCore();
    if (g_core.tick) g_core.tick();
    shv::api.scriptWait(0);
  }
}

void OnKey(DWORD key, WORD, BYTE, BOOL, BOOL, BOOL wasDownBefore, BOOL isUpNow) {
  if (wasDownBefore || isUpNow) return;
  AcquireSRWLockShared(&g_keyLock);
  if (g_core.key) g_core.key(key);
  ReleaseSRWLockShared(&g_keyLock);
}

}  // namespace

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID) {
  if (reason == DLL_PROCESS_ATTACH) {
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(module, path, MAX_PATH);
    g_dir = fs::path(path).parent_path() / L"gta5op";
    // as early as this, because the game has been seen to take every file that is in its folder as it starts and hold
    // it against writing and deleting until it exits
    Prune(g_dir / L"core", L"gta5op_core_", L".dll", KEEP_CORES);
    Prune(g_dir / L"logs", L"gta5op_", L".log", KEEP_LOGS);
    shv::api = shv::Imports();
    shv::api.scriptRegister(module, ScriptMain);
    shv::api.keyboardHandlerRegister(OnKey);
  } else if (reason == DLL_PROCESS_DETACH) {
    shv::api.keyboardHandlerUnregister(OnKey);
    shv::api.scriptUnregister(module);
  }
  return TRUE;
}
