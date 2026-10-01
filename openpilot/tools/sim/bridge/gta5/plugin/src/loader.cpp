// gta5op.asi: a thin script that runs gta5op\gta5op_core.dll and reloads it whenever the file changes, so a rebuilt core
// takes effect without restarting the game. The core runs inside this script's fiber, where natives may be called.
#include <windows.h>
#include <share.h>
#include <filesystem>
#include <string>

#include "core_api.h"
#include "shv.h"

namespace fs = std::filesystem;

shv::Api shv::api;

namespace {

HMODULE g_self = nullptr;
fs::path g_dir;  // <game>\gta5op

struct Core {
  HMODULE module = nullptr;
  fs::path loadedPath;
  fs::file_time_type stamp{};
  CoreInitFn init = nullptr;
  CoreTickFn tick = nullptr;
  CoreShutdownFn shutdown = nullptr;
  CoreKeyFn key = nullptr;
} g_core;

int g_loadCount = 0;
SRWLOCK g_keyLock = SRWLOCK_INIT;  // OnKey runs on the window thread while the script fiber may swap the core

void Log(const std::string &msg) {
  OutputDebugStringA(("gta5op: " + msg + "\n").c_str());
  // shared, so the log can be followed while the game runs
  FILE *f = _wfsopen((g_dir / L"gta5op.log").c_str(), L"a", _SH_DENYNO);
  if (f) {
    SYSTEMTIME t;
    GetLocalTime(&t);
    fprintf(f, "[%02d:%02d:%02d.%03d] loader: %s\n", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds, msg.c_str());
    fclose(f);
  }
}

void UnloadCore() {
  if (!g_core.module) return;
  AcquireSRWLockExclusive(&g_keyLock);
  g_core.key = nullptr;
  ReleaseSRWLockExclusive(&g_keyLock);
  if (g_core.shutdown) g_core.shutdown();
  FreeLibrary(g_core.module);
  std::error_code ec;
  fs::remove(g_core.loadedPath, ec);
  g_core = Core{};
}

void LoadCore(const fs::path &src, fs::file_time_type stamp) {
  UnloadCore();
  // load a copy, so the build can overwrite the original while it is in use
  fs::path copy = g_dir / (L"gta5op_core.loaded" + std::to_wstring(++g_loadCount) + L".dll");
  std::error_code ec;
  fs::copy_file(src, copy, fs::copy_options::overwrite_existing, ec);
  if (ec) {
    Log("copying the core failed: " + ec.message());
    return;
  }
  HMODULE m = LoadLibraryW(copy.c_str());
  if (!m) {
    Log("loading the core failed: error " + std::to_string(GetLastError()));
    fs::remove(copy, ec);
    return;
  }
  g_core.module = m;
  g_core.loadedPath = copy;
  g_core.stamp = stamp;
  g_core.init = reinterpret_cast<CoreInitFn>(GetProcAddress(m, "CoreInit"));
  g_core.tick = reinterpret_cast<CoreTickFn>(GetProcAddress(m, "CoreTick"));
  g_core.shutdown = reinterpret_cast<CoreShutdownFn>(GetProcAddress(m, "CoreShutdown"));
  if (!g_core.init || !g_core.tick || !g_core.shutdown) {
    Log("the core lacks its exports");
    FreeLibrary(m);
    fs::remove(copy, ec);
    g_core = Core{};
    return;
  }
  CoreHost host{CORE_API_VERSION, &shv::api, g_dir.c_str()};
  g_core.init(&host);
  AcquireSRWLockExclusive(&g_keyLock);
  g_core.key = reinterpret_cast<CoreKeyFn>(GetProcAddress(m, "CoreKey"));
  ReleaseSRWLockExclusive(&g_keyLock);
  Log("core loaded (" + std::to_string(g_loadCount) + ")");
}

void CheckCore() {
  static DWORD lastCheck = 0;
  static fs::file_time_type pending{};
  static DWORD pendingSince = 0;
  DWORD now = GetTickCount();
  if (g_core.module && now - lastCheck < 500) return;
  lastCheck = now;
  fs::path src = g_dir / L"gta5op_core.dll";
  std::error_code ec;
  auto stamp = fs::last_write_time(src, ec);
  if (ec || stamp == g_core.stamp) return;
  // wait for the build to finish writing it
  if (stamp != pending) {
    pending = stamp;
    pendingSince = now;
    return;
  }
  if (now - pendingSince < 1000) return;
  LoadCore(src, stamp);
  if (!g_core.module) g_core.stamp = stamp;  // don't retry a broken file until it changes again
}

void ScriptMain() {
  // remove copies left by a crash
  std::error_code ec;
  for (auto &e : fs::directory_iterator(g_dir, ec)) {
    auto name = e.path().filename().wstring();
    if (name.rfind(L"gta5op_core.loaded", 0) == 0) fs::remove(e.path(), ec);
  }
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
    g_self = module;
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(module, path, MAX_PATH);
    g_dir = fs::path(path).parent_path() / L"gta5op";
    shv::api = shv::Imports();
    shv::api.scriptRegister(module, ScriptMain);
    shv::api.keyboardHandlerRegister(OnKey);
  } else if (reason == DLL_PROCESS_DETACH) {
    shv::api.keyboardHandlerUnregister(OnKey);
    shv::api.scriptUnregister(module);
  }
  return TRUE;
}
