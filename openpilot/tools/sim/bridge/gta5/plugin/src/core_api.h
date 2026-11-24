// Interface between the loader (gta5op.asi) and the reloadable core (gta5op\core\gta5op_core_<build time>.dll).
#pragma once
#include <windows.h>
#include <share.h>
#include <cstdio>
#include <string>

#include "shv.h"

constexpr int CORE_API_VERSION = 1;

struct CoreHost {
  int version;
  const shv::Api *api;
  const wchar_t *dir;  // <game>\gta5op: config, cores, logs
};

using CoreInitFn = void (*)(const CoreHost *);
using CoreTickFn = void (*)();      // once per game frame, in the script fiber
using CoreShutdownFn = void (*)();  // in the script fiber, before the core is unloaded
using CoreKeyFn = void (*)(DWORD);  // key down, from the game's window thread

// The log's name for this run of the game: gta5op_<when the game started, yyyymmdd-hhmmss>.log, in <dir>\logs. The loader
// and every core loaded in the run work out the same name, so they share the file, and the newest run's sorts last.
inline std::wstring LogName(int n = 0) {
  FILETIME created{}, exited, kernel, user;
  GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user);
  SYSTEMTIME utc{}, t{};
  FileTimeToSystemTime(&created, &utc);
  SystemTimeToTzSpecificLocalTime(nullptr, &utc, &t);
  wchar_t name[64];
  swprintf(name, 64, L"gta5op_%04d%02d%02d-%02d%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
  return name + (n ? L"_" + std::to_wstring(n + 1) : L"") + L".log";
}

// Appends a line to this run's log. Opened for each line and shared, so the log can be followed while the game runs.
// A new file for each run because the game has been seen to hold every file that was in its folder when it started
// against writing; should even this run's file not open, the same name with _2, _3... is used.
inline void WriteLog(const std::wstring &dir, const char *who, const std::string &msg) {
  static int n = 0;
  std::wstring logs = dir + L"\\logs";
  FILE *f = nullptr;
  for (int tries = 0; tries < 2 && !f; tries++) {
    for (; n < 9 && !f; n++) f = _wfsopen((logs + L"\\" + LogName(n)).c_str(), L"a", _SH_DENYNO);
    if (f) n--;
    else n = 0, CreateDirectoryW(dir.c_str(), nullptr), CreateDirectoryW(logs.c_str(), nullptr);
  }
  if (!f) return;
  SYSTEMTIME t;
  GetLocalTime(&t);
  fprintf(f, "[%02d:%02d:%02d.%03d] %s: %s\n", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds, who, msg.c_str());
  fclose(f);
}
