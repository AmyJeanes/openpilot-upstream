// Interface between the loader (gta5op.asi) and the reloadable core (gta5op_core.dll).
#pragma once
#include "shv.h"

constexpr int CORE_API_VERSION = 1;

struct CoreHost {
  int version;
  const shv::Api *api;
  const wchar_t *dir;  // <game>\gta5op: config, log
};

using CoreInitFn = void (*)(const CoreHost *);
using CoreTickFn = void (*)();      // once per game frame, in the script fiber
using CoreShutdownFn = void (*)();  // in the script fiber, before the core is unloaded
using CoreKeyFn = void (*)(DWORD);  // key down, from the game's window thread
