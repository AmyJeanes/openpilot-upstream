#include "script_hook.h"

#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <vector>

#include "natives.h"

// Layouts and patterns for GTA V Enhanced, as YimMenuV2 uses them (src/game/pointers/Pointers.cpp,
// src/types/script/scrProgram.hpp). Game scripts call natives through their program's table of handlers, so replacing
// one entry in pi_menu's table changes the native for that script alone.

using namespace nat;

namespace {

struct NativeContext {
  void *ret;
  uint32_t argCount;
  void *args;
};
using Handler = void (*)(NativeContext *);

struct Program {  // rage::scrProgram
  uint8_t pad0[0x10];
  uint8_t **codeBlocks;
  uint32_t hash;
  uint32_t codeSize;
  uint8_t pad1[0x2C - 0x20];
  uint32_t nativeCount;
  uint8_t pad2[0x40 - 0x30];
  Handler *natives;
  uint8_t pad3[0x58 - 0x48];
  uint32_t nameHash;
  uint8_t pad4[0x80 - 0x5C];
};
static_assert(offsetof(Program, natives) == 0x40 && offsetof(Program, nameHash) == 0x58);

constexpr int PROGRAM_SLOTS = 176;
constexpr uint64_t GAMEPLAY_CAM_RENDERING = 0x174DBD3C5DB3557B;  // IS_GAMEPLAY_CAM_RENDERING's hash in Enhanced

std::function<void(const std::string &)> g_log;
Program **g_programs = nullptr;
Handler g_original = nullptr;
Handler *g_slot = nullptr;  // the patched entry in pi_menu's table
Program *g_program = nullptr;
Program *g_checked = nullptr;  // a pi_menu program without the handler, not searched again
uint32_t g_piMenu = 0;
bool g_override = false;

void GameplayCamRendering(NativeContext *c) {
  g_original(c);
  if (g_override) *static_cast<int *>(c->ret) = TRUE;
}

// the first match of a pattern ("48 8B ? ?") in the game's readable memory
uint8_t *Find(const char *pattern) {
  std::vector<int> bytes;
  for (const char *p = pattern; *p;) {
    if (*p == ' ') { p++; continue; }
    if (*p == '?') { bytes.push_back(-1); p++; continue; }
    bytes.push_back(static_cast<int>(std::strtoul(p, const_cast<char **>(&p), 16)));
  }
  auto base = reinterpret_cast<uint8_t *>(GetModuleHandleW(nullptr));
  auto nt = reinterpret_cast<IMAGE_NT_HEADERS *>(base + reinterpret_cast<IMAGE_DOS_HEADER *>(base)->e_lfanew);
  uint8_t *end = base + nt->OptionalHeader.SizeOfImage;
  for (uint8_t *region = base; region < end;) {
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery(region, &mbi, sizeof(mbi))) break;
    uint8_t *next = static_cast<uint8_t *>(mbi.BaseAddress) + mbi.RegionSize;
    bool readable = mbi.State == MEM_COMMIT && !(mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS));
    if (readable) {
      size_t n = static_cast<size_t>(std::min(next, end) - region);
      for (size_t i = 0; i + bytes.size() <= n; i++) {
        size_t j = 0;
        while (j < bytes.size() && (bytes[j] < 0 || region[i + j] == bytes[j])) j++;
        if (j == bytes.size()) return region + i;
      }
    }
    region = next;
  }
  return nullptr;
}

uint8_t *Rip(uint8_t *p) { return p + 4 + *reinterpret_cast<int32_t *>(p); }

void Restore() {
  if (g_slot && *g_slot == GameplayCamRendering) *g_slot = g_original;
  g_slot = nullptr;
  g_program = nullptr;
}

}  // namespace

namespace script_hook {

bool Install(std::function<void(const std::string &)> log) {
  g_log = std::move(log);
  uint8_t *init = Find("EB 2A 0F 1F 40 00 48 8B 54 17 10");
  uint8_t *programs = Find("48 C7 84 C8 D8 00 00 00 00 00 00 00");
  if (!init || !programs) {
    g_log("script hook: game code not found; the interaction menu closes while connected");
    return false;
  }
  g_programs = reinterpret_cast<Program **>(Rip(programs + 0x16) + 0xD8);
  // the game's InitNativeTables turns a program's native hashes into handlers in place; a program with only this
  // native gives its handler
  uint64_t entry = GAMEPLAY_CAM_RENDERING;
  Program fake{};
  fake.nativeCount = 1;
  fake.natives = reinterpret_cast<Handler *>(&entry);
  reinterpret_cast<void (*)(Program *)>(init - 0x2A)(&fake);
  g_original = reinterpret_cast<Handler>(entry);
  if (entry == GAMEPLAY_CAM_RENDERING || !entry) {
    g_log("script hook: no handler for IS_GAMEPLAY_CAM_RENDERING");
    g_original = nullptr;
    return false;
  }
  g_piMenu = GET_HASH_KEY("pi_menu");
  g_log("script hook: ready");
  return true;
}

void Update() {
  if (!g_original) return;
  if (g_program) {
    // still loaded with our entry: nothing to do
    bool loaded = false;
    for (int i = 0; i < PROGRAM_SLOTS; i++) loaded |= g_programs[i] == g_program;
    if (loaded && g_program->nameHash == g_piMenu && *g_slot == GameplayCamRendering) return;
    g_slot = nullptr;
    g_program = nullptr;
  }
  for (int i = 0; i < PROGRAM_SLOTS; i++) {
    Program *p = g_programs[i];
    if (!p || p == g_checked || !p->codeBlocks || !p->codeSize || p->nameHash != g_piMenu || !p->natives) continue;
    g_checked = p;
    for (uint32_t n = 0; n < p->nativeCount; n++) {
      if (p->natives[n] != g_original) continue;
      p->natives[n] = GameplayCamRendering;
      g_slot = &p->natives[n];
      g_program = p;
      g_checked = nullptr;
      g_log("script hook: pi_menu patched");
      return;
    }
  }
}

void Uninstall() {
  Restore();
  g_original = nullptr;
}

void SetOverride(bool on) { g_override = on; }

}  // namespace script_hook
