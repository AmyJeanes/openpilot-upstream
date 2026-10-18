#include "ortho.h"

#include <windows.h>
#include <d3d12.h>
#include <tlhelp32.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "present_hook.h"

namespace ortho {

namespace {

constexpr double DEG = 3.14159265358979323846 / 180.0;
constexpr int MAX_HITS = 400;

using LogFn = std::function<void(const std::string &)>;

uintptr_t g_exeLo = 0, g_exeHi = 0, g_ownLo = 0, g_ownHi = 0;

void ModuleRange(HMODULE m, uintptr_t &lo, uintptr_t &hi) {
  auto base = reinterpret_cast<uint8_t *>(m);
  auto nt = reinterpret_cast<IMAGE_NT_HEADERS *>(base + reinterpret_cast<IMAGE_DOS_HEADER *>(base)->e_lfanew);
  lo = reinterpret_cast<uintptr_t>(base), hi = lo + nt->OptionalHeader.SizeOfImage;
}

void Ranges() {
  if (g_exeLo) return;
  ModuleRange(GetModuleHandleW(nullptr), g_exeLo, g_exeHi);
  HMODULE own = nullptr;
  GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&Ranges), &own);
  if (own) ModuleRange(own, g_ownLo, g_ownHi);
}

// memory freed under us or a guard page reads as a fault; a touched guard page is put back (a stack grows through it)
int ReadFault(EXCEPTION_POINTERS *e) {
  DWORD code = e->ExceptionRecord->ExceptionCode;
  if (code == STATUS_GUARD_PAGE_VIOLATION && e->ExceptionRecord->NumberParameters >= 2) {
    void *page = reinterpret_cast<void *>(e->ExceptionRecord->ExceptionInformation[1]);
    MEMORY_BASIC_INFORMATION mbi;
    DWORD old;
    if (VirtualQuery(page, &mbi, sizeof(mbi))) VirtualProtect(page, 1, mbi.Protect | PAGE_GUARD, &old);
  }
  return code == EXCEPTION_ACCESS_VIOLATION || code == STATUS_GUARD_PAGE_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH;
}

bool Copy(void *dst, const void *src, size_t n) {
  __try {
    std::memcpy(dst, src, n);
    return true;
  } __except (ReadFault(GetExceptionInformation())) {
    return false;
  }
}

std::string Hex(uintptr_t v) {
  char b[32];
  snprintf(b, sizeof(b), "%llx", static_cast<unsigned long long>(v));
  return b;
}

std::string Where(uintptr_t a) {
  if (a >= g_exeLo && a < g_exeHi) return "exe+" + Hex(a - g_exeLo);
  return Hex(a);
}

std::string Floats(const float *m, int n) {
  std::string s;
  char b[32];
  for (int i = 0; i < n; i++) {
    snprintf(b, sizeof(b), "%s%.6g", i ? "," : "", m[i]);
    s += b;
  }
  return s;
}

bool Zero(float v) { return std::fabs(v) < 1e-6f; }
bool One(float v) { return std::fabs(std::fabs(v) - 1.0f) < 1e-6f; }

// a perspective projection, row-major (m[11] = +-1) or transposed (m[14] = +-1), with y scale near ys
bool Projection(const float *m, float ys, float tol, bool &transposed) {
  if (!(m[0] > 0 && m[5] > 0 && std::fabs(m[5] - ys) <= tol * ys)) return false;
  float aspect = m[5] / m[0];
  if (aspect < 0.25f || aspect > 4.0f) return false;
  if (!(Zero(m[1]) && Zero(m[3]) && Zero(m[4]) && Zero(m[7]) && Zero(m[15]))) return false;
  if (One(m[11]) && Zero(m[12]) && Zero(m[13]) && Zero(m[2]) && Zero(m[6])) return transposed = false, true;
  if (One(m[14]) && Zero(m[2]) && Zero(m[6]) && Zero(m[12]) && Zero(m[13])) return transposed = true, true;
  return false;
}

std::atomic<bool> g_scanStop{false};
std::thread g_scan;

void ScanThread(float ys, float tol, float cx, float cy, float cz, LogFn log) {
  Ranges();
  double t0 = GetTickCount64() / 1000.0;
  struct Region {
    uintptr_t lo, hi;
  };
  std::vector<Region> regions;
  for (uintptr_t p = 0x10000; p < 0x7FFFFFFF0000ull && !g_scanStop;) {
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery(reinterpret_cast<void *>(p), &mbi, sizeof(mbi))) break;
    uintptr_t lo = reinterpret_cast<uintptr_t>(mbi.BaseAddress), hi = lo + mbi.RegionSize;
    DWORD prot = mbi.Protect & 0xFF;
    bool writable = prot == PAGE_READWRITE || prot == PAGE_WRITECOPY || prot == PAGE_EXECUTE_READWRITE;
    if (mbi.State == MEM_COMMIT && writable && !(mbi.Protect & PAGE_GUARD) && (hi <= g_ownLo || lo >= g_ownHi)) regions.push_back({lo, hi});
    p = hi;
  }
  uint64_t total = 0;
  for (auto &r : regions) total += r.hi - r.lo;
  log("ortho scan: " + std::to_string(regions.size()) + " regions, " + std::to_string(total >> 20) + " MB, y scale " + std::to_string(ys));
  constexpr size_t CHUNK = 1 << 20;
  std::vector<uint8_t> buf(CHUNK + 64);
  int hits = 0;
  uint64_t scanned = 0;
  for (auto &r : regions) {
    for (uintptr_t a = r.lo; a < r.hi && !g_scanStop && hits < MAX_HITS; a += CHUNK) {
      size_t n = static_cast<size_t>(std::min<uintptr_t>(CHUNK + 64, r.hi - a));
      if (n < 64 || !Copy(buf.data(), reinterpret_cast<void *>(a), n)) continue;
      scanned += std::min<size_t>(n, CHUNK);
      for (size_t i = 0; i + 64 <= n && i < CHUNK; i += 4) {
        const float *m = reinterpret_cast<const float *>(buf.data() + i);
        bool tr;
        if (!Projection(m, ys, tol, tr)) continue;
        uintptr_t at = a + i;
        std::string line = "ortho hit " + Where(at) + (tr ? " transposed" : "") + " m " + Floats(m, 16);
        // the camera's position (an inverse view's last row) or its negation (a straight-down view's) near the hit, by
        // offset from it, and pointers into the game's module (vtables) before it
        constexpr int AROUND = 0x400;
        float around[2 * AROUND / 4];
        if (Copy(around, reinterpret_cast<void *>(at - AROUND), sizeof(around))) {
          for (int k = 0; k + 3 <= 2 * AROUND / 4; k++) {
            const float *v = around + k;
            bool pos = std::fabs(v[0] - cx) < 0.05f && std::fabs(v[1] - cy) < 0.05f && std::fabs(v[2] - cz) < 0.05f;
            bool neg = std::fabs(v[0] + cx) < 0.05f && std::fabs(v[1] + cy) < 0.05f && std::fabs(v[2] + cz) < 0.05f;
            if (pos || neg) line += std::string(pos ? " pos@" : " -pos@") + (k * 4 < AROUND ? "-" + Hex(AROUND - k * 4) : "+" + Hex(k * 4 - AROUND));
          }
          for (int k = 0; k < AROUND / 4; k += 2) {
            uintptr_t p;
            std::memcpy(&p, around + k, 8);
            if (p >= g_exeLo && p < g_exeHi) line += " ptr@-" + Hex(AROUND - k * 4) + "=" + Where(p);
          }
        }
        log(line);
        if (++hits >= MAX_HITS) log("ortho scan: hit cap");
      }
    }
  }
  log("ortho scan: done, " + std::to_string(hits) + " hits in " + std::to_string(scanned >> 20) + " MB, " +
      std::to_string(GetTickCount64() / 1000.0 - t0) + " s");
}

// hardware write watch
struct Writer {
  std::atomic<uintptr_t> rip{0};
  std::atomic<uint32_t> count{0};
  uintptr_t stack[8] = {};
  float value = 0;
  DWORD thread = 0;
};
constexpr int WRITERS = 64;
Writer g_writers[WRITERS];
std::atomic<uint32_t> g_overflow{0};
PVOID g_veh = nullptr;
uintptr_t g_watchAddr = 0;
ULONGLONG g_watchEnd = 0;
LogFn g_watchLog;

// grcViewport (GTA V Enhanced): view at +0x40, projection at +0x80 and again at +0x180, camera matrix at +0xC0, near and
// far at +0x400/+0x404, vertical fov (degrees) at +0x410, aspect at +0x43C, perspective flag at +0x440, invert-Z at +0x441.
// A CViewport has its vtable at +0, its grcViewport at +0x10; a render phase keeps a copy of one at +0x20, taken each frame
// (exe+424790) from the CViewport its +0x498 points at.
constexpr int VP_PROJ = 0x80, VP_PROJ2 = 0x180, VP_FOV = 0x410, VP_PERSPECTIVE = 0x440;
constexpr int CVP_GRC = 0x10, CVP_COPY = 0x800, PHASE_GRC = 0x20, PHASE_VIEWPORT = 0x498;
using OrthoFn = void (*)(void *vp, float left, float right, float bottom, float top, float zNear, float zFar);
OrthoFn g_ortho = nullptr;  // grcViewport's projection of a box: left, right, bottom, top (view space), near, far
struct Box {
  float fov = 0;  // the game viewport to replace: the one with this vertical field of view, the top camera's
  float w = 76.8f, h = 43.2f, zNear = -500.0f, zFar = 2000.0f;
  bool jitter = true;  // keep the anti-aliasing jitter (as a screen offset)
  float clip = NAN;    // m in front of the camera: the GPU clips what's nearer (under-deck shots); culling keeps zNear
  float clipDepth = 120.0f;  // m the clipped depth range reaches beyond clip
};
Box g_box;
bool g_boxOn = false;

// The render phases are pointed at a copy of the game's viewport, made orthographic each tick, rather than the game's
// own, which the camera code sets back to its perspective every frame; no game code is changed. The copy lags the
// camera by a frame, which a still top camera doesn't show.
struct Phase {
  uintptr_t obj;
  uintptr_t vtable;
};
std::atomic<int> g_attach{0};  // 0 idle, 1 searching, 2 found
uintptr_t g_gameVp = 0;
std::vector<Phase> g_phases;
uint8_t *g_fake = nullptr;  // never freed: a phase may still read it after this module unloads, if a restore missed one
bool g_redirected = false;
std::thread g_attachThread;
std::atomic<bool> g_sourceSet{false};  // a grab source (the albedo target) is set

bool Exe(uintptr_t p) { return p >= g_exeLo && p < g_exeHi; }

template <class F>
void EachChunk(F &&f) {
  struct Region {
    uintptr_t lo, hi;
  };
  std::vector<Region> regions;
  for (uintptr_t p = 0x10000; p < 0x7FFFFFFF0000ull && !g_scanStop;) {
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery(reinterpret_cast<void *>(p), &mbi, sizeof(mbi))) break;
    uintptr_t lo = reinterpret_cast<uintptr_t>(mbi.BaseAddress), hi = lo + mbi.RegionSize;
    if (mbi.State == MEM_COMMIT && mbi.Type == MEM_PRIVATE && mbi.Protect == PAGE_READWRITE && (hi <= g_ownLo || lo >= g_ownHi)) regions.push_back({lo, hi});
    p = hi;
  }
  constexpr size_t CHUNK = 1 << 20;
  std::vector<uint8_t> buf(CHUNK);
  for (auto &r : regions)
    for (uintptr_t a = r.lo; a < r.hi && !g_scanStop; a += CHUNK) {
      size_t n = static_cast<size_t>(std::min<uintptr_t>(CHUNK, r.hi - a));
      if (Copy(buf.data(), reinterpret_cast<void *>(a), n)) f(a, buf.data(), n);
    }
}

void AttachThread(float fov, LogFn log) {
  double t0 = GetTickCount64() / 1000.0;
  float ys = static_cast<float>(1.0 / std::tan(0.5 * fov * DEG));
  // the game's CViewport: a vtable in the game's module, then a perspective grcViewport at this fov
  std::vector<uintptr_t> vps;
  EachChunk([&](uintptr_t a, const uint8_t *b, size_t n) {
    for (size_t i = 0; i + 0x500 <= n; i += 16) {
      float f, y;
      uintptr_t vt;
      std::memcpy(&f, b + i + CVP_GRC + VP_FOV, 4);
      if (std::fabs(f - fov) > 1e-4f) continue;
      std::memcpy(&vt, b + i, 8);
      std::memcpy(&y, b + i + CVP_GRC + VP_PROJ + 0x14, 4);
      if (Exe(vt) && std::fabs(y - ys) < 1e-3f * ys && b[i + CVP_GRC + VP_PERSPECTIVE] == 1 && vps.size() < 16) vps.push_back(a + i);
    }
  });
  for (uintptr_t v : vps) log("ortho attach: viewport " + Hex(v));
  // the render phases that copy one of them
  std::vector<Phase> phases;
  std::vector<uintptr_t> owner;
  EachChunk([&](uintptr_t a, const uint8_t *b, size_t n) {
    for (size_t i = 0; i + 8 <= n; i += 8) {
      uintptr_t p;
      std::memcpy(&p, b + i, 8);
      if (std::find(vps.begin(), vps.end(), p) == vps.end() || a + i < PHASE_VIEWPORT) continue;
      uintptr_t obj = a + i - PHASE_VIEWPORT, vt = 0;
      float f = 0;
      if (!Copy(&vt, reinterpret_cast<void *>(obj), 8) || !Exe(vt)) continue;
      if (!Copy(&f, reinterpret_cast<void *>(obj + PHASE_GRC + VP_FOV), 4)) continue;
      phases.push_back({obj, vt}), owner.push_back(p);
      log("ortho attach: phase " + Hex(obj) + " vtable " + Where(vt) + " -> viewport " + Hex(p) + ", its copy's fov " + std::to_string(f));
    }
  });
  // the viewport most phases copy is the game's
  uintptr_t best = 0;
  size_t most = 0;
  for (uintptr_t v : vps) {
    size_t k = static_cast<size_t>(std::count(owner.begin(), owner.end(), v));
    if (k > most) most = k, best = v;
  }
  std::vector<Phase> mine;
  for (size_t i = 0; i < phases.size(); i++)
    if (owner[i] == best) mine.push_back(phases[i]);
  g_gameVp = best;
  g_phases = mine;
  log("ortho attach: " + std::to_string(vps.size()) + " viewports, game viewport " + Hex(best) + " with " + std::to_string(mine.size()) +
      " phases, " + std::to_string(GetTickCount64() / 1000.0 - t0) + " s");
  g_attach = best ? 2 : 0;
}

// the copy, made orthographic: the game's viewport as it is now (its last frame), then the box's projection
void UpdateFake() {
  if (!Copy(g_fake, reinterpret_cast<void *>(g_gameVp), CVP_COPY)) return;
  uint8_t *vp = g_fake + CVP_GRC;
  Box b = g_box;
  // Under-deck shots: the depth range that clips comes from a projection from `clip` to clipDepth beyond it, while the
  // frustum the culling uses (regenerated by the second ortho call) keeps zNear. The game draws nothing whose depth
  // comes within about 1% of the near end (reversed Z: above 0.99), so the range is kept short enough that what lies
  // just under the clip plane stays under that.
  float clipZ[2] = {};
  bool clip = !std::isnan(b.clip);
  if (clip) {
    g_ortho(vp, -0.5f * b.w, 0.5f * b.w, -0.5f * b.h, 0.5f * b.h, b.clip, b.clip + b.clipDepth);
    auto c = reinterpret_cast<float *>(vp + VP_PROJ);
    clipZ[0] = c[10], clipZ[1] = c[14];
  }
  g_ortho(vp, -0.5f * b.w, 0.5f * b.w, -0.5f * b.h, 0.5f * b.h, b.zNear, b.zFar);
  if (clip) {
    auto c = reinterpret_cast<float *>(vp + VP_PROJ);
    c[10] = clipZ[0], c[14] = clipZ[1];
  }
  // the game puts its anti-aliasing jitter where a perspective divide turns it into a screen offset (m[8], m[9]); with no
  // divide it would shift each depth differently, so it becomes the plain offset (m[12], m[13]) it stands for
  auto m = reinterpret_cast<float *>(vp + VP_PROJ);
  if (b.jitter) m[12] -= m[8], m[13] -= m[9];
  m[8] = m[9] = 0;
  std::memcpy(vp + VP_PROJ2, vp + VP_PROJ, 64);
}

void Redirect(bool on) {
  uintptr_t to = on ? reinterpret_cast<uintptr_t>(g_fake) : g_gameVp, from = on ? g_gameVp : reinterpret_cast<uintptr_t>(g_fake);
  for (auto &p : g_phases) {
    auto slot = reinterpret_cast<uintptr_t *>(p.obj + PHASE_VIEWPORT);
    uintptr_t vt = 0, cur = 0;
    // only where the phase is still there and still ours to change
    if (Copy(&vt, reinterpret_cast<void *>(p.obj), 8) && vt == p.vtable && Copy(&cur, slot, 8) && cur == from) *slot = to;
  }
  g_redirected = on;
}

LONG CALLBACK OnException(EXCEPTION_POINTERS *e) {
  if (e->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP) return EXCEPTION_CONTINUE_SEARCH;
  CONTEXT *c = e->ContextRecord;
  if (!(c->Dr6 & 1) || !g_watchAddr) return EXCEPTION_CONTINUE_SEARCH;
  uintptr_t rip = c->Rip;
  c->Dr6 = 0;
  for (int i = 0, h = static_cast<int>((rip >> 2) % WRITERS); i < WRITERS; i++, h = (h + 1) % WRITERS) {
    Writer &w = g_writers[h];
    uintptr_t expect = 0;
    if (w.rip.load() == rip || w.rip.compare_exchange_strong(expect, rip)) {
      if (w.count.fetch_add(1) == 0) {
        // return addresses into the game's code above the stack pointer, a rough call stack
        ULONG_PTR lo = 0, hi = 0;
        GetCurrentThreadStackLimits(&lo, &hi);
        int n = 0;
        for (uintptr_t a = c->Rsp; a + 8 <= hi && a < c->Rsp + 256 * 8 && n < 8; a += 8) {
          uintptr_t v = *reinterpret_cast<const uintptr_t *>(a);
          if (v >= g_exeLo && v < g_exeHi) w.stack[n++] = v;
        }
        w.value = *reinterpret_cast<const float *>(g_watchAddr);
        w.thread = GetCurrentThreadId();
      }
      return EXCEPTION_CONTINUE_EXECUTION;
    }
  }
  g_overflow++;
  return EXCEPTION_CONTINUE_EXECUTION;
}

// sets (addr != 0) or clears debug register `slot` in every other thread of the process, as a 4-byte write watch or an
// execute breakpoint; run from a thread of its own so the game's main thread, which runs our script fiber, is among them
void SetBreakpoints(int slot, uintptr_t addr, bool execute, int &threads, int &failed) {
  DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId();
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
  if (snap == INVALID_HANDLE_VALUE) return;
  uint64_t mask = (3ull << (2 * slot)) | (0xFull << (16 + 4 * slot));
  uint64_t bits = (1ull << (2 * slot)) | (execute ? 0 : (1ull << (16 + 4 * slot)) | (3ull << (18 + 4 * slot)));
  THREADENTRY32 te{sizeof(te)};
  for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
    if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) continue;
    HANDLE t = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, FALSE, te.th32ThreadID);
    if (!t) {
      failed++;
      continue;
    }
    if (SuspendThread(t) != static_cast<DWORD>(-1)) {
      CONTEXT c{};
      c.ContextFlags = CONTEXT_DEBUG_REGISTERS;
      if (GetThreadContext(t, &c)) {
        (&c.Dr0)[slot] = addr;
        c.Dr7 = (c.Dr7 & ~mask) | (addr ? bits : 0);
        if (SetThreadContext(t, &c)) threads++;
        else failed++;
      } else {
        failed++;
      }
      ResumeThread(t);
    }
    CloseHandle(t);
  }
  CloseHandle(snap);
}

void Arm(int slot, uintptr_t addr, bool execute, int &threads, int &failed) {
  std::thread([&] { SetBreakpoints(slot, addr, execute, threads, failed); }).join();
}

void UpdateHandler() {
  bool want = g_watchAddr != 0;
  if (want && !g_veh) g_veh = AddVectoredExceptionHandler(1, OnException);
  else if (!want && g_veh) RemoveVectoredExceptionHandler(g_veh), g_veh = nullptr;
}

void StopWatch() {
  if (!g_watchAddr) return;
  int threads = 0, failed = 0;
  Arm(0, 0, false, threads, failed);
  std::vector<Writer *> ws;
  for (auto &w : g_writers)
    if (w.rip) ws.push_back(&w);
  std::sort(ws.begin(), ws.end(), [](Writer *a, Writer *b) { return a->count > b->count; });
  if (g_watchLog) {
    g_watchLog("ortho watch " + Hex(g_watchAddr) + " done: " + std::to_string(ws.size()) + " writers, overflow " + std::to_string(g_overflow.load()));
    for (Writer *w : ws) {
      std::string s = "ortho writer " + Where(w->rip) + " x" + std::to_string(w->count.load()) + " thread " + std::to_string(w->thread) +
                      " value " + std::to_string(w->value) + " stack";
      for (uintptr_t r : w->stack)
        if (r) s += " " + Where(r);
      g_watchLog(s);
    }
  }
  for (auto &w : g_writers) w.rip = 0, w.count = 0, std::fill(std::begin(w.stack), std::end(w.stack), 0);
  g_overflow = 0;
  g_watchAddr = 0;
  UpdateHandler();
}

// the first match of a pattern ("48 8B ? ?") in the game's module, or 0
uintptr_t Find(const char *pattern) {
  std::vector<int> bytes;
  for (const char *p = pattern; *p;) {
    if (*p == ' ') { p++; continue; }
    if (*p == '?') { bytes.push_back(-1); p++; continue; }
    bytes.push_back(static_cast<int>(std::strtoul(p, const_cast<char **>(&p), 16)));
  }
  for (uintptr_t region = g_exeLo; region < g_exeHi;) {
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery(reinterpret_cast<void *>(region), &mbi, sizeof(mbi))) break;
    uintptr_t next = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
    bool code = mbi.State == MEM_COMMIT && (mbi.Protect & (PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE));
    if (code) {
      auto p = reinterpret_cast<const uint8_t *>(region);
      size_t n = static_cast<size_t>(std::min(next, g_exeHi) - region);
      for (size_t i = 0; i + bytes.size() <= n; i++) {
        size_t j = 0;
        while (j < bytes.size() && (bytes[j] < 0 || p[i + j] == bytes[j])) j++;
        if (j == bytes.size()) return region + i;
      }
    }
    region = next;
  }
  return 0;
}

}  // namespace

void Scan(float vfovDeg, float tol, float camX, float camY, float camZ, LogFn log) {
  if (g_scan.joinable()) {
    g_scanStop = true;
    g_scan.join();
  }
  g_scanStop = false;
  float ys = static_cast<float>(1.0 / std::tan(0.5 * vfovDeg * DEG));
  g_scan = std::thread(ScanThread, ys, tol, camX, camY, camZ, std::move(log));
}

void GrabSource(uintptr_t target, int resOffset, unsigned state, LogFn log) {
  Ranges();
  if (!target) {
    present_hook::SetGrabSource(nullptr, 0);
    g_sourceSet = false;
    return log("ortho grab source: the frame");
  }
  uintptr_t res = 0;
  if (!Copy(&res, reinterpret_cast<void *>(target + resOffset), 8) || !res) return log("ortho grab source: no resource pointer");
  // a COM object whose QueryInterface gives an ID3D12Resource, checked by its vtable lying in a loaded module first
  uintptr_t vt = 0;
  HMODULE mod = nullptr;
  if (!Copy(&vt, reinterpret_cast<void *>(res), 8) ||
      !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(vt), &mod))
    return log("ortho grab source: " + Hex(res) + " isn't a COM object");
  wchar_t name[MAX_PATH] = {};
  GetModuleFileNameW(mod, name, MAX_PATH);
  ID3D12Resource *r = nullptr;
  if (FAILED(reinterpret_cast<IUnknown *>(res)->QueryInterface(IID_PPV_ARGS(&r))) || !r) return log("ortho grab source: not an ID3D12Resource");
  D3D12_RESOURCE_DESC d = r->GetDesc();
  r->Release();
  char narrow[MAX_PATH] = {};
  WideCharToMultiByte(CP_UTF8, 0, name, -1, narrow, MAX_PATH, nullptr, nullptr);
  log("ortho grab source " + Hex(res) + " (" + narrow + "): " + std::to_string(d.Width) + "x" + std::to_string(d.Height) +
      ", format " + std::to_string(int(d.Format)) + ", dimension " + std::to_string(int(d.Dimension)) + ", flags " + std::to_string(int(d.Flags)) +
      ", state at present taken as " + Hex(state));
  present_hook::SetGrabSource(reinterpret_cast<void *>(res), state);
  g_sourceSet = true;
}

void FindGBuffer(unsigned state, const std::string &name, LogFn log) {
  Ranges();
  if (g_scan.joinable()) {
    g_scanStop = true;
    g_scan.join();
  }
  g_scanStop = false;
  g_scan = std::thread([state, name, log] {
    // the render target object names itself (a heap copy of its name, e.g. "GBUFFER_0") at +0x28 and holds its
    // ID3D12Resource at +0x48
    constexpr int NAME = 0x28, RES = 0x48;
    std::string needle = name;
    const char *NEEDLE = needle.c_str();
    const size_t NEEDLE_N = needle.size() + 1;
    ULONG_PTR stackLo = 0, stackHi = 0;
    GetCurrentThreadStackLimits(&stackLo, &stackHi);
    std::vector<uintptr_t> names;
    EachChunk([&](uintptr_t a, const uint8_t *b, size_t n) {
      for (const uint8_t *p = b; names.size() < 16;) {
        p = static_cast<const uint8_t *>(std::memchr(p, NEEDLE[0], n - (p - b)));
        if (!p || static_cast<size_t>(p - b) + NEEDLE_N > n) break;
        uintptr_t at = a + (p - b);
        if (std::memcmp(p, NEEDLE, NEEDLE_N) == 0 && (p == b || p[-1] == 0) && (at < stackLo || at >= stackHi)) names.push_back(at);
        p++;
      }
    });
    uintptr_t found = 0;
    EachChunk([&](uintptr_t a, const uint8_t *b, size_t n) {
      for (size_t i = 0; i + 8 <= n && !found; i += 8) {
        uintptr_t v, vt = 0;
        std::memcpy(&v, b + i, 8);
        if (std::find(names.begin(), names.end(), v) == names.end()) continue;
        uintptr_t obj = a + i - NAME;
        if (Copy(&vt, reinterpret_cast<void *>(obj), 8) && Exe(vt)) found = obj;
      }
    });
    if (!found) return log("ortho gbuffer: " + name + " not found (" + std::to_string(names.size()) + " names)");
    log("ortho gbuffer: " + name + " at " + Hex(found));
    GrabSource(found, RES, state, log);
  });
}

void ListTargets(uintptr_t at, int before, int after, bool any, LogFn log) {
  Ranges();
  // render target objects sit 0x80 apart, each with the same vtable, its name at +0x28 and ID3D12Resource at +0x48
  uintptr_t vt0 = 0;
  if (!Copy(&vt0, reinterpret_cast<void *>(at), 8) || !Exe(vt0)) return log("ortho targets: no object at " + Hex(at));
  for (int k = -before; k <= after; k++) {
    // stride 0x80 with the same vtable; any=1: every 16 bytes, any vtable in the game's module
    uintptr_t obj = any ? at + k * 0x10 : at + k * 0x80, vt = 0, namep = 0, res = 0, rvt = 0;
    if (!Copy(&vt, reinterpret_cast<void *>(obj), 8) || (any ? !Exe(vt) : vt != vt0)) continue;
    char name[48] = {};
    if (Copy(&namep, reinterpret_cast<void *>(obj + 0x28), 8) && namep) Copy(name, reinterpret_cast<void *>(namep), sizeof(name) - 1);
    for (char &ch : name)
      if (ch && (ch < 32 || ch > 126)) ch = '?';
    std::string info;
    HMODULE mod = nullptr;
    if (Copy(&res, reinterpret_cast<void *>(obj + 0x48), 8) && res && Copy(&rvt, reinterpret_cast<void *>(res), 8) &&
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(rvt), &mod) &&
        mod == GetModuleHandleW(L"D3D12Core.dll")) {
      ID3D12Resource *r = nullptr;
      if (SUCCEEDED(reinterpret_cast<IUnknown *>(res)->QueryInterface(IID_PPV_ARGS(&r))) && r) {
        D3D12_RESOURCE_DESC d = r->GetDesc();
        r->Release();
        info = std::to_string(d.Width) + "x" + std::to_string(d.Height) + " fmt " + std::to_string(int(d.Format)) + " flags " + std::to_string(int(d.Flags)) +
               " ms " + std::to_string(d.SampleDesc.Count) + " arr " + std::to_string(d.DepthOrArraySize);
      }
    }
    if (any && info.empty()) continue;
    log("ortho target " + Hex(obj) + " [" + std::to_string(k) + "] " + name + " " + info);
  }
}

void StringScan(const std::string &text, LogFn log) {
  Ranges();
  if (g_scan.joinable()) {
    g_scanStop = true;
    g_scan.join();
  }
  g_scanStop = false;
  g_scan = std::thread([text, log] {
    std::string needle = text + '\0';
    int hits = 0;
    EachChunk([&](uintptr_t a, const uint8_t *b, size_t n) {
      for (const uint8_t *p = b; hits < 100;) {
        p = static_cast<const uint8_t *>(std::memchr(p, needle[0], n - (p - b)));
        if (!p || static_cast<size_t>(p - b) + needle.size() > n) break;
        if (std::memcmp(p, needle.data(), needle.size()) == 0 && (p == b || p[-1] == 0))
          hits++, log("ortho str " + Hex(a + (p - b)) + " " + text);
        p++;
      }
    });
    log("ortho str scan for " + text + ": " + std::to_string(hits) + " hits");
  });
}

void PointerScan(uintptr_t target, int before, LogFn log) {
  Ranges();
  if (g_scan.joinable()) {
    g_scanStop = true;
    g_scan.join();
  }
  g_scanStop = false;
  g_scan = std::thread([target, before, log] {
    int hits = 0;
    EachChunk([&](uintptr_t a, const uint8_t *b, size_t n) {
      for (size_t i = 0; i + 8 <= n && hits < 200; i += 8) {
        uintptr_t p;
        std::memcpy(&p, b + i, 8);
        if (p != target) continue;
        hits++;
        uintptr_t at = a + i, q[24] = {};
        std::string line = "ortho ptr " + Hex(at) + ":";
        if (Copy(q, reinterpret_cast<void *>(at - before), sizeof(q)))
          for (int k = 0; k < 24; k++) line += (k * 8 == before ? " [" : " ") + Where(q[k]) + (k * 8 == before ? "]" : "");
        log(line);
      }
    });
    log("ortho ptr scan for " + Where(target) + ": " + std::to_string(hits) + " hits");
  });
}

void Dump(uintptr_t addr, int count, LogFn log, bool qwords) {
  Ranges();
  count = std::clamp(count, 1, 256);
  if (qwords) {
    std::vector<uintptr_t> q(count);
    if (!Copy(q.data(), reinterpret_cast<void *>(addr), q.size() * 8)) return log("ortho dump " + Hex(addr) + ": not readable");
    for (int i = 0; i < count; i += 8) {
      std::string s = "ortho qwords " + Where(addr + i * 8) + ":";
      for (int k = i; k < std::min(count, i + 8); k++) s += " " + Where(q[k]);
      log(s);
    }
    return;
  }
  std::vector<float> v(count);
  if (!Copy(v.data(), reinterpret_cast<void *>(addr), v.size() * 4)) return log("ortho dump " + Hex(addr) + ": not readable");
  for (int i = 0; i < count; i += 16)
    log("ortho dump " + Where(addr + i * 4) + " +" + Hex(i * 4) + ": " + Floats(v.data() + i, std::min(16, count - i)));
}

void DumpImage(const std::string &path, LogFn log) {
  Ranges();
  std::vector<uint8_t> img(g_exeHi - g_exeLo);
  size_t unread = 0;
  for (size_t off = 0; off < img.size(); off += 0x1000)
    if (!Copy(img.data() + off, reinterpret_cast<void *>(g_exeLo + off), std::min<size_t>(0x1000, img.size() - off))) unread += 0x1000;
  // as a file whose sections sit at their addresses, so a disassembler loads it like the original
  auto nt = reinterpret_cast<IMAGE_NT_HEADERS *>(img.data() + reinterpret_cast<IMAGE_DOS_HEADER *>(img.data())->e_lfanew);
  auto sec = IMAGE_FIRST_SECTION(nt);
  for (int i = 0; i < nt->FileHeader.NumberOfSections; i++) {
    sec[i].PointerToRawData = sec[i].VirtualAddress;
    sec[i].SizeOfRawData = std::max<DWORD>(sec[i].Misc.VirtualSize, sec[i].SizeOfRawData);
    if (sec[i].VirtualAddress + sec[i].SizeOfRawData > img.size()) sec[i].SizeOfRawData = static_cast<DWORD>(img.size() - sec[i].VirtualAddress);
  }
  FILE *f = nullptr;
  if (fopen_s(&f, path.c_str(), "wb") || !f) return log("ortho image: can't write " + path);
  fwrite(img.data(), 1, img.size(), f);
  fclose(f);
  log("ortho image: " + std::to_string(img.size() >> 20) + " MB at base " + Hex(g_exeLo) + " to " + path + ", " + std::to_string(unread >> 12) + " pages unreadable");
}

void Watch(uintptr_t addr, float secs, LogFn log) {
  Ranges();
  StopWatch();
  g_watchLog = std::move(log);
  g_watchAddr = addr & ~uintptr_t(3);
  UpdateHandler();
  int threads = 0, failed = 0;
  Arm(0, g_watchAddr, false, threads, failed);
  g_watchEnd = GetTickCount64() + static_cast<ULONGLONG>(std::clamp(secs, 0.1f, 30.0f) * 1000);
  g_watchLog("ortho watch " + Hex(g_watchAddr) + ": " + std::to_string(threads) + " threads armed, " + std::to_string(failed) + " failed");
}

bool SetBox(bool on, float fov, float w, float h, float zNear, float zFar, bool jitter, float clip, float clipDepth, LogFn log) {
  Ranges();
  if (!g_ortho) {
    g_ortho = reinterpret_cast<OrthoFn>(Find("56 48 83 EC 50 44 0F 29 44 24 40 0F 29 7C 24 30 0F 29 74 24 20 48 89 CE F3 0F 10 84 24 90 00 00"));
    if (!g_ortho) return log("ortho: the game's ortho projection function not found"), false;
    log("ortho: ortho projection " + Where(reinterpret_cast<uintptr_t>(g_ortho)));
  }
  if (!g_fake) g_fake = static_cast<uint8_t *>(VirtualAlloc(nullptr, CVP_COPY, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
  bool refind = on && (fov != g_box.fov || g_attach == 0);
  g_box.fov = fov, g_box.w = w, g_box.h = h, g_box.zNear = zNear, g_box.zFar = zFar, g_box.jitter = jitter, g_box.clip = clip, g_box.clipDepth = clipDepth;
  g_boxOn = on;
  if (!on && g_redirected) Redirect(false);
  if (refind && g_attach != 1) {
    if (g_redirected) Redirect(false);
    if (g_attachThread.joinable()) g_attachThread.join();
    g_attach = 1;
    g_attachThread = std::thread(AttachThread, fov, log);
  }
  log("ortho box " + std::string(on ? "on" : "off") + ": fov " + std::to_string(fov) + ", " + std::to_string(w) + " x " + std::to_string(h) +
      " m, z " + std::to_string(zNear) + " to " + std::to_string(zFar) + (jitter ? "" : ", no jitter"));
  return true;
}

std::string Stats() {
  return "{\"on\":" + std::string(g_boxOn ? "true" : "false") + ",\"redirected\":" + (g_redirected ? "true" : "false") + ",\"viewport\":\"" +
         Hex(g_gameVp) + "\",\"phases\":" + std::to_string(g_phases.size()) + ",\"searching\":" + (g_attach == 1 ? "true" : "false") +
         ",\"source\":" + (g_sourceSet ? "true" : "false") + "}";
}

void Update() {
  if (g_watchAddr && GetTickCount64() >= g_watchEnd) StopWatch();
  if (g_boxOn && g_attach == 2 && g_fake) {
    UpdateFake();
    if (!g_redirected) Redirect(true);
  }
}

void Shutdown() {
  g_scanStop = true;
  if (g_scan.joinable()) g_scan.join();
  if (g_attachThread.joinable()) g_attachThread.join();
  StopWatch();
  if (g_redirected) Redirect(false);
}

}  // namespace ortho
