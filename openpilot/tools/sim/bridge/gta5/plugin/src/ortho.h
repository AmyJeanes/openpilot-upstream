// Orthographic top-down renders: finds the game's viewport matrices in memory and replaces the projection.
#pragma once
#include <cstdint>
#include <functional>
#include <string>

namespace ortho {

// starts a scan of the game's writable memory for perspective projection matrices whose y scale is 1/tan(vfov/2)
// (within tol, relative), on a thread of its own; each hit is logged with the camera position its viewport's inverse
// view holds, when it sits in a grcViewport at the known offsets
void Scan(float vfovDeg, float tol, float camX, float camY, float camZ, std::function<void(const std::string &)> log);
// the floats (or qwords, as offsets into the game's module where they point there) at addr, logged; a fault reads as nothing
void Dump(uintptr_t addr, int count, std::function<void(const std::string &)> log, bool qwords = false);
// grabs take the ID3D12Resource a render target object (at target) holds at +resOffset, taken to be in `state` at
// present; target 0 goes back to the frame
void GrabSource(uintptr_t target, int resOffset, unsigned state, std::function<void(const std::string &)> log);
// finds the G-buffer's albedo target (GBUFFER_0) by a memory search on a thread of its own, and makes it the grab source
void FindGBuffer(unsigned state, std::function<void(const std::string &)> log);
// logs where the game's heap holds the zero-terminated string text
void StringScan(const std::string &text, std::function<void(const std::string &)> log);
// logs where the game's heap holds the value target, with the 24 qwords from `before` bytes ahead of each
void PointerScan(uintptr_t target, int before, std::function<void(const std::string &)> log);
// the game's module as loaded, to a file a disassembler loads as a PE at the same base
void DumpImage(const std::string &path, std::function<void(const std::string &)> log);
// hardware write breakpoints on [addr, addr + 4) in every thread of the game for secs; the code that writes there is
// logged with its return addresses, as offsets into the game's module
void Watch(uintptr_t addr, float secs, std::function<void(const std::string &)> log);
// renders the game's viewport whose vertical fov is `fov` (the top camera's) orthographically: a w x h m box, zNear to
// zFar m in front of the camera (negative: behind it); the render phases are pointed at an orthographic copy of it, found
// by a memory search on a thread of its own the first time (a few tens of seconds), so no game code is changed
bool SetBox(bool on, float fov, float w, float h, float zNear, float zFar, bool jitter, std::function<void(const std::string &)> log);
// the box's state, as JSON
std::string Stats();
// call each tick: ends a watch whose time is up and logs its writers
void Update();
// stops the scan thread and clears any breakpoints, before this module unloads
void Shutdown();

}  // namespace ortho
