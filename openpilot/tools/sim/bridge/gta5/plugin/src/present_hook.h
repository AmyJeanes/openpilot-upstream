// Takes the openpilot camera's frames from the game's D3D12 presents, and shows the player's previous frame in their place.
#pragma once
#include <windows.h>
#include <cstdint>
#include <functional>
#include <string>

// an openpilot camera frame, in a texture shared with other D3D devices on the same adapter
struct HookFrame {
  HANDLE handle;  // NT handle to the texture, open while the hook's textures last
  uint64_t id;    // unique to the texture, so a consumer can keep its own opened copy
  LUID adapter;
  int width, height;
  double t;  // QPC seconds at the present
  int view;  // HOOK_BOTH: one render for both openpilot cameras; HOOK_ROAD, HOOK_WIDE: one each, road then wide
  uint64_t n;  // the present's number
};

enum { HOOK_BOTH = 0, HOOK_ROAD = 1, HOOK_WIDE = 2 };

namespace present_hook {
// onFrame runs on the hook's own thread
bool Install(std::function<void(const std::string &)> log, std::function<void(const HookFrame &)> onFrame);
void Uninstall();
// while on, frames with the interleave marker go to onFrame and are replaced on screen
void SetEnabled(bool on);
// saves the player's next frame (as the screen shows it) to a 32-bit BMP at path; false while one is pending
bool RequestGrab(const std::string &path);
// grabs take this ID3D12Resource of the game's (one of its render targets, in `state` at present) instead of the player's
// frame; nullptr goes back to the frame
void SetGrabSource(void *resource, unsigned state);
// copies the grab source after each of the game's submissions through the next whole frame, to <path minus
// extension>_<k>.bmp; the source taken to be in assumedState between submissions
bool RequestTrace(const std::string &path, unsigned assumedState);
// while on, a grab with a source takes the source from a trace of the next frame (the copy as the G-buffer passes
// leave it, before later passes write over it) instead of at the present
void SetGrabTrace(bool on);
// whether frame n's texture may have been overwritten by a later frame: a consumer that read it since should drop it
bool Reused(uint64_t n);
}  // namespace present_hook
