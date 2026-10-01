// Captures the game window and resamples it into openpilot's road and wide camera frames (NV12, 1928x1208) on the GPU.
#pragma once
#include <windows.h>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

constexpr int CAM_W = 1928, CAM_H = 1208;
constexpr size_t NV12_BYTES = size_t(CAM_W) * CAM_H * 3 / 2;  // unpadded: Y rows, then interleaved UV rows

struct CaptureConfig {
  float vfovDeg = 76.0f;     // vertical field of view the game camera renders with
  bool lens = true;          // the comma 3X lenses, else openpilot's plain pinhole cameras
  float lensMaxAngleDeg = 65.0f;
  float fps = 20.0f;
};

// frames: road then wide, each NV12_BYTES; t: QPC seconds when the game presented the frame
using FrameCallback = std::function<void(std::vector<uint8_t> &&frames, double t)>;

class Capture {
 public:
  Capture();
  ~Capture();
  bool Start(HWND hwnd, const CaptureConfig &cfg, FrameCallback onFrame, std::function<void(const std::string &)> log);
  void Stop();
  void SetEnabled(bool on) { enabled_ = on; }
  bool Running() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::atomic<bool> enabled_{false};
};

double QpcSeconds();
