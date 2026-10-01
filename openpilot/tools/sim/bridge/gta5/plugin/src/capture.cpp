#include "capture.h"

#include <d3d11.h>
#include <d3dcompiler.h>
#include <dwmapi.h>
#include <dxgi1_2.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3d11.h>
#include <winrt/Windows.Graphics.DirectX.h>

#include <cmath>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "windowsapp.lib")

namespace wgc = winrt::Windows::Graphics::Capture;
namespace wgd = winrt::Windows::Graphics::DirectX;
using winrt::com_ptr;

double QpcSeconds() {
  static LARGE_INTEGER freq = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return double(t.QuadPart) / double(freq.QuadPart);
}

namespace {

// Lens models, at 1928x1208 (fleet-median comma 3X calibrations, as in the Slow Roads bridge):
// pinholeK1: r = f*rho*(1 + k1*rho^2), rho = tan(theta); fisheye: r = f*theta*(1 + k1 theta^2 + k2 theta^4 + k3 theta^6),
// linear past tc. Without lenses, openpilot's sim assumes plain pinholes with these focal lengths.
enum LensModel { PINHOLE = 0, PINHOLE_K1 = 1, FISHEYE = 2 };
struct Lens {
  int model;
  float f, k1, tc, fk[3];
};
constexpr Lens ROAD_LENS{PINHOLE_K1, 2600.85f, -0.364f, 10.0f, {0, 0, 0}};
constexpr Lens WIDE_LENS{FISHEYE, 597.732f, 0.0f, 1.51354f, {-0.011968f, 0.024043f, -0.0091132f}};
constexpr Lens ROAD_PINHOLE{PINHOLE, 2648.0f, 0, 10.0f, {0, 0, 0}};
constexpr Lens WIDE_PINHOLE{PINHOLE, 567.0f, 0, 10.0f, {0, 0, 0}};

struct alignas(16) Params {
  float outSize[2], srcOrigin[2];
  float srcSize[2], texSize[2];
  float f, srcF, k1, tc;
  float fk1, fk2, fk3, maxAngle;
  int model, pad[3];
};

// Each output pixel's ray goes back through the lens to a pixel of the game's pinhole render. The UV pass averages the 2x2
// block it covers, as openpilot's own RGB -> NV12 conversion does (BT.601, the same integer coefficients).
const char *SHADER = R"(
Texture2D src : register(t0);
SamplerState samp : register(s0);
cbuffer P : register(b0) {
  float2 outSize; float2 srcOrigin;
  float2 srcSize; float2 texSize;
  float f; float srcF; float k1; float tc;
  float fk1; float fk2; float fk3; float maxAngle;
  int model; int3 pad;
};
float4 vs(uint id : SV_VertexID) : SV_Position {
  // one clockwise triangle covering the viewport (the default rasterizer culls counterclockwise ones)
  float2 p = float2(id == 2 ? 3.0 : -1.0, id == 1 ? 3.0 : -1.0);
  return float4(p, 0.0, 1.0);
}
float thetaD(float t) { float t2 = t * t; return t * (1.0 + fk1 * t2 + fk2 * t2 * t2 + fk3 * t2 * t2 * t2); }
float dThetaD(float t) { float t2 = t * t; return 1.0 + 3.0 * fk1 * t2 + 5.0 * fk2 * t2 * t2 + 7.0 * fk3 * t2 * t2 * t2; }
float3 sampleRay(float2 p) {
  float2 d = p - outSize * 0.5;
  float r = length(d), rd = r / f, rho = rd;
  if (model == 2) {
    float th = rd;
    [unroll] for (int i = 0; i < 6; i++) th -= ((th <= tc ? thetaD(th) : thetaD(tc) + dThetaD(tc) * (th - tc)) - rd) / dThetaD(min(th, tc));
    if (th > maxAngle) return float3(0, 0, 0);
    rho = tan(th);
  } else if (model == 1) {
    [unroll] for (int i = 0; i < 6; i++) rho -= (rho * (1.0 + k1 * rho * rho) - rd) / (1.0 + 3.0 * k1 * rho * rho);
  }
  float2 s = (r > 0.0 ? d * (rho / r) : float2(0, 0)) * srcF + srcSize * 0.5;
  if (any(s < 0.0) || any(s > srcSize)) return float3(0, 0, 0);
  return src.SampleLevel(samp, (srcOrigin + s) / texSize, 0).rgb * 255.0;
}
float psY(float4 pos : SV_Position) : SV_Target {
  float3 c = sampleRay(pos.xy);
  return ((13.0 * c.b + 65.0 * c.g + 33.0 * c.r) / 128.0 + 16.0) / 255.0;
}
float2 psUV(float4 pos : SV_Position) : SV_Target {
  float2 base = floor(pos.xy) * 2.0;
  float3 c = (sampleRay(base + float2(0.5, 0.5)) + sampleRay(base + float2(1.5, 0.5)) +
              sampleRay(base + float2(0.5, 1.5)) + sampleRay(base + float2(1.5, 1.5))) * 0.25;
  float u = (56.0 * c.b - 37.0 * c.g - 19.0 * c.r) / 256.0 + 128.0;
  float v = (56.0 * c.r - 47.0 * c.g - 9.0 * c.b) / 256.0 + 128.0;
  return float2(u, v) / 255.0;
}
)";

struct View {
  Lens lens;
  com_ptr<ID3D11Texture2D> y, uv, yStage, uvStage;
  com_ptr<ID3D11RenderTargetView> yRtv, uvRtv;
};

}  // namespace

struct Capture::Impl {
  HWND hwnd = nullptr;
  CaptureConfig cfg;
  FrameCallback onFrame;
  std::function<void(const std::string &)> log;

  std::thread thread;
  std::mutex mutex;  // guards the D3D context and the stop flag
  std::condition_variable cv;
  bool stop = false;
  std::atomic<bool> running{false};

  com_ptr<ID3D11Device> device;
  com_ptr<ID3D11DeviceContext> ctx;
  com_ptr<ID3D11VertexShader> vs;
  com_ptr<ID3D11PixelShader> psY, psUV;
  com_ptr<ID3D11SamplerState> sampler;
  com_ptr<ID3D11Buffer> cb;
  View views[2];

  winrt::Windows::Graphics::SizeInt32 poolSize{};
  wgc::Direct3D11CaptureFramePool pool{nullptr};
  wgc::GraphicsCaptureSession session{nullptr};
  wgc::Direct3D11CaptureFramePool::FrameArrived_revoker revoker;
  int64_t lastFrame = 0;  // 100 ns units, in the frames' own clock
  bool loggedSize = false;

  bool InitD3D() {
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, &fl, 1, D3D11_SDK_VERSION, device.put(), nullptr, ctx.put()))) {
      log("D3D11 device creation failed");
      return false;
    }
    auto compile = [&](const char *entry, const char *target, com_ptr<ID3DBlob> &out) {
      com_ptr<ID3DBlob> err;
      if (FAILED(D3DCompile(SHADER, strlen(SHADER), "lens", nullptr, nullptr, entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, out.put(), err.put()))) {
        log(std::string("shader ") + entry + ": " + (err ? static_cast<const char *>(err->GetBufferPointer()) : "?"));
        return false;
      }
      return true;
    };
    com_ptr<ID3DBlob> vsb, ysb, uvsb;
    if (!compile("vs", "vs_5_0", vsb) || !compile("psY", "ps_5_0", ysb) || !compile("psUV", "ps_5_0", uvsb)) return false;
    device->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, vs.put());
    device->CreatePixelShader(ysb->GetBufferPointer(), ysb->GetBufferSize(), nullptr, psY.put());
    device->CreatePixelShader(uvsb->GetBufferPointer(), uvsb->GetBufferSize(), nullptr, psUV.put());

    D3D11_SAMPLER_DESC sd{};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    device->CreateSamplerState(&sd, sampler.put());

    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = sizeof(Params);
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    device->CreateBuffer(&bd, nullptr, cb.put());

    views[0].lens = cfg.lens ? ROAD_LENS : ROAD_PINHOLE;
    views[1].lens = cfg.lens ? WIDE_LENS : WIDE_PINHOLE;
    for (auto &v : views) {
      auto make = [&](int w, int h, DXGI_FORMAT fmt, com_ptr<ID3D11Texture2D> &tex, com_ptr<ID3D11Texture2D> &stage, com_ptr<ID3D11RenderTargetView> &rtv) {
        D3D11_TEXTURE2D_DESC td{};
        td.Width = w;
        td.Height = h;
        td.MipLevels = td.ArraySize = 1;
        td.Format = fmt;
        td.SampleDesc.Count = 1;
        td.BindFlags = D3D11_BIND_RENDER_TARGET;
        device->CreateTexture2D(&td, nullptr, tex.put());
        device->CreateRenderTargetView(tex.get(), nullptr, rtv.put());
        td.BindFlags = 0;
        td.Usage = D3D11_USAGE_STAGING;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        device->CreateTexture2D(&td, nullptr, stage.put());
      };
      make(CAM_W, CAM_H, DXGI_FORMAT_R8_UNORM, v.y, v.yStage, v.yRtv);
      make(CAM_W / 2, CAM_H / 2, DXGI_FORMAT_R8G8_UNORM, v.uv, v.uvStage, v.uvRtv);
    }
    return true;
  }

  // the client area's offset and size within the captured window image, which includes any frame
  bool ClientRect(int texW, int texH, float &x, float &y, float &w, float &h) {
    RECT client, frame;
    if (!GetClientRect(hwnd, &client)) return false;
    POINT origin{0, 0};
    ClientToScreen(hwnd, &origin);
    if (FAILED(DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, &frame, sizeof(frame)))) GetWindowRect(hwnd, &frame);
    x = float(std::max<LONG>(0, origin.x - frame.left));
    y = float(std::max<LONG>(0, origin.y - frame.top));
    w = std::min(float(client.right - client.left), texW - x);
    h = std::min(float(client.bottom - client.top), texH - y);
    return w > 16 && h > 16;
  }

  void Process(ID3D11Texture2D *tex, double t) {
    D3D11_TEXTURE2D_DESC desc;
    tex->GetDesc(&desc);
    float cx, cy, cw, ch;
    if (!ClientRect(desc.Width, desc.Height, cx, cy, cw, ch)) return;
    if (!loggedSize) {
      log("capturing " + std::to_string(desc.Width) + "x" + std::to_string(desc.Height) + ", client area " +
          std::to_string(int(cw)) + "x" + std::to_string(int(ch)) + " at " + std::to_string(int(cx)) + "," + std::to_string(int(cy)));
      loggedSize = true;
    }
    com_ptr<ID3D11ShaderResourceView> srv;
    if (FAILED(device->CreateShaderResourceView(tex, nullptr, srv.put()))) return;

    // square pixels with the principal point at the center: the focal length follows from the vertical FOV
    float srcF = ch * 0.5f / std::tan(cfg.vfovDeg * 3.14159265f / 360.0f);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->IASetInputLayout(nullptr);
    ctx->VSSetShader(vs.get(), nullptr, 0);
    ID3D11ShaderResourceView *srvs[] = {srv.get()};
    ctx->PSSetShaderResources(0, 1, srvs);
    ID3D11SamplerState *samps[] = {sampler.get()};
    ctx->PSSetSamplers(0, 1, samps);
    ID3D11Buffer *cbs[] = {cb.get()};
    ctx->PSSetConstantBuffers(0, 1, cbs);

    for (auto &v : views) {
      Params p{};
      p.outSize[0] = CAM_W;
      p.outSize[1] = CAM_H;
      p.srcOrigin[0] = cx;
      p.srcOrigin[1] = cy;
      p.srcSize[0] = cw;
      p.srcSize[1] = ch;
      p.texSize[0] = float(desc.Width);
      p.texSize[1] = float(desc.Height);
      p.f = v.lens.f;
      p.srcF = srcF;
      p.k1 = v.lens.k1;
      p.tc = v.lens.tc;
      p.fk1 = v.lens.fk[0];
      p.fk2 = v.lens.fk[1];
      p.fk3 = v.lens.fk[2];
      p.maxAngle = cfg.lensMaxAngleDeg * 3.14159265f / 180.0f;
      p.model = v.lens.model;
      D3D11_MAPPED_SUBRESOURCE m;
      if (FAILED(ctx->Map(cb.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return;
      memcpy(m.pData, &p, sizeof(p));
      ctx->Unmap(cb.get(), 0);

      D3D11_VIEWPORT vp{0, 0, float(CAM_W), float(CAM_H), 0, 1};
      ID3D11RenderTargetView *rtv[] = {v.yRtv.get()};
      ctx->OMSetRenderTargets(1, rtv, nullptr);
      ctx->RSSetViewports(1, &vp);
      ctx->PSSetShader(psY.get(), nullptr, 0);
      ctx->Draw(3, 0);
      vp.Width = CAM_W / 2;
      vp.Height = CAM_H / 2;
      rtv[0] = v.uvRtv.get();
      ctx->OMSetRenderTargets(1, rtv, nullptr);
      ctx->RSSetViewports(1, &vp);
      ctx->PSSetShader(psUV.get(), nullptr, 0);
      ctx->Draw(3, 0);
      ctx->CopyResource(v.yStage.get(), v.y.get());
      ctx->CopyResource(v.uvStage.get(), v.uv.get());
    }
    ID3D11RenderTargetView *none[] = {nullptr};
    ctx->OMSetRenderTargets(1, none, nullptr);
    ID3D11ShaderResourceView *noSrv[] = {nullptr};
    ctx->PSSetShaderResources(0, 1, noSrv);

    std::vector<uint8_t> out(NV12_BYTES * 2);
    uint8_t *dst = out.data();
    for (auto &v : views) {
      D3D11_MAPPED_SUBRESOURCE m;
      if (FAILED(ctx->Map(v.yStage.get(), 0, D3D11_MAP_READ, 0, &m))) return;
      for (int r = 0; r < CAM_H; r++) memcpy(dst + size_t(r) * CAM_W, static_cast<uint8_t *>(m.pData) + size_t(r) * m.RowPitch, CAM_W);
      ctx->Unmap(v.yStage.get(), 0);
      dst += size_t(CAM_W) * CAM_H;
      if (FAILED(ctx->Map(v.uvStage.get(), 0, D3D11_MAP_READ, 0, &m))) return;
      for (int r = 0; r < CAM_H / 2; r++) memcpy(dst + size_t(r) * CAM_W, static_cast<uint8_t *>(m.pData) + size_t(r) * m.RowPitch, CAM_W);
      ctx->Unmap(v.uvStage.get(), 0);
      dst += size_t(CAM_W) * CAM_H / 2;
    }
    onFrame(std::move(out), t);
  }

  void OnFrameArrived(wgc::Direct3D11CaptureFramePool const &sender, bool enabled) {
    auto frame = sender.TryGetNextFrame();
    if (!frame) return;
    std::lock_guard lk(mutex);
    if (stop) return;
    auto size = frame.ContentSize();
    if (size.Width != poolSize.Width || size.Height != poolSize.Height) {
      poolSize = size;
      pool.Recreate(Direct3DDevice(), wgd::DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, size);
      loggedSize = false;
      return;
    }
    if (!enabled) return;
    // keep a steady cadence from the ~60 Hz presents: take a frame once a period has passed, allowing for jitter
    int64_t now = frame.SystemRelativeTime().count();
    int64_t period = int64_t(1e7 / cfg.fps);
    if (now - lastFrame < period - period / 5) return;
    lastFrame = (now - lastFrame < 2 * period) ? lastFrame + period : now;
    auto access = frame.Surface().as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
    com_ptr<ID3D11Texture2D> tex;
    if (FAILED(access->GetInterface(__uuidof(ID3D11Texture2D), tex.put_void()))) return;
    // SystemRelativeTime is the QPC clock in 100 ns units, the same clock as QpcSeconds()
    Process(tex.get(), double(now) / 1e7);
  }

  winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice Direct3DDevice() {
    com_ptr<IDXGIDevice> dxgi = device.as<IDXGIDevice>();
    winrt::com_ptr<::IInspectable> insp;
    winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgi.get(), insp.put()));
    return insp.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
  }

  void Run(std::atomic<bool> *enabled, std::promise<bool> *started) {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    bool ok = false;
    try {
      ok = InitD3D();
      if (ok) {
        auto interop = winrt::get_activation_factory<wgc::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
        wgc::GraphicsCaptureItem item{nullptr};
        winrt::check_hresult(interop->CreateForWindow(hwnd, winrt::guid_of<wgc::GraphicsCaptureItem>(), winrt::put_abi(item)));
        poolSize = item.Size();
        pool = wgc::Direct3D11CaptureFramePool::CreateFreeThreaded(Direct3DDevice(), wgd::DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, poolSize);
        revoker = pool.FrameArrived(winrt::auto_revoke, [this, enabled](auto const &sender, auto const &) { OnFrameArrived(sender, *enabled); });
        session = pool.CreateCaptureSession(item);
        session.IsCursorCaptureEnabled(false);
        try {
          session.IsBorderRequired(false);
        } catch (...) {
        }
        session.StartCapture();
        log("capture started");
      }
    } catch (winrt::hresult_error const &e) {
      log("capture failed: " + winrt::to_string(e.message()));
      ok = false;
    }
    running = ok;
    started->set_value(ok);
    if (ok) {
      std::unique_lock lk(mutex);
      cv.wait(lk, [this] { return stop; });
    }
    revoker.revoke();
    if (session) session.Close();
    if (pool) pool.Close();
    session = nullptr;
    pool = nullptr;
    running = false;
    winrt::uninit_apartment();
  }
};

Capture::Capture() = default;
Capture::~Capture() { Stop(); }

bool Capture::Start(HWND hwnd, const CaptureConfig &cfg, FrameCallback onFrame, std::function<void(const std::string &)> log) {
  Stop();
  impl_ = std::make_unique<Impl>();
  impl_->hwnd = hwnd;
  impl_->cfg = cfg;
  impl_->onFrame = std::move(onFrame);
  impl_->log = std::move(log);
  std::promise<bool> started;
  auto fut = started.get_future();
  impl_->thread = std::thread([this, &started] { impl_->Run(&enabled_, &started); });
  bool ok = fut.get();
  if (!ok) Stop();
  return ok;
}

void Capture::Stop() {
  if (!impl_) return;
  {
    std::lock_guard lk(impl_->mutex);
    impl_->stop = true;
  }
  impl_->cv.notify_all();
  if (impl_->thread.joinable()) impl_->thread.join();
  impl_.reset();
}

bool Capture::Running() const { return impl_ && impl_->running; }
