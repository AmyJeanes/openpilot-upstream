#include "present_hook.h"

#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <winrt/base.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <thread>

#include "capture.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using winrt::com_ptr;

namespace {

using PresentFn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT);
using Present1Fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain1 *, UINT, UINT, const DXGI_PRESENT_PARAMETERS *);
using ExecuteFn = void(STDMETHODCALLTYPE *)(ID3D12CommandQueue *, UINT, ID3D12CommandList *const *);
constexpr int PRESENT_INDEX = 8, PRESENT1_INDEX = 22;  // in IDXGISwapChain1's vtable
constexpr int EXECUTE_INDEX = 10;                      // in ID3D12CommandQueue's

// Presents we keep per-frame resources for. An openpilot frame's texture is reused this many presents later at the
// earliest, which leaves the consumer time to read it.
constexpr int FRAMES = 8;
constexpr UINT PRED_STRIDE = 16;  // per frame: is-openpilot and is-player predicates, 64 bits each

// Sets the frame's predicates from the marker the plugin draws over openpilot camera frames, its colour giving the view:
// the is-openpilot predicate holds the HookFrame view plus one.
const char *SHADER = R"(
Texture2D<float4> frame : register(t0);
RWByteAddressBuffer pred : register(u0);
cbuffer C : register(b0) { uint mx; uint my; uint off; };
[numthreads(1, 1, 1)] void main() {
  bool both = true, road = true, wide = true;
  [unroll] for (uint y = 0; y < 4; y++)
    [unroll] for (uint x = 0; x < 4; x++) {
      float3 c = frame.Load(int3(mx + x, my + y, 0)).rgb;
      bool3 hi = c > 0.8, lo = c < 0.25;
      both = both && hi.r && lo.g && hi.b;  // magenta
      road = road && lo.r && hi.g && hi.b;  // cyan
      wide = wide && hi.r && hi.g && lo.b;  // yellow
    }
  uint view = both ? 1 : road ? 2 : wide ? 3 : 0;
  pred.Store4(off, uint4(view, 0, view ? 0 : 1, 0));
}
)";

struct VtableEntry {
  void **slot = nullptr;
  void *original = nullptr;
};

VtableEntry g_present, g_present1, g_execute;
// the direct queue each thread last submitted to: the game renders a frame on the queue it then presents it on
thread_local ID3D12CommandQueue *t_lastQueue = nullptr;
std::atomic<ID3D12CommandQueue *> g_lastQueue{nullptr};
std::atomic<int> g_inflight{0};
std::atomic<bool> g_enabled{false};
thread_local bool t_inPresent = false;
std::function<void(const std::string &)> g_log;
std::function<void(const HookFrame &)> g_onFrame;

void Log(const std::string &s) {
  if (g_log) g_log("present hook: " + s);
}

bool Patch(VtableEntry &e, void **vtable, int index, void *hook) {
  DWORD old;
  if (!VirtualProtect(&vtable[index], sizeof(void *), PAGE_READWRITE, &old)) return false;
  e.slot = &vtable[index];
  e.original = *e.slot;  // before the hook can run on the render thread
  InterlockedExchangePointer(e.slot, hook);
  VirtualProtect(e.slot, sizeof(void *), old, &old);
  return true;
}

void Unpatch(VtableEntry &e) {
  if (!e.slot) return;
  DWORD old;
  if (VirtualProtect(e.slot, sizeof(void *), PAGE_READWRITE, &old)) {
    InterlockedExchangePointer(e.slot, e.original);
    VirtualProtect(e.slot, sizeof(void *), old, &old);
  }
  e.slot = nullptr;
}

DXGI_FORMAT ViewFormat(DXGI_FORMAT f) {
  switch (f) {
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    default: return f;
  }
}

D3D12_RESOURCE_BARRIER Transition(ID3D12Resource *r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
  D3D12_RESOURCE_BARRIER b{};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  b.Transition.pResource = r;
  b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  b.Transition.StateBefore = from;
  b.Transition.StateAfter = to;
  return b;
}

struct Pending {
  uint64_t n;
  double t;
};

// Everything on the game's device. Only the present thread changes it; the worker reads the frame textures under a
// shared lock, which the present thread takes exclusively to recreate them.
struct Renderer {
  IDXGISwapChain *swap = nullptr;  // identity only
  com_ptr<ID3D12Device> device;
  com_ptr<ID3D12CommandQueue> queue;
  LUID adapter{};
  com_ptr<ID3D12RootSignature> rootSig;
  com_ptr<ID3D12PipelineState> pso;
  com_ptr<ID3D12DescriptorHeap> heap;
  com_ptr<ID3D12CommandAllocator> alloc[FRAMES];
  uint64_t allocFence[FRAMES]{};
  com_ptr<ID3D12GraphicsCommandList> list;
  com_ptr<ID3D12Fence> fence;
  HANDLE event = nullptr;
  uint64_t presents = 0;
  com_ptr<ID3D12Resource> pred, readback;
  const uint64_t *readbackData = nullptr;

  D3D12_RESOURCE_DESC frameDesc{};
  com_ptr<ID3D12Resource> last, marker, frames[FRAMES];  // marker: the marker's corner, which the shader reads
  HANDLE handles[FRAMES]{};
  uint64_t ids[FRAMES]{};
  uint64_t nextId = 1;
  bool failed = false;

  void WaitIdle() {
    if (!fence || fence->GetCompletedValue() >= presents) return;
    fence->SetEventOnCompletion(presents, event);
    WaitForSingleObject(event, 2000);
  }

  void ReleaseFrames() {
    for (int i = 0; i < FRAMES; i++) {
      frames[i] = nullptr;
      if (handles[i]) CloseHandle(handles[i]);
      handles[i] = nullptr;
    }
    last = marker = nullptr;
    frameDesc = {};
  }

  void Release() {
    WaitIdle();
    ReleaseFrames();
    if (readback) readback->Unmap(0, nullptr);
    readbackData = nullptr;
    pred = readback = nullptr;
    list = nullptr;
    for (auto &a : alloc) a = nullptr;
    fence = nullptr;
    heap = nullptr;
    pso = nullptr;
    rootSig = nullptr;
    queue = nullptr;
    device = nullptr;
    swap = nullptr;
    if (event) CloseHandle(event);
    event = nullptr;
  }

  bool Init(IDXGISwapChain *sc) {
    Release();
    swap = sc;      // tried once per swap chain
    failed = true;  // until done
    // a D3D12 swap chain's "device" is the command queue it presents on, though not through every wrapper
    HRESULT hr = sc->GetDevice(IID_PPV_ARGS(queue.put()));
    if (FAILED(hr)) {
      com_ptr<IUnknown> unk;
      HRESULT hr2 = sc->GetDevice(IID_PPV_ARGS(unk.put()));
      if (SUCCEEDED(hr2)) unk.try_as(queue);
      if (!queue) {
        queue.copy_from(t_lastQueue ? t_lastQueue : g_lastQueue.load());
        char buf[96];
        snprintf(buf, sizeof(buf), "swap chain GetDevice: 0x%08lx / 0x%08lx; using the last submitted queue", hr, hr2);
        Log(buf);
      }
    }
    if (!queue) {
      Log("no command queue yet");
      swap = nullptr;  // retry next present
      return false;
    }
    if (queue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT || FAILED(queue->GetDevice(IID_PPV_ARGS(device.put())))) {
      Log("queue isn't a direct queue");
      return false;
    }
    adapter = device->GetAdapterLuid();

    com_ptr<ID3DBlob> cs, err, rs;
    if (FAILED(D3DCompile(SHADER, strlen(SHADER), "marker", nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, cs.put(), err.put()))) {
      Log(std::string("shader: ") + (err ? static_cast<const char *>(err->GetBufferPointer()) : "?"));
      return false;
    }
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable = {1, &range};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor = {0, 0};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants = {0, 0, 3};
    D3D12_ROOT_SIGNATURE_DESC rsd{3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, rs.put(), err.put())) ||
        FAILED(device->CreateRootSignature(0, rs->GetBufferPointer(), rs->GetBufferSize(), IID_PPV_ARGS(rootSig.put())))) {
      Log("root signature failed");
      return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = rootSig.get();
    pd.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    if (FAILED(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(pso.put())))) return false;

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(heap.put())))) return false;
    for (auto &a : alloc)
      if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(a.put())))) return false;
    if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc[0].get(), nullptr, IID_PPV_ARGS(list.put())))) return false;
    list->Close();
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put())))) return false;
    event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    presents = 0;
    for (auto &f : allocFence) f = 0;

    auto buffer = [&](D3D12_HEAP_TYPE type, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, com_ptr<ID3D12Resource> &out) {
      D3D12_HEAP_PROPERTIES hp{type};
      D3D12_RESOURCE_DESC d{};
      d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      d.Width = FRAMES * PRED_STRIDE;
      d.Height = d.DepthOrArraySize = d.MipLevels = 1;
      d.SampleDesc.Count = 1;
      d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      d.Flags = flags;
      return SUCCEEDED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(out.put())));
    };
    if (!buffer(D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, pred) ||
        !buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, readback))
      return false;
    void *p = nullptr;
    if (FAILED(readback->Map(0, nullptr, &p))) return false;
    readbackData = static_cast<const uint64_t *>(p);
    failed = false;
    Log("ready on the game's queue");
    return true;
  }

  bool MakeFrames(const D3D12_RESOURCE_DESC &bb, std::shared_mutex &lock) {
    WaitIdle();
    std::unique_lock lk(lock);
    ReleaseFrames();
    D3D12_RESOURCE_DESC d = bb;
    d.Alignment = 0;
    d.Flags = D3D12_RESOURCE_FLAG_NONE;
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(last.put()))))
      return false;
    D3D12_RESOURCE_DESC md = d;
    md.Width = md.Height = MARKER_PX;
    if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &md, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(marker.put()))))
      return false;
    D3D12_RESOURCE_DESC sd = d;
    sd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;  // D3D11 only opens shared textures that could be render targets
    D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = ViewFormat(d.Format);
    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(marker.get(), &sv, heap->GetCPUDescriptorHandleForHeapStart());
    for (int i = 0; i < FRAMES; i++) {
      if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &sd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(frames[i].put()))) ||
          FAILED(device->CreateSharedHandle(frames[i].get(), nullptr, GENERIC_ALL, nullptr, &handles[i]))) {
        Log("shared texture creation failed");
        ReleaseFrames();
        return false;
      }
      ids[i] = nextId++;
    }
    frameDesc = bb;
    Log("frames " + std::to_string(bb.Width) + "x" + std::to_string(bb.Height) + ", format " + std::to_string(bb.Format));
    return true;
  }

  // Queues, on the game's queue ahead of its present: the marker check; for an openpilot frame, a copy for openpilot and
  // the player's last frame copied over it; otherwise a copy kept as the player's last frame. Returns the present's number.
  uint64_t Record(IDXGISwapChain3 *sc, std::shared_mutex &lock) {
    com_ptr<ID3D12Resource> bb;
    if (FAILED(sc->GetBuffer(sc->GetCurrentBackBufferIndex(), IID_PPV_ARGS(bb.put())))) return 0;
    D3D12_RESOURCE_DESC bd = bb->GetDesc();
    if (bd.Width != frameDesc.Width || bd.Height != frameDesc.Height || bd.Format != frameDesc.Format) {
      if (!MakeFrames(bd, lock)) {
        failed = true;
        return 0;
      }
    }
    uint64_t n = presents + 1;
    int f = int(n % FRAMES);
    if (fence->GetCompletedValue() < allocFence[f]) {
      fence->SetEventOnCompletion(allocFence[f], event);
      WaitForSingleObject(event, 100);
    }
    if (FAILED(alloc[f]->Reset()) || FAILED(list->Reset(alloc[f].get(), pso.get()))) return 0;

    ID3D12Resource *frame = frames[f].get();
    auto read = D3D12_RESOURCE_STATE_COPY_SOURCE;
    auto predRead = D3D12_RESOURCE_STATE_PREDICATION | D3D12_RESOURCE_STATE_COPY_SOURCE;
    D3D12_RESOURCE_BARRIER b0[] = {
        Transition(bb.get(), D3D12_RESOURCE_STATE_PRESENT, read),
        Transition(marker.get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
        Transition(frame, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
        Transition(last.get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
    };
    list->ResourceBarrier(4, b0);
    // copied out, as the swap chain's buffers needn't allow shader reads
    D3D12_TEXTURE_COPY_LOCATION dst{marker.get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_TEXTURE_COPY_LOCATION src{bb.get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_BOX box{UINT(MARKER_AT), UINT(MARKER_AT), 0, UINT(MARKER_AT + MARKER_PX), UINT(MARKER_AT + MARKER_PX), 1};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    D3D12_RESOURCE_BARRIER b1 = Transition(marker.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list->ResourceBarrier(1, &b1);
    ID3D12DescriptorHeap *heaps[] = {heap.get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(rootSig.get());
    list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    list->SetComputeRootUnorderedAccessView(1, pred->GetGPUVirtualAddress());
    UINT consts[3] = {0, 0, UINT(f) * PRED_STRIDE};
    list->SetComputeRoot32BitConstants(2, 3, consts, 0);
    list->Dispatch(1, 1, 1);
    D3D12_RESOURCE_BARRIER b2 = Transition(pred.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, predRead);
    list->ResourceBarrier(1, &b2);

    // EQUAL_ZERO skips the commands while the predicate is zero
    UINT64 isOp = UINT64(f) * PRED_STRIDE, isPlayer = isOp + 8;
    list->SetPredication(pred.get(), isOp, D3D12_PREDICATION_OP_EQUAL_ZERO);
    list->CopyResource(frame, bb.get());
    list->SetPredication(pred.get(), isPlayer, D3D12_PREDICATION_OP_EQUAL_ZERO);
    list->CopyResource(last.get(), bb.get());
    list->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    D3D12_RESOURCE_BARRIER b3[] = {
        Transition(bb.get(), read, D3D12_RESOURCE_STATE_COPY_DEST),
        Transition(last.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE),
    };
    list->ResourceBarrier(2, b3);
    list->SetPredication(pred.get(), isOp, D3D12_PREDICATION_OP_EQUAL_ZERO);
    list->CopyResource(bb.get(), last.get());
    list->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    list->CopyBufferRegion(readback.get(), isOp, pred.get(), isOp, PRED_STRIDE);
    D3D12_RESOURCE_BARRIER b4[] = {
        Transition(bb.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT),
        Transition(frame, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
        Transition(last.get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
        Transition(pred.get(), predRead, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        Transition(marker.get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
    };
    list->ResourceBarrier(5, b4);
    if (FAILED(list->Close())) return 0;
    ID3D12CommandList *lists[] = {list.get()};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence.get(), n);
    allocFence[f] = n;
    presents = n;
    return n;
  }
} g_r;

std::mutex g_presentMutex;   // serializes presents, and Uninstall against them
std::shared_mutex g_frameMutex;
std::mutex g_queueMutex;
std::condition_variable g_queueCv;
std::deque<Pending> g_pending;
bool g_stopWorker = false;
std::thread g_worker;
// The game exits without the core's shutdown, and destroying a joinable std::thread aborts the process; by then its
// thread is gone anyway.
struct WorkerExit {
  ~WorkerExit() {
    if (g_worker.joinable()) g_worker.detach();
  }
} g_workerExit;
int g_statPresents = 0, g_statOp = 0;
double g_statT = 0;

void Worker() {
  HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  for (;;) {
    Pending p;
    {
      std::unique_lock lk(g_queueMutex);
      g_queueCv.wait(lk, [] { return g_stopWorker || !g_pending.empty(); });
      if (g_stopWorker) break;
      p = g_pending.front();
      g_pending.pop_front();
    }
    std::shared_lock lk(g_frameMutex);
    if (!g_r.fence || !g_r.readbackData) continue;
    if (g_r.fence->GetCompletedValue() < p.n) {
      g_r.fence->SetEventOnCompletion(p.n, event);
      if (WaitForSingleObject(event, 500) != WAIT_OBJECT_0) continue;
    }
    int f = int(p.n % FRAMES);
    int view = int(g_r.readbackData[f * 2]) - 1;
    bool op = view >= 0;
    g_statPresents++;
    g_statOp += op;
    if (p.t - g_statT > 10) {
      if (g_statT) Log(std::to_string(g_statPresents) + " presents, " + std::to_string(g_statOp) + " openpilot frames in 10 s");
      g_statPresents = g_statOp = 0;
      g_statT = p.t;
    }
    if (op && g_onFrame && g_r.handles[f]) {
      HookFrame hf{g_r.handles[f], g_r.ids[f], g_r.adapter, int(g_r.frameDesc.Width), int(g_r.frameDesc.Height), p.t, view};
      g_onFrame(hf);
    }
  }
  CloseHandle(event);
}

void OnPresent(IDXGISwapChain *sc, UINT flags) {
  if (!g_enabled || (flags & DXGI_PRESENT_TEST)) return;
  double t = QpcSeconds();
  std::lock_guard lk(g_presentMutex);
  if (sc != g_r.swap) {
    std::unique_lock flk(g_frameMutex);
    if (!g_r.Init(sc)) return;
  }
  if (g_r.failed) return;
  com_ptr<IDXGISwapChain3> sc3;
  if (FAILED(sc->QueryInterface(IID_PPV_ARGS(sc3.put())))) return;
  uint64_t n = g_r.Record(sc3.get(), g_frameMutex);
  if (!n) return;
  std::lock_guard qlk(g_queueMutex);
  g_pending.push_back({n, t});
  while (g_pending.size() > 2 * FRAMES) g_pending.pop_front();
  g_queueCv.notify_one();
}

void STDMETHODCALLTYPE HookExecute(ID3D12CommandQueue *q, UINT n, ID3D12CommandList *const *lists) {
  g_inflight++;
  if (q->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT) {
    t_lastQueue = q;
    g_lastQueue = q;
  }
  reinterpret_cast<ExecuteFn>(g_execute.original)(q, n, lists);
  g_inflight--;
}

HRESULT STDMETHODCALLTYPE HookPresent(IDXGISwapChain *sc, UINT sync, UINT flags) {
  g_inflight++;
  bool outer = !t_inPresent;
  t_inPresent = true;
  if (outer) OnPresent(sc, flags);
  HRESULT hr = reinterpret_cast<PresentFn>(g_present.original)(sc, sync, flags);
  if (outer) t_inPresent = false;
  g_inflight--;
  return hr;
}

HRESULT STDMETHODCALLTYPE HookPresent1(IDXGISwapChain1 *sc, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS *params) {
  g_inflight++;
  bool outer = !t_inPresent;
  t_inPresent = true;
  if (outer) OnPresent(sc, flags);
  HRESULT hr = reinterpret_cast<Present1Fn>(g_present1.original)(sc, sync, flags, params);
  if (outer) t_inPresent = false;
  g_inflight--;
  return hr;
}

// the swap chain and command queue vtables, from throwaway ones: every flip-model DXGI swap chain shares one, and every
// D3D12 queue the other
void **SwapChainVtable(void **&queueVt) {
  com_ptr<IDXGIFactory2> factory;
  com_ptr<ID3D12Device> device;
  com_ptr<ID3D12CommandQueue> queue;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(factory.put()))) ||
      FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put())))) return nullptr;
  D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
  if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.put())))) return nullptr;
  queueVt = *reinterpret_cast<void ***>(queue.get());
  WNDCLASSEXW wc{sizeof(wc)};
  wc.lpfnWndProc = DefWindowProcW;
  wc.hInstance = GetModuleHandleW(nullptr);
  wc.lpszClassName = L"gta5op_dummy";
  RegisterClassExW(&wc);
  HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"", WS_OVERLAPPEDWINDOW, 0, 0, 64, 64, nullptr, nullptr, wc.hInstance, nullptr);
  void **vt = nullptr;
  if (hwnd) {
    DXGI_SWAP_CHAIN_DESC1 d{};
    d.Width = d.Height = 64;
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.SampleDesc.Count = 1;
    d.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    d.BufferCount = 2;
    d.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    com_ptr<IDXGISwapChain1> sc;
    if (SUCCEEDED(factory->CreateSwapChainForHwnd(queue.get(), hwnd, &d, nullptr, nullptr, sc.put()))) vt = *reinterpret_cast<void ***>(sc.get());
    sc = nullptr;
    DestroyWindow(hwnd);
  }
  UnregisterClassW(wc.lpszClassName, wc.hInstance);
  return vt;
}

}  // namespace

namespace present_hook {

bool Install(std::function<void(const std::string &)> log, std::function<void(const HookFrame &)> onFrame) {
  g_log = std::move(log);
  g_onFrame = std::move(onFrame);
  void **queueVt = nullptr;
  void **vt = SwapChainVtable(queueVt);
  if (!vt || !queueVt) {
    Log("couldn't find the swap chain vtable");
    return false;
  }
  g_stopWorker = false;
  g_worker = std::thread(Worker);
  if (!Patch(g_execute, queueVt, EXECUTE_INDEX, reinterpret_cast<void *>(&HookExecute)) ||
      !Patch(g_present, vt, PRESENT_INDEX, reinterpret_cast<void *>(&HookPresent)) ||
      !Patch(g_present1, vt, PRESENT1_INDEX, reinterpret_cast<void *>(&HookPresent1))) {
    Log("vtable patch failed");
    Uninstall();
    return false;
  }
  Log("installed");
  return true;
}

void Uninstall() {
  g_enabled = false;
  Unpatch(g_present);
  Unpatch(g_present1);
  Unpatch(g_execute);
  // a present that read the hooked entry may not have entered yet
  Sleep(50);
  while (g_inflight.load()) Sleep(1);
  {
    std::lock_guard lk(g_queueMutex);
    g_stopWorker = true;
    g_pending.clear();
  }
  g_queueCv.notify_all();
  if (g_worker.joinable()) g_worker.join();
  {
    std::lock_guard lk(g_presentMutex);
    std::unique_lock flk(g_frameMutex);
    g_r.Release();
  }
  g_onFrame = nullptr;
}

void SetEnabled(bool on) { g_enabled = on; }

}  // namespace present_hook
