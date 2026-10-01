// Slow Roads -> openpilot bridge, page side. inject.ps1 runs this after lens.js, scene.js and controls.js, all in one
// function scope, once window.__srGame (the game's main object) is set.
// Renders comma-like road and wide cameras from the car, streams frames + vehicle state over a WebSocket and applies controls.
if (window.__srb) window.__srb.stop();

const cfg = Object.assign({
  url: 'ws://localhost:8790',
  fps: 20,
  width: 1928,
  height: 1208,
  bitrate: 15e6,  // per view, hardware H.264 (WebCodecs)
  // pinhole focal lengths at 1928 wide, as openpilot assumes for the AR0231/OX03C10 road and wide cameras
  roadFocal: 2648,
  wideFocal: 567,
  // render through the comma 3X's real lenses instead (fleet-median measurements, px at 1928x1208): the narrow camera is
  // a pinhole with barrel distortion r = f·ρ(1+k1ρ²), ρ = tanθ; the wide a fisheye r = f·θ(1+k1θ²+k2θ⁴+k3θ⁶), linear past tc
  lens: true,
  roadLens: { model: 'pinholeK1', f: 2600.85, k1: -0.364 },
  wideLens: { model: 'fisheye', f: 597.732, k: [-0.011968, 0.024043, -0.0091132], tc: 1.51354 },
  // the fisheye sees ~95 deg off-axis, past what one pinhole render can cover; the model only samples its central ~30 deg
  lensMaxAngleDeg: 65,
  wide: true,
  mountHeight: 1.22,    // meters above the car origin (ground level); matches the sim's calibration
  mountForward: null,   // meters forward of the car origin; null = cockpit position
  pitchDeg: 0,
}, window.__srbConfig || {});

const game = window.__srGame;
const renderer = game.renderer;
const gl = renderer.getContext();

const findCar = () => {
  let car = null;
  game.renderScene.traverse(o => { if (!car && 'controller' in o && 'speed' in o) car = o; });
  return car;
};
let car = findCar();
const getCar = () => car;

const views = [makeView(game, cfg, 'road', cfg.roadFocal, cfg.roadLens)];
if (cfg.wide) views.push(makeView(game, cfg, 'wide', cfg.wideFocal, cfg.wideLens));
const lensPass = createLensPass(renderer, cfg, views);
const renderView = createViewRenderer(game, getCar);
let ws = null, wsReady = false;
const controls = createControls(game, getCar, () => wsReady);

// binds a view's finished frame for reading; call renderer.resetState() once done when it has a lens
const bindFrame = (view) => {
  if (view.lens) lensPass.run(view);
  // a multisampled target is left bound after rendering; read from the resolved one
  else renderer.state.bindFramebuffer(gl.READ_FRAMEBUFFER, renderer.properties.get(view.rt).__webglFramebuffer);
};
const cam = views[0].cam;
const V3 = car.position.constructor;
const Q = car.quaternion.constructor;
const mount = new V3();
// three cameras look down -Z; the car's forward axis is +X
const camRot = new Q().setFromEuler(new cam.rotation.constructor(cfg.pitchDeg * Math.PI / 180, -Math.PI / 2, 0, 'YXZ'));

const updateMount = () => {
  const cock = car.worldToLocal(car.cockPosition.clone());
  mount.set(cfg.mountForward ?? cock.x, cfg.mountHeight, 0);
};
updateMount();

const bytes = cfg.width * cfg.height * 4;  // per view
const slots = [0, 1, 2].map(() => ({ pbo: gl.createBuffer(), sync: null, header: null }));
for (const s of slots) {
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, s.pbo);
  gl.bufferData(gl.PIXEL_PACK_BUFFER, bytes * views.length, gl.STREAM_READ);
}
gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);

const state = { frameId: 0, lastCapture: 0, sent: 0, dropped: 0, needKey: true, prevPos: null, resets: 0 };

// frameId -> { header, chunks: {view: Uint8Array} }, sent once every view's chunk for that frame is encoded
const pending = new Map();
const sendFrame = (header, parts) => {
  const head = new TextEncoder().encode(JSON.stringify(header));
  const total = parts.reduce((n, p) => n + p.length, 0);
  const msg = new Uint8Array(4 + head.length + total);
  new DataView(msg.buffer).setUint32(0, head.length, true);
  msg.set(head, 4);
  let off = 4 + head.length;
  for (const part of parts) { msg.set(part, off); off += part.length; }
  ws.send(msg);
  state.sent++;
};
const encoders = views.map((view) => {
  const enc = new VideoEncoder({
    output: (chunk) => {
      const id = Math.round(chunk.timestamp / 50000);
      const entry = pending.get(id);
      if (!entry) return;
      const data = new Uint8Array(chunk.byteLength);
      chunk.copyTo(data);
      entry.chunks[view.name] = { data, key: chunk.type === 'key' };
      if (Object.keys(entry.chunks).length < views.length) return;
      pending.delete(id);
      if (!wsReady) return;
      const parts = views.map(v => entry.chunks[v.name]);
      entry.header.views = views.map((v, i) => ({ name: v.name, key: parts[i].key, len: parts[i].data.length }));
      sendFrame(entry.header, parts.map(p => p.data));
    },
    error: (e) => { window.__srbError = 'encoder: ' + e; },
  });
  enc.configure({ codec: 'avc1.640033', width: cfg.width, height: cfg.height, bitrate: cfg.bitrate, framerate: cfg.fps,
                  hardwareAcceleration: 'prefer-hardware', latencyMode: 'realtime', avc: { format: 'annexb' } });
  return enc;
});
const frameBufs = views.map(() => new Uint8Array(bytes));

const connect = () => {
  ws = new WebSocket(cfg.url);
  ws.binaryType = 'arraybuffer';
  // keyframes are several times larger and arrive late enough to stall modeld, so only send them on (re)connect or when asked
  ws.onopen = () => { wsReady = true; state.needKey = true; };
  ws.onclose = () => { wsReady = false; if (!stopped) setTimeout(connect, 1000); };
  ws.onerror = () => {};
  ws.onmessage = (m) => {
    if (typeof m.data !== 'string') return;
    const msg = JSON.parse(m.data);
    if (msg.type === 'control') Object.assign(controls.control, msg, { t: performance.now() });
    if (msg.type === 'keyframe') state.needKey = true;
    if (msg.type === 'indicatorOff') controls.indicatorOff();
    if (msg.type === 'autodrive') controls.setAutodrive(!!msg.on);
  };
};

const vehicleState = () => {
  const vc = car.controller;
  // the game resets a stopped or strayed car by moving it back onto the road
  if (state.prevPos && car.position.distanceTo(state.prevPos) > 5 + car.speed) state.resets++;
  state.prevPos = car.position.clone();
  return {
    vEgo: car.speed * (car.direction < 0 ? -1 : 1),
    steer: vc.steer,
    wheelBase: car.metrics.wheelBase,
    yawRate: vc.chassisRotVel.y,
    heading: car.heading,
    aMeas: controls.lon.aMeas,
    user: { ...controls.user },
    indicator: controls.indicator(),
    autodrive: controls.autodrive(),
    resets: state.resets,
  };
};

const capture = (now) => {
  const slot = slots[state.frameId % slots.length];
  if (slot.sync) { state.dropped++; gl.deleteSync(slot.sync); slot.sync = null; }
  const prevTarget = renderer.getRenderTarget();
  views.forEach((view, i) => {
    view.cam.position.copy(mount); car.localToWorld(view.cam.position);
    view.cam.quaternion.copy(car.quaternion).multiply(camRot);
    view.cam.updateMatrixWorld(true);
    renderer.setRenderTarget(view.rt);
    renderView(view, now);
    bindFrame(view);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, slot.pbo);
    gl.readPixels(0, 0, cfg.width, cfg.height, gl.RGBA, gl.UNSIGNED_BYTE, i * bytes);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    if (view.lens) renderer.resetState();
  });
  renderer.setRenderTarget(prevTarget);
  slot.sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
  gl.flush();
  slot.header = { frameId: state.frameId++, t: now, width: cfg.width, height: cfg.height, state: vehicleState() };
};

const flush = () => {
  for (const s of slots) {
    if (!s.sync || gl.getSyncParameter(s.sync, gl.SYNC_STATUS) !== gl.SIGNALED) continue;
    gl.deleteSync(s.sync); s.sync = null;
    if (!wsReady) { state.dropped++; continue; }
    if (encoders.some(e => e.encodeQueueSize > 2) || ws.bufferedAmount > 8e6) { state.dropped++; continue; }
    const id = s.header.frameId;
    const keyFrame = state.needKey;
    state.needKey = false;
    pending.set(id, { header: s.header, chunks: {} });
    if (pending.size > 10) pending.delete(pending.keys().next().value);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, s.pbo);
    views.forEach((view, i) => {
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, i * bytes, frameBufs[i]);
      const frame = new VideoFrame(frameBufs[i], { format: 'RGBA', codedWidth: cfg.width, codedHeight: cfg.height, timestamp: id * 50000 });
      encoders[i].encode(frame, { keyFrame });
      frame.close();
    });
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  }
};

// hook the renderer rather than game.render, which the game reassigns between load and live states
const origRender = renderer.render;
let stopped = false, inCapture = false;
renderer.render = function (scene, camera) {
  const r = origRender.apply(this, arguments);
  if (inCapture || camera !== game.camera || this.getRenderTarget() !== null) return r;
  inCapture = true;
  try {
    if (!car || !car.parent) { car = findCar(); if (car) updateMount(); }
    if (car) {
      controls.hookVc(car.controller);
      const now = performance.now();
      flush();
      if (now - state.lastCapture >= 1000 / cfg.fps - 2) { state.lastCapture = now; capture(now); }
    }
  } catch (e) { window.__srbError = String(e && e.stack || e); }
  inCapture = false;
  return r;
};

connect();
window.__srb = {
  cfg, state, controls, views,
  get car() { return car; },
  // debug: read the most recent frame of a view synchronously into a JPEG blob
  snapshot(scale = 1, view = 0) {
    const px = new Uint8Array(bytes), v = views[view];
    bindFrame(v);
    gl.readPixels(0, 0, cfg.width, cfg.height, gl.RGBA, gl.UNSIGNED_BYTE, px);
    renderer.resetState();
    const c = new OffscreenCanvas(cfg.width, cfg.height), ctx = c.getContext('2d');
    const img = new ImageData(new Uint8ClampedArray(px.buffer), cfg.width, cfg.height);
    ctx.putImageData(img, 0, 0);
    const out = new OffscreenCanvas(cfg.width * scale, cfg.height * scale), o = out.getContext('2d');
    o.scale(1, -1); o.drawImage(c, 0, 0, cfg.width, cfg.height, 0, -cfg.height * scale, cfg.width * scale, cfg.height * scale);
    return out.convertToBlob({ type: 'image/jpeg', quality: 0.85 });
  },
  stop() {
    stopped = true;
    renderer.render = origRender;
    controls.dispose();
    if (ws) ws.close();
    for (const s of slots) { if (s.sync) gl.deleteSync(s.sync); gl.deleteBuffer(s.pbo); }
    views.forEach(v => v.rt.dispose());
    lensPass?.dispose();
    encoders.forEach(e => { try { e.close(); } catch (e) {} });
    delete window.__srb;
  },
};
return 'srb installed: ' + car.controller.name + ' mount=' + JSON.stringify(mount) + ' views=' + views.map(v => v.name + ':' + v.cam.fov.toFixed(1)).join(',');
