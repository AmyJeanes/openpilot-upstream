// Camera views: three renders a pinhole camera; a view with a lens renders a pinhole wide enough to cover it and then
// resamples that through the lens.

// undistorted ray tangent (x/z, y/z) for an output pixel offset from the image center, or null past maxAngle
const lensRay = (L, dx, dy, maxAngle) => {
  const r = Math.hypot(dx, dy), rd = r / L.f;
  if (r === 0) return [0, 0];
  let t = rd;
  if (L.model === 'fisheye') {
    const [k1, k2, k3] = L.k, tc = L.tc;
    const p = (x) => x * (1 + k1 * x ** 2 + k2 * x ** 4 + k3 * x ** 6);
    const dp = (x) => 1 + 3 * k1 * x ** 2 + 5 * k2 * x ** 4 + 7 * k3 * x ** 6;
    for (let i = 0; i < 6; i++) t -= ((t <= tc ? p(t) : p(tc) + dp(tc) * (t - tc)) - rd) / dp(Math.min(t, tc));
    if (t > maxAngle) return null;
    t = Math.tan(t);
  } else {
    for (let i = 0; i < 6; i++) t -= (t * (1 + L.k1 * t * t) - rd) / (1 + 3 * L.k1 * t * t);
    if (Math.atan(t) > maxAngle) return null;
  }
  return [dx * t / r, dy * t / r];
};

const makeView = (game, cfg, name, focal, lensCfg) => {
  const RenderTarget = game.scene.sunLight.shadow.map.constructor;
  const Camera = game.camera.constructor;
  const scale = cfg.width / 1928;
  const lens = cfg.lens && lensCfg ? { ...lensCfg, f: lensCfg.f * scale } : null;
  // with a lens, render a pinhole view wide enough to cover the lens's, at its center's resolution, and distort that
  let f = focal * scale, w = cfg.width, h = cfg.height;
  if (lens) {
    const maxAngle = cfg.lensMaxAngleDeg * Math.PI / 180;
    let tx = 0, ty = 0;
    for (let y = 0; y <= cfg.height; y += 8) {
      for (let x = 0; x <= cfg.width; x += 8) {
        const t = lensRay(lens, x - cfg.width / 2, y - cfg.height / 2, maxAngle);
        if (t) { tx = Math.max(tx, Math.abs(t[0])); ty = Math.max(ty, Math.abs(t[1])); }
      }
    }
    f = Math.min(lens.f, 4094 / (2 * tx), 4094 / (2 * ty));
    w = Math.ceil(2 * f * tx) + 2;
    h = Math.ceil(2 * f * ty) + 2;
  }
  // multisampled like the game's canvas; it also gets a 24-bit depth buffer instead of 16-bit
  const rt = new RenderTarget(w, h, { depthBuffer: true, samples: 4 });
  rt.texture.colorSpace = 'srgb';
  // three r155 only applies tone mapping and the output color space when drawing to screen or an XR target
  rt.isXRRenderTarget = true;
  // the shader already outputs sRGB; plain storage, or three picks SRGB8_ALPHA8 and the GPU encodes it a second time
  rt.texture.internalFormat = 'RGBA8';
  const vfov = 2 * Math.atan(h / 2 / f) * 180 / Math.PI;
  return { name, rt, lens, focal: f, srcW: w, srcH: h, cam: new Camera(vfov, w / h, 0.1, game.camera.far) };
};

// Resamples a view's pinhole render through its lens into a cfg-sized frame, in raw WebGL behind three's back
// (renderer.resetState() afterwards). Leaves the frame bound for reading.
const createLensPass = (renderer, cfg, views) => {
  if (!views.some(v => v.lens)) return null;
  const gl = renderer.getContext();
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('lens shader: ' + gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, `#version 300 es
    in vec2 p;
    void main() { gl_Position = vec4(p, 0.0, 1.0); }`));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, `#version 300 es
    precision highp float;
    uniform sampler2D src;
    uniform vec2 outSize, srcSize;
    uniform float f, srcF, k1, tc, maxAngle;
    uniform vec3 k;
    uniform int fisheye;
    out vec4 color;
    float thetaD(float t) { float t2 = t * t; return t * (1.0 + k.x * t2 + k.y * t2 * t2 + k.z * t2 * t2 * t2); }
    float dThetaD(float t) { float t2 = t * t; return 1.0 + 3.0 * k.x * t2 + 5.0 * k.y * t2 * t2 + 7.0 * k.z * t2 * t2 * t2; }
    void main() {
      // rows are bottom-up in GL; image y grows downwards
      vec2 d = vec2(gl_FragCoord.x, outSize.y - gl_FragCoord.y) - outSize * 0.5;
      float r = length(d), rd = r / f, rho = rd;
      if (fisheye == 1) {
        float th = rd;
        for (int i = 0; i < 6; i++) th -= ((th <= tc ? thetaD(th) : thetaD(tc) + dThetaD(tc) * (th - tc)) - rd) / dThetaD(min(th, tc));
        if (th > maxAngle) { color = vec4(0.0, 0.0, 0.0, 1.0); return; }
        rho = tan(th);
      } else {
        for (int i = 0; i < 6; i++) rho -= (rho * (1.0 + k1 * rho * rho) - rd) / (1.0 + 3.0 * k1 * rho * rho);
      }
      vec2 s = (r > 0.0 ? d * (rho / r) : vec2(0.0)) * srcF + srcSize * 0.5;
      if (any(lessThan(s, vec2(0.0))) || any(greaterThan(s, srcSize))) { color = vec4(0.0, 0.0, 0.0, 1.0); return; }
      color = texture(src, vec2(s.x / srcSize.x, 1.0 - s.y / srcSize.y));
    }`));
  gl.bindAttribLocation(prog, 0, 'p');
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('lens program: ' + gl.getProgramInfoLog(prog));
  const u = Object.fromEntries(['src', 'outSize', 'srcSize', 'f', 'srcF', 'k1', 'tc', 'maxAngle', 'k', 'fisheye']
    .map(n => [n, gl.getUniformLocation(prog, n)]));
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  for (const v of views.filter(v => v.lens)) {
    v.outTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, v.outTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, cfg.width, cfg.height);
    v.outFb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, v.outFb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, v.outTex, 0);
  }
  renderer.resetState();
  return {
    run(v) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, v.outFb);
      gl.viewport(0, 0, cfg.width, cfg.height);
      for (const cap of [gl.BLEND, gl.DEPTH_TEST, gl.SCISSOR_TEST, gl.CULL_FACE, gl.STENCIL_TEST]) gl.disable(cap);
      gl.colorMask(true, true, true, true);
      gl.useProgram(prog);
      gl.bindVertexArray(vao);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, renderer.properties.get(v.rt.texture).__webglTexture);
      gl.uniform1i(u.src, 0);
      gl.uniform2f(u.outSize, cfg.width, cfg.height);
      gl.uniform2f(u.srcSize, v.srcW, v.srcH);
      gl.uniform1f(u.f, v.lens.f);
      gl.uniform1f(u.srcF, v.focal);
      gl.uniform1f(u.k1, v.lens.k1 ?? 0);
      gl.uniform1f(u.tc, v.lens.tc ?? 10);
      gl.uniform1f(u.maxAngle, cfg.lensMaxAngleDeg * Math.PI / 180);
      gl.uniform3fv(u.k, v.lens.k ?? [0, 0, 0]);
      gl.uniform1i(u.fisheye, v.lens.model === 'fisheye' ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
    dispose() {
      gl.deleteProgram(prog); gl.deleteVertexArray(vao); gl.deleteBuffer(vbo);
      for (const v of views.filter(v => v.lens)) { gl.deleteTexture(v.outTex); gl.deleteFramebuffer(v.outFb); }
    },
  };
};
