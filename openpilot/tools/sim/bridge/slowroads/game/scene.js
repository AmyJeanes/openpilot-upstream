// The game sets up parts of the scene for its own camera each frame, which the player may have moved anywhere; redo
// them for ours while we render and put them back for the game's next frame:
// - the car is shown or hidden by camera mode; always hide it, since from the device's spot the wide camera sees
//   through its floor and bonnet (the cabin model is on the layer we skip, and the body isn't closed from inside)
// - the night sky and distance haze is a fogged quad parented to the game camera (its mirrors clone it the same way)
// - the cloud layer and stars are moved over the game camera, and the sky shaders read its position as camPos
// - falling snow is on a layer our cameras skip (it also holds the car interior), drawn by a shader that wraps
//   world-fixed flakes into a box the game puts half a box ahead of its camera, with the `offset` uniform at its position
// Flake size in pixels is 0.045 * scale / depth, with scale half the canvas height whatever the camera's focal length;
// scale it for each of our cameras so all of them see the flakes the game shows through its default 68 degree view
const createViewRenderer = (game, getCar) => {
  const renderer = game.renderer;
  const V3 = game.camera.position.constructor;
  const pointScale = { value: 1 };
  const SNOW_REF_TAN = Math.tan(34 * Math.PI / 180);
  const canvasSize = { set(x, y) { this.width = x; this.height = y; return this; } };
  const patchSnow = (m) => {
    if (m.userData.srPointScale) return;
    const compile = m.onBeforeCompile;
    m.userData.srPointScale = pointScale;
    m.onBeforeCompile = (shader, r) => {
      compile.call(m, shader, r);
      shader.uniforms.srPointScale = pointScale;
      shader.vertexShader = 'uniform float srPointScale;\n' +
        shader.vertexShader.replace('#include <logdepthbuf_vertex>', 'gl_PointSize *= srPointScale;\n#include <logdepthbuf_vertex>');
    };
    m.needsUpdate = true;
  };

  let weather = [], weatherFound = -Infinity;
  const findWeather = (now) => {
    if (now - weatherFound < 2000) return weather;
    weather = [];
    game.renderScene.traverse(o => {
      if (o.isPoints && !(o.layers.mask & 1) && o.parent?.parent && o.material.userData.offset) {
        patchSnow(o.material);
        weather.push(o);
      }
    });
    weatherFound = now;
    return weather;
  };
  // the cloud layer and star field are kept over the game camera, and the sky shaders fade by a camPos uniform
  let followers = [], camUniforms = [], followersFound = -Infinity;
  const findFollowers = (now) => {
    if (now - followersFound < 2000) return;
    followersFound = now;
    const camPos = game.camera.getWorldPosition(new V3()), p = new V3(), uniforms = new Set();
    followers = [];
    game.renderScene.traverse(o => {
      const u = o.material?.userData?.camPos;
      if (u?.value?.isVector3) uniforms.add(u);
      if (o.renderOrder <= -9 && o.parent !== game.camera && !o.material?.userData?.offset) {
        o.getWorldPosition(p);
        if (Math.hypot(p.x - camPos.x, p.z - camPos.z) < 1) followers.push({ o, followY: Math.abs(p.y - camPos.y) < 1 });
      }
    });
    camUniforms = [...uniforms];
  };
  const fwd = new V3(), boxPos = new V3(), gameCamPos = new V3(), delta = new V3(), moved = new V3();

  return (view, now) => {
    const car = getCar();
    const gameSky = game.camera.skyPlane;
    if (gameSky && view.sky?.userData.source !== gameSky) {
      if (view.sky) view.cam.remove(view.sky);
      view.sky = gameSky.clone();
      view.sky.userData.source = gameSky;
      view.cam.add(view.sky);
    }
    if (view.sky) {
      view.sky.visible = gameSky.visible && !!gameSky.parent;  // some scenes make it without adding it to the camera
      view.sky.position.copy(gameSky.position);
      view.sky.scale.copy(gameSky.scale);
      view.sky.updateMatrixWorld(true);
    }
    findFollowers(now);
    delta.copy(view.cam.position).sub(game.camera.getWorldPosition(gameCamPos));
    const followed = followers.filter(f => f.o.parent).map(({ o, followY }) => {
      const prev = o.position.clone();
      o.getWorldPosition(moved).add(boxPos.set(delta.x, followY ? delta.y : 0, delta.z));
      o.position.copy(o.parent.worldToLocal(moved));
      o.updateMatrixWorld(true);
      return [o, prev];
    });
    const camPosSaved = camUniforms.map(u => { const prev = u.value.clone(); u.value.copy(view.cam.position); return prev; });
    view.cam.far = game.camera.far;
    view.cam.updateProjectionMatrix();

    view.cam.getWorldDirection(fwd);
    fwd.y = 0;
    fwd.normalize();
    const snow = findWeather(now).filter(o => o.parent?.parent);
    const saved = snow.map(o => {
      const volume = o.parent, offset = o.material.userData.offset.value;
      const prev = [volume.position.clone(), offset.clone()];
      boxPos.copy(fwd).multiplyScalar(o.material.userData.boxWidth.value / 2).add(view.cam.position);
      offset.copy(boxPos);
      volume.position.copy(volume.parent.worldToLocal(boxPos));
      volume.updateMatrixWorld(true);
      o.layers.enable(0);
      return prev;
    });
    pointScale.value = view.focal * SNOW_REF_TAN / (renderer.getSize(canvasSize).height / 2);
    const carVisible = car.container.visible, gameSkyVisible = gameSky?.visible;
    car.container.visible = false;
    if (gameSky) gameSky.visible = false;

    // in the scene so that its sky quad is drawn: it hides distant scenery the game leaves unshaded behind it
    game.renderScene.add(view.cam);
    renderer.render(game.renderScene, view.cam);
    game.renderScene.remove(view.cam);

    if (gameSky) gameSky.visible = gameSkyVisible;
    camUniforms.forEach((u, i) => u.value.copy(camPosSaved[i]));
    for (const [o, prev] of followed) { o.position.copy(prev); o.updateMatrixWorld(true); }
    pointScale.value = 1;
    car.container.visible = carVisible;
    snow.forEach((o, i) => {
      o.layers.disable(0);
      o.parent.position.copy(saved[i][0]);
      o.parent.updateMatrixWorld(true);
      o.material.userData.offset.value.copy(saved[i][1]);
    });
  };
};
