// Applies openpilot's commands to the game car and links the game's driver controls to openpilot's: the autodrive toggle is
// the engage button, the brake disengages, gas and steering override, and the indicators are the blinker stalk.
const createControls = (game, getCar, isConnected) => {
  // steer: road-wheel angle in the game's sign convention (rad); accelCmd: requested acceleration (m/s^2)
  const control = { active: false, steer: 0, accelCmd: 0, t: 0, timeoutMs: 300 };
  // longitudinal: feed-forward from the car's tuning plus a slow integral on measured acceleration
  const lon = { prevSpeed: null, aMeas: 0, integral: 0, brakeLerp: 0, braking: false };
  // like a car, slow down a little by coasting and only brake for more, with hysteresis: the request hovers around zero
  // when cruising, and braking on every dip below it flickers the brake lights (m/s^2)
  const BRAKE_ON = 0.3, BRAKE_OFF = 0.1;
  const user = { steer: 0, accel: false, brake: false };
  // the driver's controls, exposed by inject.ps1; absent if the game changed how it reads them
  const input = window.__srInput;
  const vm = game.vehicleManager;
  // the game's autodrive toggle button stands in for openpilot's engage button: the bridge mirrors openpilot's state onto
  // it by pressing the button for a frame, which keeps the game's UI and its own autodrive bookkeeping consistent
  let autodrivePresses = 0, pressed = input?.signal.Autodrive;
  if (input) {
    Object.defineProperty(input.signal, 'Autodrive', {
      configurable: true, enumerable: true,
      get() { if (autodrivePresses) { autodrivePresses--; return 1; } return pressed; },
      set(v) { pressed = v; },
    });
  }
  // The game drops autodrive on any pedal or steering input. openpilot only disengages on the brake (which reaches it, and
  // then turns autodrive off), and lets the driver override the throttle and steering, so stop the game doing that.
  let canDisableAutodrive = vm.canDisableAutodrive;
  Object.defineProperty(vm, 'canDisableAutodrive', {
    configurable: true, enumerable: true,
    get() { return !isConnected() && canDisableAutodrive; },
    set(v) { canDisableAutodrive = v; },
  });
  const pedal = (name) => Math.max(input?.signal[name] || 0, input?.gamepadSignal[name] || 0);
  const live = () => control.active && performance.now() - control.t < control.timeoutMs;

  // handleInput() ends by calling these two, so overriding inputs at their entry replaces the driver's/autodrive's.
  let hookedVc = null, origSteerState = null, origAccelState = null;
  const unhookVc = () => {
    if (!hookedVc) return;
    delete hookedVc.updateSteerState; delete hookedVc.updateAccelState;
    hookedVc = null;
  };
  const hookVc = (vc) => {
    if (hookedVc === vc) return;
    unhookVc();
    origSteerState = vc.updateSteerState; origAccelState = vc.updateAccelState;
    vc.updateSteerState = function (t) {
      user.steer = this.inputs.steer;
      if (live()) {
        // the driver's steering adds to openpilot's, as their torque would on a real wheel
        const maxSteer = getCar().tuning.maxSteer;
        this.inputs.steer = Math.max(-maxSteer, Math.min(maxSteer, control.steer + user.steer));
        this.autoSteer = 0;
      }
      return origSteerState.call(this, t);
    };
    vc.updateAccelState = function (t) {
      const car = getCar();
      // read the pedals directly too: the vehicle ignores them while the game's autodrive drives
      const gas = pedal('Forward');
      user.accel = this.hasManualAccel || gas > 0.05; user.brake = this.hasManualBrake || pedal('Backward') > 0.05;
      const speed = car.speed * (car.direction < 0 ? -1 : 1);
      if (lon.prevSpeed !== null && t > 0) {
        const raw = (speed - lon.prevSpeed) / t;
        // a game reset teleports the car and zeroes its speed; no accelerometer sees that, and the spike corrupts locationd
        if (Math.abs(raw) < 30) lon.aMeas = Math.max(-15, Math.min(15, lon.aMeas + 0.1 * (raw - lon.aMeas)));
      }
      lon.prevSpeed = speed;
      if (live()) {
        if (Math.abs(speed) < 0.3 && control.accelCmd <= 0) {
          // holding at a stop can't decelerate further; integrating that error would wind up and block pulling away
          lon.integral = 0;
        } else {
          lon.integral = Math.max(-1.5, Math.min(1.5, lon.integral + 2 * (control.accelCmd - lon.aMeas) * t));
          if (Math.abs(speed) < 0.3) lon.integral = Math.max(0, lon.integral);
        }
        // stay within what openpilot itself requests (-3.5..2 m/s^2): overshooting trips its excessive-actuation check
        const u = Math.max(-4, Math.min(2.5, control.accelCmd + lon.integral));
        // hold at a stop like a real car's brake hold: the game's brakes only oppose motion, so on a slope it creeps back and forth
        const hold = Math.abs(speed) < 0.3 && control.accelCmd < 0;
        this.isDriven = true; this.prevDriveDir = 1; this.holdHandbrake = hold; car.hasHandbrake = hold;
        if (this.softBraking) this.setSoftBrake(false);
        lon.braking = u < (lon.braking ? -BRAKE_OFF : -BRAKE_ON) && gas <= 0.05;
        if (gas > 0.05) lon.integral = 0;  // the driver's throttle overrides openpilot's, which would otherwise wind up
        if (!lon.braking) {
          this.inputs.accel = Math.max(gas, Math.min(1, u / this.tuning.accel));
          this.setBrake(false);
          lon.brakeLerp = 0;
        } else {
          this.inputs.accel = 0;
          // set directly: handleInput releases the brake every tick (autodrive), and setBrake(true) would restart the brake ramp
          this.braking = true;
          this.inputs.brake = Math.min(1, -u / this.tuning.brake);
          this.brakeLerp = lon.brakeLerp;
        }
      } else {
        lon.integral = 0;
        lon.brakeLerp = 0;
        lon.braking = false;
      }
      const r = origAccelState.call(this, t);
      if (live() && this.braking) lon.brakeLerp = this.brakeLerp;
      return r;
    };
    hookedVc = vc;
  };

  return {
    control, user, lon, hookVc,
    autodrive: () => input ? !!vm.hasAutodrive : null,
    setAutodrive(on) {
      if (input && vm.hasAutodrive !== on && !autodrivePresses) autodrivePresses = 1;
    },
    indicator() {
      const ind = getCar().indicators;
      return ind?.active ? (ind.left ? 'left' : ind.right ? 'right' : null) : null;
    },
    // the game toggles an indicator with its arrow key; it polls input each frame, so hold the key long enough to be seen
    indicatorOff() {
      const ind = getCar().indicators;
      if (!ind?.active || !(ind.left || ind.right)) return;
      const code = ind.left ? 'ArrowLeft' : 'ArrowRight';
      const key = (type) => window.dispatchEvent(new KeyboardEvent(type, { code, key: code, bubbles: true }));
      key('keydown');
      setTimeout(() => key('keyup'), 120);
    },
    dispose() {
      unhookVc();
      if (input) { delete input.signal.Autodrive; input.signal.Autodrive = pressed; }
      delete vm.canDisableAutodrive;
      vm.canDisableAutodrive = canDisableAutodrive;
    },
  };
};
