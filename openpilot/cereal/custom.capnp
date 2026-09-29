using Cxx = import "/include/c++.capnp";
$Cxx.namespace("cereal");

@0xb526ba661d550a59;

# custom.capnp: a home for empty structs reserved for custom forks
# These structs are guaranteed to remain reserved and empty in mainline
# cereal, so use these if you want custom events in your fork.

# DO rename the structs
# DON'T change the identifier (e.g. @0x81c2f05a394cf4af)

struct ReprojectCalibration @0x81c2f05a394cf4af {
  # reprojectcalibd: the 3X->comma 4 reprojection's narrow->wide rotation, 2 Hz and on every change. Rotations are rotvecs
  # in radians (x=pitch, y=yaw, z=roll)
  enum Status {
    waiting @0;   # no fit accepted yet; why says what it waits for
    fitting @1;   # at least one fit accepted, more to come
    building @2;  # converged, the lookup tables are being built
    fitted @3;    # applied is this unit's fit (calibrationd holds until then)
    ready @4;     # tables built: reprojectd swaps target in, then saves it
  }
  enum Why {
    none @0;
    cameras @1;
    model @2;
    speed @3;
    straight @4;
    pair @5;
    features @6;  # the last frame had too little texture to fit
  }
  status @0 :Status;
  why @1 :Why;
  pct @2 :UInt8;             # progress of the fit, 0-100
  mean @3 :List(Float32);    # component-wise median of the accepted fits
  lastFrameId @4 :UInt32;    # the narrow camera frame the last fit ran on
  lastAccepted @5 :Bool;
  applied @6 :List(Float64); # the rotation reprojectd runs
  target @7 :List(Float64);  # building / ready: the fitted rotation, exact (its tables are cached by it)
}

struct CustomReserved1 @0xaedffd8f31e7b55d {
}

struct CustomReserved2 @0xf35cc4560bbf6ec2 {
}

struct CustomReserved3 @0xda96579883444c35 {
}

struct CustomReserved4 @0x80ae746ee2596b11 {
}

struct CustomReserved5 @0xa5cd762cd951a455 {
}

struct CustomReserved6 @0xf98d843bfd7004a3 {
}

struct CustomReserved7 @0xb86e6369214c01c8 {
}

struct CustomReserved8 @0xf416ec09499d9d19 {
}

struct CustomReserved9 @0xa1680744031fdb2d {
}

struct CustomReserved10 @0xcb9fd56c7057593a {
}

struct CustomReserved11 @0xc2243c65e0340384 {
}

struct CustomReserved12 @0x9ccdc8676701b412 {
}

struct CustomReserved13 @0xcd96dafb67a082d0 {
}

struct CustomReserved14 @0xb057204d7deadf3f {
}

struct CustomReserved15 @0xbd443b539493bc68 {
}

struct CustomReserved16 @0xfc6241ed8877b611 {
}

struct CustomReserved17 @0xa30662f84033036c {
}

struct CustomReserved18 @0xc86a3d38d13eb3ef {
}

struct CustomReserved19 @0xa4f1eb3323f5f582 {
}
