#pragma once

// Include this instead of <windows.h>. It keeps the Win32 headers lean and drops the
// macros that collide with identifiers in the capnp schemas. Never include it from a
// header, and never in a translation unit that also sees common/params.h: its BOOL,
// INT and FLOAT enumerators clash with the Win32 typedefs.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#include <windows.h>
#undef NO_ERROR
#undef MessageBox
#endif
