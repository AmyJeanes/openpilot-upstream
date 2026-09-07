#pragma once

// File primitives whose POSIX spelling differs on Windows. Kept apart from util.h: that header's Rect collides with
// the one MacTypes.h brings in when a translation unit also includes CoreFoundation on macOS (cabana's settings).
#ifdef _WIN32
#include <io.h>
inline int fsync(int fd) { return _commit(fd); }
#endif

namespace util {
// an exclusive lock held until fd closes, and a rename that replaces an existing target (rename() refuses to on Windows)
int lock_file_exclusive(int fd);
int replace_file(const char *from, const char *to);
}  // namespace util
