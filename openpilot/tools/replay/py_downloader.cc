#include "tools/replay/py_downloader.h"

#include <csignal>
#include <fcntl.h>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#ifdef _WIN32
#include "common/win32.h"
#else
#include <spawn.h>
#ifdef __APPLE__
#include <crt_externs.h>
#endif
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "tools/replay/util.h"

namespace {

static std::mutex handler_mutex;
static DownloadProgressHandler progress_handler = nullptr;

void reportProgress(const char *line) {
  uint64_t cur = 0, total = 0;
  if (sscanf(line, "PROGRESS:%llu:%llu", (unsigned long long *)&cur, (unsigned long long *)&total) != 2) return;
  std::lock_guard<std::mutex> lk(handler_mutex);
  if (progress_handler && total > 0) progress_handler(cur, total, true);
}

// Run a Python command and capture stdout. Stderr is scanned for PROGRESS lines and otherwise passed
// through to the parent's stderr. Returns stdout content. If abort is signaled, kills the child process.
#ifdef _WIN32
// CreateProcess with pipes instead of fork/exec; stderr still carries the progress lines
std::string runPython(const std::vector<std::string> &args, std::atomic<bool> *abort = nullptr) {
  std::string cmdline = "python3 -m openpilot.tools.lib.file_downloader";
  for (const auto &a : args) {
    cmdline += " \"" + a + "\"";
  }

  // Clear OPENPILOT_PREFIX for the child only so the Python process uses default
  // paths (e.g. ~/.comma/auth.json). The prefix is only for IPC in the parent.
  std::string env_block;
  if (LPCH env = GetEnvironmentStringsA()) {
    for (const char *p = env; *p; p += strlen(p) + 1) {
      if (strncmp(p, "OPENPILOT_PREFIX=", 17) != 0) {
        env_block.append(p, strlen(p) + 1);
      }
    }
    FreeEnvironmentStringsA(env);
  }
  env_block.push_back('\0');

  SECURITY_ATTRIBUTES sa = {};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  HANDLE out_r = NULL, out_w = NULL, err_r = NULL, err_w = NULL;
  if (!CreatePipe(&out_r, &out_w, &sa, 0) || !CreatePipe(&err_r, &err_w, &sa, 0)) {
    rWarning("py_downloader: CreatePipe() failed");
    return {};
  }
  SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(err_r, HANDLE_FLAG_INHERIT, 0);

  STARTUPINFOA si = {};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = out_w;
  si.hStdError = err_w;
  si.hStdInput = INVALID_HANDLE_VALUE;
  PROCESS_INFORMATION pi = {};
  BOOL ok = CreateProcessA(NULL, cmdline.data(), NULL, NULL, TRUE, CREATE_NO_WINDOW, env_block.data(), NULL, &si, &pi);
  CloseHandle(out_w);
  CloseHandle(err_w);
  if (!ok) {
    CloseHandle(out_r);
    CloseHandle(err_r);
    rWarning("py_downloader: CreateProcess() failed: %lu", GetLastError());
    return {};
  }
  CloseHandle(pi.hThread);

  std::thread stderr_thread([err_r]() {
    std::string line;
    char buf[4096];
    DWORD n = 0;
    auto flush_line = [&line]() {
      if (strncmp(line.c_str(), "PROGRESS:", 9) == 0) {
        reportProgress(line.c_str());
      } else {
        fprintf(stderr, "%s\n", line.c_str());
      }
      line.clear();
    };
    while (ReadFile(err_r, buf, sizeof(buf), &n, NULL) && n > 0) {
      for (DWORD i = 0; i < n; i++) {
        if (buf[i] == '\n') {
          flush_line();
        } else {
          line.push_back(buf[i]);
        }
      }
    }
    if (!line.empty()) flush_line();
    CloseHandle(err_r);
  });

  std::string stdout_data;
  char buf[4096];
  DWORD n = 0;
  // Poll the pipe so abort can interrupt while waiting for Python output
  while (true) {
    if (abort && *abort) {
      TerminateProcess(pi.hProcess, 1);
      break;
    }
    DWORD avail = 0;
    if (!PeekNamedPipe(out_r, NULL, 0, NULL, &avail, NULL)) break;  // writer gone and pipe drained
    if (avail == 0) {
      if (WaitForSingleObject(pi.hProcess, 100) == WAIT_OBJECT_0) {
        if (!PeekNamedPipe(out_r, NULL, 0, NULL, &avail, NULL) || avail == 0) break;
      }
      continue;
    }
    if (!ReadFile(out_r, buf, sizeof(buf), &n, NULL) || n == 0) break;
    stdout_data.append(buf, n);
  }
  while (ReadFile(out_r, buf, sizeof(buf), &n, NULL) && n > 0) {
    stdout_data.append(buf, n);
  }
  CloseHandle(out_r);
  stderr_thread.join();

  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD code = 1;
  GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hProcess);

  const bool aborted = abort && *abort;
  if (aborted || code != 0) {
    if (!aborted) rWarning("py_downloader: process exited with code %lu", code);
    std::lock_guard<std::mutex> lk(handler_mutex);
    if (progress_handler) {
      progress_handler(0, 0, false);
    }
    return {};
  }

  // Trim trailing newline
  while (!stdout_data.empty() && (stdout_data.back() == '\n' || stdout_data.back() == '\r')) {
    stdout_data.pop_back();
  }
  return stdout_data;
}
#else
std::string runPython(const std::vector<std::string> &args, std::atomic<bool> *abort = nullptr) {
  // Build argv for the downloader module
  std::vector<const char *> argv;
  argv.push_back("python3");
  argv.push_back("-m");
  argv.push_back("openpilot.tools.lib.file_downloader");
  for (const auto &a : args) {
    argv.push_back(a.c_str());
  }
  argv.push_back(nullptr);

  auto open_pipe = [](int (&fds)[2]) {
#ifdef __linux__
    return pipe2(fds, O_CLOEXEC);
#else
    if (pipe(fds) != 0) return -1;
    if (fcntl(fds[0], F_SETFD, FD_CLOEXEC) == 0 && fcntl(fds[1], F_SETFD, FD_CLOEXEC) == 0) return 0;
    close(fds[0]); close(fds[1]);
    return -1;
#endif
  };
  int stdout_pipe[2], stderr_pipe[2];
  if (open_pipe(stdout_pipe) != 0) {
    rWarning("py_downloader: pipe() failed");
    return {};
  }
  if (open_pipe(stderr_pipe) != 0) {
    rWarning("py_downloader: pipe() failed");
    close(stdout_pipe[0]); close(stdout_pipe[1]);
    return {};
  }

  // Avoid copying the large replay address space and running atfork handlers on
  // every segment download: both can stall rendering even from a worker thread.
  std::vector<std::string> environment;
#ifdef __APPLE__
  char **parent_environment = *_NSGetEnviron();
#else
  char **parent_environment = environ;
#endif
  for (char **entry = parent_environment; *entry; ++entry) {
    if (strncmp(*entry, "OPENPILOT_PREFIX=", 17) != 0) environment.emplace_back(*entry);
  }
  std::vector<char *> envp;
  for (auto &entry : environment) envp.push_back(entry.data());
  envp.push_back(nullptr);

  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  int error = posix_spawn_file_actions_init(&actions);
  const bool actions_initialized = error == 0;
  if (!error) error = posix_spawnattr_init(&attributes);
  const bool attributes_initialized = error == 0;
  if (!error) error = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
  if (!error) error = posix_spawn_file_actions_adddup2(&actions, stdout_pipe[1], STDOUT_FILENO);
  if (!error) error = posix_spawn_file_actions_adddup2(&actions, stderr_pipe[1], STDERR_FILENO);
  for (int fd : {stdout_pipe[0], stdout_pipe[1], stderr_pipe[0], stderr_pipe[1]}) {
    if (!error) error = posix_spawn_file_actions_addclose(&actions, fd);
  }
#ifdef POSIX_SPAWN_SETSID
  if (!error) error = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSID);
#else
  if (!error) error = posix_spawnattr_setpgroup(&attributes, 0);
  if (!error) error = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
#endif
  pid_t pid = -1;
  if (!error) error = posix_spawnp(&pid, "python3", &actions, &attributes, const_cast<char *const *>(argv.data()), envp.data());
  if (attributes_initialized) posix_spawnattr_destroy(&attributes);
  if (actions_initialized) posix_spawn_file_actions_destroy(&actions);
  if (error) {
    rWarning("py_downloader: posix_spawnp() failed: %s", strerror(error));
    close(stdout_pipe[0]); close(stdout_pipe[1]);
    close(stderr_pipe[0]); close(stderr_pipe[1]);
    return {};
  }

  // Parent process
  close(stdout_pipe[1]);
  close(stderr_pipe[1]);

  // stderr carries the progress lines, so a thread reads it while the loop below waits on stdout
  std::thread stderr_thread([fd = stderr_pipe[0]]() {
    FILE *f = fdopen(fd, "r");
    if (!f) {
      close(fd);
      return;
    }
    char *line = nullptr;
    size_t cap = 0;
    while (getline(&line, &cap, f) > 0) {
      if (strncmp(line, "PROGRESS:", 9) == 0) {
        reportProgress(line);
      } else {
        fputs(line, stderr);
      }
    }
    free(line);
    fclose(f);
  });

  std::string stdout_data;
  char buf[4096];

  // Use select() so abort can interrupt while waiting for Python output.
  fd_set rfds;
  bool stdout_open = true;

  while (stdout_open) {
    if (abort && *abort) {
      kill(pid, SIGTERM);
      break;
    }

    FD_ZERO(&rfds);
    FD_SET(stdout_pipe[0], &rfds);

    struct timeval tv = {0, 100000};  // 100ms timeout
    int ret = select(stdout_pipe[0] + 1, &rfds, nullptr, nullptr, &tv);
    if (ret < 0) break;

    if (FD_ISSET(stdout_pipe[0], &rfds)) {
      ssize_t n = read(stdout_pipe[0], buf, sizeof(buf));
      if (n <= 0) {
        stdout_open = false;
      } else {
        stdout_data.append(buf, n);
      }
    }
  }

  // Drain remaining pipe data to prevent child from blocking on write
  while (true) {
    ssize_t n = read(stdout_pipe[0], buf, sizeof(buf));
    if (n <= 0) break;
    stdout_data.append(buf, n);
  }
  close(stdout_pipe[0]);
  stderr_thread.join();

  int status;
  waitpid(pid, &status, 0);

  const bool aborted = abort && *abort;
  const bool expected_sigterm = aborted && WIFSIGNALED(status) && WTERMSIG(status) == SIGTERM;
  bool failed = aborted ||
                (WIFEXITED(status) && WEXITSTATUS(status) != 0) ||
                WIFSIGNALED(status);
  if (failed) {
    if (expected_sigterm) {
      // Route/camera teardown cancels outstanding downloader subprocesses.
      // Keep that expected shutdown path quiet.
    } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
      rWarning("py_downloader: process exited with code %d", WEXITSTATUS(status));
    } else if (WIFSIGNALED(status)) {
      rWarning("py_downloader: process killed by signal %d", WTERMSIG(status));
    }
    std::lock_guard<std::mutex> lk(handler_mutex);
    if (progress_handler) {
      progress_handler(0, 0, false);
    }
    return {};
  }

  // Trim trailing newline
  while (!stdout_data.empty() && (stdout_data.back() == '\n' || stdout_data.back() == '\r')) {
    stdout_data.pop_back();
  }

  return stdout_data;
}
#endif

}  // namespace

void installDownloadProgressHandler(DownloadProgressHandler handler) {
  std::lock_guard<std::mutex> lk(handler_mutex);
  progress_handler = handler;
}

namespace PyDownloader {

std::string download(const std::string &url, bool use_cache, std::atomic<bool> *abort) {
  std::vector<std::string> args = {"download", url};
  if (!use_cache) {
    args.push_back("--no-cache");
  }
  return runPython(args, abort);
}

std::string decompress(const std::string &path, std::atomic<bool> *abort) {
  return runPython({"decompress", path}, abort);
}

std::string getRouteFiles(const std::string &route) {
  return runPython({"route-files", route});
}

std::string getDevices() {
  return runPython({"devices"});
}

std::string getDeviceRoutes(const std::string &dongle_id, int64_t start_ms, int64_t end_ms, bool preserved) {
  std::vector<std::string> args = {"device-routes", dongle_id};
  if (preserved) {
    args.push_back("--preserved");
  } else {
    if (start_ms > 0) {
      args.push_back("--start");
      args.push_back(std::to_string(start_ms));
    }
    if (end_ms > 0) {
      args.push_back("--end");
      args.push_back(std::to_string(end_ms));
    }
  }
  return runPython(args);
}

}  // namespace PyDownloader
