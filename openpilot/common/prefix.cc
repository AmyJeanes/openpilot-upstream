#include "common/prefix.h"

#include <cassert>
#include <filesystem>

#include "common/params.h"
#include "common/util.h"
#include "common/hardware/hw.h"

OpenpilotPrefix::OpenpilotPrefix(std::string prefix) {
  if (prefix.empty()) {
    prefix = util::random_string(15);
  }
  msgq_path = Path::shm_path() + "/msgq_" + prefix;
  bool ret = util::create_directories(msgq_path, 0777);
  assert(ret);
  setenv("OPENPILOT_PREFIX", prefix.c_str(), 1);
}

OpenpilotPrefix::~OpenpilotPrefix() {
  // best effort: Windows refuses to delete queue files that sockets in this process still map
  std::error_code ec;
  auto param_path = Params().getParamPath();
  if (util::file_exists(param_path)) {
#ifdef _WIN32
    std::filesystem::remove_all(param_path, ec);  // a plain directory, see params.cc
#else
    std::string real_path = util::readlink(param_path);
    util::check_system(util::string_format("rm -rf %s", real_path.c_str()));
    unlink(param_path.c_str());
#endif
  }
  if (getenv("COMMA_CACHE") == nullptr) {
    std::filesystem::remove_all(Path::download_cache_root(), ec);
  }
  std::filesystem::remove_all(Path::comma_home(), ec);
  std::filesystem::remove_all(msgq_path, ec);
  unsetenv("OPENPILOT_PREFIX");
}
