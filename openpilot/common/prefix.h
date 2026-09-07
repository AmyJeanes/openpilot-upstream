#pragma once

#include <string>

class OpenpilotPrefix {
public:
  OpenpilotPrefix(std::string prefix = {});
  ~OpenpilotPrefix();

private:
  std::string msgq_path;
};
