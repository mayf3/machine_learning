// Copyright @2020 mayf3

#include "utils/time/stop_watch.h"

#include <iostream>
#include <utility>

namespace utils {
namespace time {

StopWatch::StopWatch(const std::string& name) : name_(name), start_time_(std::chrono::steady_clock::now()) {}

StopWatch::~StopWatch() {
  const auto now = std::chrono::steady_clock::now();
  const double cost_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_).count();
  std::cout << name_ << " cost time: " << std::fixed << cost_time << std::endl;
}

}  // namespace time
}  // namespace utils
