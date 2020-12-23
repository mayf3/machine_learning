// Copyright @2020 mayf3

#pragma once

#include <chrono>
#include <string>

namespace utils {
namespace time {

class StopWatch {
 public:
  explicit StopWatch(const std::string& name);

  virtual ~StopWatch();

 private:
  const std::string name_;
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
};

}  // namespace time
}  // namespace utils
