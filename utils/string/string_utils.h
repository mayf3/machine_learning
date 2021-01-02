// Copyright @2020 mayf3

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace utils {
namespace string {

std::vector<std::string> Split(const std::string& s);

std::vector<std::string> Split(const std::string& s, char delimiter);

std::vector<std::string> Split(const std::string& s, char delimiter, bool allow_empty);

template<typename T>
T StringToValue(const std::string& str) {
  std::cout << " Do not implement. " << std::endl;
  assert(false);
}

template<>
inline double StringToValue<double>(const std::string& str) {
  return std::stod(str);
}

template<>
inline int StringToValue<int>(const std::string& str) {
  return std::stoi(str);
}

}  // namespace string
}  // namespace utils
