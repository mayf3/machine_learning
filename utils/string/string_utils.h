// Copyright @2020 mayf3

#pragma once

#include <string>
#include <vector>

namespace utils {
namespace string {

std::vector<std::string> Split(const std::string& s);

std::vector<std::string> Split(const std::string& s, char delimiter);

std::vector<std::string> Split(const std::string& s, char delimiter, bool allow_empty);

}  // namespace string
}  // namespace utils
