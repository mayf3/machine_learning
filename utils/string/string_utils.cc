// Copyright @2020 mayf3

#include "utils/string/string_utils.h"

namespace utils {
namespace string {

namespace {

constexpr char kDefaultDelimeter = ',';

template <class OutStream>
void Split(const std::string& s, char delimeter, int delimeter_length, bool allow_empty,
           OutStream out) {
  assert(delimeter_length > 0);
  int start_pos = 0;
  int end_pos = 0;
  while (start_pos < s.length() && (end_pos = s.find(delimeter, start_pos)) != std::string::npos) {
    if (allow_empty || start_pos != end_pos) {
      *out = s.substr(start_pos, end_pos - start_pos);
    }
    start_pos = end_pos + delimeter_length;
  }
  if (allow_empty || start_pos != s.length()) {
    *out = s.substr(start_pos);
  }
}

}  // namespace

std::vector<std::string> Split(const std::string& s) { return Split(s, kDefaultDelimeter); }

std::vector<std::string> Split(const std::string& s, char delimiter) {
  return Split(s, delimiter, /*allow_empty*/ true);
}

std::vector<std::string> Split(const std::string& s, char delimiter, bool allow_empty) {
  std::vector<std::string> tokens;
  Split(s, delimiter, 1, allow_empty, std::back_inserter(tokens));
  return tokens;
}

}  // namespace string
}  // namespace utils
