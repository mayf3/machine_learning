// Copyright @2020 mayf3

#pragma once

#include <fstream>
#include <random>

namespace utils {
namespace data {

class MnistDataParser {
 public:
  static constexpr int kRow = 28;
  static constexpr int kCol = 28;
  using Matrix = std::vector<std::vector<uint8_t>>;

  MnistDataParser(const char* training_image_file, const char* training_label_file,
                  const char* testing_image_file, const char* testing_label_file);

  const std::vector<Matrix>& training_image() const { return training_image_; }
  const std::vector<int>& training_label() const { return training_label_; }
  const std::vector<Matrix>& testing_image() const { return testing_image_; }
  const std::vector<int>& testing_label() const { return testing_label_; }

 private:
  void ParseImage(const char* filename, std::vector<Matrix>* image);

  void ParseLabel(const char* filename, std::vector<int>* label);

  std::vector<Matrix> training_image_;
  std::vector<int> training_label_;
  std::vector<Matrix> testing_image_;
  std::vector<int> testing_label_;
};

}  // namespace data
}  // namespace utils
