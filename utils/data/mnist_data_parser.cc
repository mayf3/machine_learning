// Copyright @2020 mayf3

#include "utils/data/mnist_data_parser.h"

#include <iostream>
#include <memory>

namespace utils {
namespace data {

namespace {

using Matrix = MnistDataParser::Matrix;

int GetInt32ByIndex(const std::vector<uint8_t>& buffer, int index) {
  int ret = 0;
  for (int i = 0; i < 4; i++) {
    ret <<= 8;
    ret |= buffer[index + i];
  }
  return ret;
}

}  // namespace 

MnistDataParser::MnistDataParser(const char* training_image_file, const char* training_label_file,
                                 const char* testing_image_file, const char* testing_label_file) {
  ParseImage(training_image_file, &training_image_);
  ParseLabel(training_label_file, &training_label_);
  ParseImage(testing_image_file, &testing_image_);
  ParseLabel(testing_label_file, &testing_label_);
}

void MnistDataParser::ParseImage(const char* filename, std::vector<Matrix>* image) {
  std::fstream input(filename, std::ios::in | std::ios::binary);
  assert(input.is_open());
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
  assert(GetInt32ByIndex(buffer, 0) == 0x00000803);
  const int number = GetInt32ByIndex(buffer, 4);
  const int row = GetInt32ByIndex(buffer, 8);
  const int col = GetInt32ByIndex(buffer, 12);
  assert(row == 28);
  assert(col == 28);
  assert(16 + number * row * col == buffer.size());
  for (int i = 0; i < number; i++) {
    Matrix matrix;
    matrix.resize(row);
    for (int j = 0; j < row; j++) {
      matrix[j].resize(col);
      for (int k = 0; k < col; k++) {
        matrix[j][k] = buffer[16 + i * row * col + j * row + k];
      }
    }
    image->emplace_back(std::move(matrix));
  }
}

void MnistDataParser::ParseLabel(const char* filename, std::vector<int>* label) {
  std::fstream input(filename, std::ios::in | std::ios::binary);
  assert(input.is_open());
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
  assert(GetInt32ByIndex(buffer, 0) == 0x00000801);
  const int number = GetInt32ByIndex(buffer, 4);
  assert(8 + number == buffer.size());
  for (int i = 0; i < number; i++) {
    label->emplace_back(buffer[8 + i]);
  }
}

}  // namespace data
}  // namespace utils
