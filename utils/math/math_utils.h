// Copyright @2020 mayf3

#pragma once

#include <cmath>

#include "utils/common/define.h"

namespace utils {
namespace math {

template <typename T>
constexpr T Sqr(T x) {
  return x * x;
}

// Normalize Data : value = (value - min_value) / (max_value - min_value)
template <typename T>
void NormalizeByMinMax(T* feature_list) {
  assert(feature_list != nullptr);
  assert(feature_list->size() > 0);
  for (int feature_index = 0; feature_index < (*feature_list)[0].size(); feature_index++) {
    T min_value = (*feature_list)[0][feature_index];
    T max_value = (*feature_list)[0][feature_index];
    for (int i = 0; i < feature_list->size(); i++) {
      min_value = std::min(min_value, (*feature_list)[i][feature_index]);
      max_value = std::max(max_value, (*feature_list)[i][feature_index]);
    }
    assert(max_value != min_value);
    for (int i = 0; i < feature_list->size(); i++) {
      (*feature_list)[i][feature_index] =
          ((*feature_list)[i][feature_index] - min_value) / (max_value - min_value);
    }
  }
}

// Normalize Data : value = (value - mean) / variance
template <typename T>
void NormalizeByMeanAndVariance(T* feature_list) {
  assert(feature_list != nullptr);
  assert(feature_list->size() > 0);
  for (int feature_index = 0; feature_index < (*feature_list)[0].size(); feature_index++) {
    T sum = 0;
    T sqr_sum = 0;
    for (int i = 0; i < feature_list->size(); i++) {
      sum += (*feature_list)[i][feature_index];
      sqr_sum += Sqr((*feature_list)[i][feature_index]);
    }
    const T mean = sum / feature_list->size();
    const T variance = std::sqrt(sqr_sum - sum * mean);
    assert(variance != T::value_type());
    for (int i = 0; i < feature_list->size(); i++) {
      (*feature_list)[i][feature_index] = ((*feature_list)[i][feature_index] - mean) / variance;
    }
  }
}

}  // namespace math
}  // namespace utils
