// Copyright @2020 mayf3

#include "algorithm/knn/knn_interface.h"

#include <unordered_map>

#include "utils/math/math_utils.h"

namespace algorithm {
namespace knn {

using NormalLabel = KnnInterface::NormalLabel;
using NormalFeature = KnnInterface::NormalFeature;

double KnnInterface::SqrDistance(const NormalFeature& a, const NormalFeature& b) const {
  double sqr_distance = 0;
  for (int i = 0; i < a.size(); i++) {
    sqr_distance += utils::math::Sqr(a[i] - b[i]);
  }
  return sqr_distance;
}

NormalLabel KnnInterface::Predict(const NormalFeature& feature) const {
  NormalLabelList k_labels;
  Search(feature, k_, &k_labels, nullptr);
  std::unordered_map<int, int> label_count;
  for (const auto& label : k_labels) {
    label_count[label]++;
  }
  int max_times = 0;
  int label_of_max_times = -1;
  for (const auto& pair : label_count) {
    if (pair.second > max_times) {
      max_times = pair.second;
      label_of_max_times = pair.first;
    }
  }
  return label_of_max_times;
}

}  // namespace knn
}  // namespace algorithm
