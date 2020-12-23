// Copyright @2020 mayf3

#pragma once

#include <algorithm>
#include <vector>

#include "algorithm/knn/knn_interface.h"
#include "utils/common/define.h"

namespace algorithm {
namespace knn {

class KnnBruteForce : public KnnInterface {
 public:
  KnnBruteForce(const FeatureList& feature_list, const LabelList& label_list, int dim)
      : KnnInterface(feature_list, label_list, dim) {}

  ~KnnBruteForce() = default;

  int Search(const Feature& feature, int k, LabelList* k_indices,
             std::vector<double>* k_sqr_distances) const override;
};

}  // namespace knn
}  // namespace algorithm
