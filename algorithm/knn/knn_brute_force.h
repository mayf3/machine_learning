// Copyright @2020 mayf3

#pragma once

#include <algorithm>
#include <vector>

#include "algorithm/knn/knn_interface.h"

namespace algorithm {
namespace knn {

class KnnBruteForce : public KnnInterface {
 public:
  KnnBruteForce(const NormalFeatureList& feature_list, const NormalLabelList& label_list, int dim)
      : KnnInterface(feature_list, label_list, dim) {}

  ~KnnBruteForce() = default;

  const std::string Name() const override { return "KnnBruteForce"; }

  int Search(const NormalFeature& feature, int k, NormalLabelList* k_indices,
             std::vector<double>* k_sqr_distances) const override;
};

}  // namespace knn
}  // namespace algorithm
