// Copyright @2020 mayf3

#pragma once

#include <algorithm>
#include <vector>

#include "algorithm/knn/knn_interface.h"
#include "algorithm/learner/learner_name.h"

namespace algorithm {
namespace knn {

class KnnBruteForce : public KnnInterface {
 public:
  KnnBruteForce(const learner::LearnerOptions& options) 
      : KnnInterface(options) {}

  ~KnnBruteForce() = default;

  const std::string Name() const override { return kKnnBruteForceName; }

  int Search(const NormalFeature& feature, int k, NormalLabelList* k_indices,
             std::vector<double>* k_sqr_distances) const override;
};

}  // namespace knn
}  // namespace algorithm
