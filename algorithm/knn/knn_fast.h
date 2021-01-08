// Copyright @2020 mayf3

#pragma once

#include <algorithm>
#include <queue>
#include <vector>

#include "algorithm/knn/knn_interface.h"
#include "algorithm/learner/learner_name.h"

namespace algorithm {
namespace knn {

class KnnFast : public KnnInterface {
 public:
  using KnnHeap = std::priority_queue<IndexAndSqrDistance>;
  KnnFast(const learner::LearnerOptions& options);

  ~KnnFast() = default;

  const std::string Name() const override { return kKnnFastName; }

  int Search(const NormalFeature& feature, int k, NormalLabelList* k_indices,
             std::vector<double>* k_sqr_distances) const override;

 private:
  void Build(int left, int right, int dim);
  void SearchInternal(const NormalFeature& feature, int left, int right, int dim, int k,
                      KnnHeap* heap) const;

  std::vector<int> index_;
};

}  // namespace knn
}  // namespace algorithm
