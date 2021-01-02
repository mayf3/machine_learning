// Copyright @2020 mayf3

#pragma once

#include <algorithm>
#include <queue>
#include <vector>

#include "algorithm/knn/knn_interface.h"

namespace algorithm {
namespace knn {

class KnnFast : public KnnInterface {
 public:
  using KnnHeap = std::priority_queue<IndexAndSqrDistance>;
  KnnFast(const NormalFeatureList& feature_list, const NormalLabelList& label_list, int dim);

  ~KnnFast() = default;

  const std::string Name() const override { return "KnnFast"; }

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
