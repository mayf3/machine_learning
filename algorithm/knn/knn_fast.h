// Copyright @2020 mayf3

#pragma once

#include <algorithm>
#include <queue>
#include <vector>

#include "algorithm/knn/knn_interface.h"
#include "utils/common/define.h"

namespace algorithm {
namespace knn {

class KnnFast : public KnnInterface {
 public:
  using KnnHeap = std::priority_queue<IndexAndSqrDistance>;
  KnnFast(const FeatureList& feature_list, const LabelList& label_list, int dim);

  ~KnnFast() = default;

  int Search(const Feature& feature, int k, LabelList* k_indices,
             std::vector<double>* k_sqr_distances) const override;

 private:
  void Build(int left, int right, int dim);
  void SearchInternal(const Feature& feature, int left, int right, int dim, int k,
                      KnnHeap* heap) const;

  std::vector<int> index_;
};

}  // namespace knn
}  // namespace algorithm
