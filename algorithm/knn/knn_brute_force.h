// Copyright @2020 mayf3

#pragma once

#include <algorithm>
#include <vector>

#include "utils/common/define.h"

namespace knn {

class KnnBruteForce {
 public:
  using Feature = utils::common::Feature<double>;
  using FeatureList = std::vector<Feature>;
  using Label = int;
  using LabelList = utils::common::LabelList<Label>;

  KnnBruteForce(const FeatureList& feature_list, const LabelList& label_list, int dim)
      : feature_list_(feature_list),
        label_list_(label_list),
        point_size_(feature_list_.size()),
        point_dim_(dim) {
    assert(feature_list_.size() == label_list_.size());
    for (const Feature& feature : feature_list) {
      assert(feature.size() == dim);
    }
  }

  ~KnnBruteForce() = default;

  int Search(const Feature& feature, int k, LabelList* k_indices,
             std::vector<double>* k_sqr_distances) const;

 private:
  const FeatureList feature_list_;
  const LabelList label_list_;
  const int point_size_;
  const int point_dim_;
};

}  // namespace knn
