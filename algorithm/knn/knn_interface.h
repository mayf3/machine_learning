// Copyright @2020 mayf3

#pragma once

#include <algorithm>
#include <vector>

#include "algorithm/learner/learner_base.h"

namespace algorithm {
namespace knn {

class KnnInterface : public learner::LearnerBase {
 public:
   using LearnerBase::NormalLabel;
   using LearnerBase::NormalFeature;

  struct IndexAndSqrDistance {
    int index = -1;
    double sqr_distance = 0.0;
    IndexAndSqrDistance(int index, double sqr_distance)
        : index(index), sqr_distance(sqr_distance) {}
    bool operator<(const IndexAndSqrDistance& other) const {
      return this->sqr_distance < other.sqr_distance;
    }
  };

  KnnInterface(const NormalFeatureList& feature_list, const NormalLabelList& label_list, int dim)
      : LearnerBase(),
        feature_list_(feature_list),
        label_list_(label_list),
        point_size_(feature_list_.size()),
        point_dim_(dim) {
    assert(feature_list_.size() == label_list_.size());
    for (const NormalFeature& feature : feature_list) {
      assert(feature.size() == dim);
    }
  }

  ~KnnInterface() = default;

  virtual int Search(const NormalFeature& feature, int k, NormalLabelList* k_indices,
                     std::vector<double>* k_sqr_distances) const = 0;

  NormalLabel Predict(const NormalFeature& feature) const override;

 protected:
  // TODO(mayf3) move to utils;
  double SqrDistance(const NormalFeature& a, const NormalFeature& b) const;

  static constexpr int kDefaultK = 20;

  const NormalFeatureList feature_list_;
  const NormalLabelList label_list_;
  const int point_size_;
  const int point_dim_;
  int k_ = kDefaultK;
};

}  // namespace knn
}  // namespace algorithm
