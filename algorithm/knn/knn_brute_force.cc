// Copyright @2020 mayf3
#include "algorithm/knn/knn_brute_force.h"

    namespace algorithm {
  namespace knn {

  namespace {

  using NormalFeature = KnnInterface::NormalFeature;
  using IndexAndSqrDistance = KnnInterface::IndexAndSqrDistance;

  }  // namespace

  int KnnBruteForce::Search(const NormalFeature& feature, int k, NormalLabelList* k_labels,
                            std::vector<double>* k_sqr_distances) const {
    assert(k > 0);
    assert(k_labels != nullptr);
    assert(point_dim_ == feature.size());

    std::vector<IndexAndSqrDistance> index_and_sqr_distance;
    index_and_sqr_distance.reserve(point_size_);
    for (int i = 0; i < point_size_; ++i) {
      index_and_sqr_distance.emplace_back(i, SqrDistance(feature_list_[i], feature));
    }

    k = std::min(k, point_size_);
    std::nth_element(index_and_sqr_distance.begin(), index_and_sqr_distance.begin() + k,
                     index_and_sqr_distance.end());

    k_labels->clear();
    k_labels->reserve(k);
    if (k_sqr_distances) {
      k_sqr_distances->clear();
      k_sqr_distances->reserve(k);
    }
    for (int i = 0; i < k; ++i) {
      k_labels->emplace_back(label_list_[index_and_sqr_distance[i].index]);
      if (k_sqr_distances) {
        k_sqr_distances->emplace_back(index_and_sqr_distance[i].sqr_distance);
      }
    }
    return k_labels->size();
  }

  }  // namespace knn
}  // namespace algorithm
