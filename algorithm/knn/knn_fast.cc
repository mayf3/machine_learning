// Copyright @2020 mayf3

#include "algorithm/knn/knn_fast.h"

#include "utils/math/math_utils.h"

namespace algorithm {
namespace knn {

namespace {

using Feature = KnnInterface::Feature;
using IndexAndSqrDistance = KnnInterface::IndexAndSqrDistance;
using KnnHeap = KnnFast::KnnHeap;

}  // namespace

KnnFast::KnnFast(const FeatureList& feature_list, const LabelList& label_list, int dim)
    : KnnInterface(feature_list, label_list, dim) {
  index_.reserve(point_size_);
  for (int i = 0; i < point_size_; i++) {
    index_.emplace_back(i);
  }
  Build(0, point_size_, 0);
}

void KnnFast::Build(int left, int right, int dim) {
  if (right - left <= 1) {
    return;
  }
  const int mid = (left + right) / 2;
  std::nth_element(
      index_.begin() + left, index_.begin() + mid, index_.begin() + right,
      [this, &dim](int a, int b) { return feature_list_[a][dim] < feature_list_[b][dim]; });
  const int next_dim = (dim == point_dim_ - 1) ? 0 : dim + 1;
  Build(left, mid, next_dim);
  Build(mid + 1, right, next_dim);
}

void KnnFast::SearchInternal(const Feature& feature, int left, int right, int dim, int k,
                             KnnHeap* heap) const {
  if (right - left < 1) {
    return;
  }
  const int mid = (left + right) / 2;
  heap->push(IndexAndSqrDistance{index_[mid], SqrDistance(feature, feature_list_[index_[mid]])});
  if (heap->size() > k) heap->pop();
  const double diff = feature[dim] - feature_list_[index_[mid]][dim];
  const int next_dim = (dim == point_dim_ - 1) ? 0 : dim + 1;
  if (diff < 0) {
    SearchInternal(feature, left, mid, next_dim, k, heap);
    if (heap->top().sqr_distance > utils::math::Sqr(diff)) {
      SearchInternal(feature, mid + 1, right, next_dim, k, heap);
    }
  } else {
    SearchInternal(feature, mid + 1, right, next_dim, k, heap);
    if (heap->top().sqr_distance > utils::math::Sqr(diff)) {
      SearchInternal(feature, left, mid, next_dim, k, heap);
    }
  }
}

int KnnFast::Search(const Feature& feature, int k, LabelList* k_labels,
                    std::vector<double>* k_sqr_distances) const {
  assert(k > 0);
  assert(k_labels != nullptr);
  assert(point_dim_ == feature.size());

  KnnHeap heap;
  SearchInternal(feature, 0, point_size_, 0, k, &heap);

  k_labels->clear();
  k_labels->reserve(k);
  if (k_sqr_distances) {
    k_sqr_distances->clear();
    k_sqr_distances->reserve(k);
  }
  while (heap.size()) {
    const IndexAndSqrDistance& index_and_sqr_distance = heap.top();
    k_labels->emplace_back(label_list_[index_and_sqr_distance.index]);
    if (k_sqr_distances) {
      k_sqr_distances->emplace_back(index_and_sqr_distance.sqr_distance);
    }
    heap.pop();
  }
  return k_labels->size();
}

}  // namespace knn
}  // namespace algorithm
