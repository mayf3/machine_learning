// Copyright @2020 mayf3

#include "algorithm/knn/knn_interface.h"

#include "utils/math/math_utils.h"

namespace algorithm {
namespace knn {

double KnnInterface::SqrDistance(const Feature& a, const Feature& b) const {
  double sqr_distance = 0;
  for (int i = 0; i < a.size(); i++) {
    sqr_distance += utils::math::Sqr(a[i] - b[i]);
  }
  return sqr_distance;
}

}  // namespace knn
}  // namespace algorithm
