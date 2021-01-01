// Copyright @2020 mayf3

#include "algorithm/perceptron/utils.h"

#include "utils/math/math_utils.h"

namespace algorithm {
namespace perceptron {

int PerceptronSign(double x) { return x < -utils::math::kEpsilon ? -1 : 1; }

int InternalTypeToNormalType(int internal_type) {
  return internal_type == kInternalNegativeType ? kNormalNegativeType : kNormalPositiveType;
}

int NormalTypeToInternalType(int normal_type) {
  return normal_type == kNormalNegativeType ? kInternalNegativeType : kInternalPositiveType;
}

}  // namespace perceptron
}  // namespace algorithm
