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

void GenerateDataForPerceptron(int size, int dim,
                               learner::LearnerBase::NormalFeatureList* feature_list,
                               learner::LearnerBase::NormalLabelList* label_list) {
  assert(feature_list != nullptr);
  assert(label_list != nullptr);
  for (int i = 0; i < size; i++) {
    learner::LearnerBase::NormalFeature feature;
    for (int j = 0; j < dim; j++) {
      feature.emplace_back(rand() % 1000);
    }
    feature_list->emplace_back(std::move(feature));
  }

  if (size == 1) {
    label_list->emplace_back(kNormalPositiveType);
    return;
  } else if (size == 2) {
    label_list->emplace_back(kNormalPositiveType);
    label_list->emplace_back(kNormalNegativeType);
    return;
  }

  learner::LearnerBase::NormalFeature diff, center;
  for (int i = 0; i < dim; i++) {
    diff.emplace_back((*feature_list)[0][i] - (*feature_list)[1][i]);
    center.emplace_back(((*feature_list)[0][i] + (*feature_list)[1][i]) / 2.0);
  }

  for (int i = 0; i < size; i++) {
    double value = 0.0;
    for (int j = 0; j < dim; j++) {
      value += ((*feature_list)[i][j] - center[j]) * diff[j];
    }
    if (value < -utils::math::kEpsilon) {
      label_list->emplace_back(kNormalNegativeType);
    } else {
      label_list->emplace_back(kNormalPositiveType);
    }
  }
}

}  // namespace perceptron
}  // namespace algorithm
