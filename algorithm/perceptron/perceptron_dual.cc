// Copyright @2020 mayf3

#include "algorithm/perceptron/perceptron_dual.h"

#include "algorithm/perceptron/utils.h"
#include "utils/math/math_utils.h"

namespace algorithm {
namespace perceptron {

constexpr double PerceptronDual::kLearningRate;

PerceptronDual::PerceptronDual(const NormalFeatureList& feature_list,
                               const NormalLabelList& label_list, int dim, double learning_rate)
    : learner::LearnerBase(), dim_(dim), learning_rate_(learning_rate) {
  alpha_.resize(feature_list.size());

  // Compute gram_matrix;
  std::vector<std::vector<double>> gram_matrix;
  gram_matrix.resize(feature_list.size());
  for (int i = 0; i < feature_list.size(); i++) {
    gram_matrix[i].resize(feature_list.size());
    for (int j = 0; j < feature_list.size(); j++) {
      for (int k = 0; k < dim_; k++) {
        gram_matrix[i][j] += feature_list[i][k] * feature_list[j][k];
      }
    }
  }

  while (true) {
    bool all_correct = true;
    for (int i = 0; i < feature_list.size(); i++) {
      const int internal_type = NormalTypeToInternalType(label_list[i]);
      double value = beta_;
      for (int j = 0; j < feature_list.size(); j++) {
        value += alpha_[j] * NormalTypeToInternalType(label_list[j]) * gram_matrix[j][i];
      }
      if (value * internal_type > utils::math::kEpsilon) {
        continue;
      }
      alpha_[i] += learning_rate_;
      beta_ += learning_rate_ * internal_type;
      all_correct = false;
      break;
    }
    if (all_correct) {
      break;
    }
  }
}

PerceptronDual::NormalLabel PerceptronDual::Predict(const NormalFeature& feature) const {
  double value = beta_;
  for (int i = 0; i < dim_; i++) {
    value += alpha_[i] * feature[i];
  }
  return InternalTypeToNormalType(PerceptronSign(value));
}

}  // namespace perceptron
}  // namespace algorithm
