// Copyright @2020 mayf3

#include "algorithm/perceptron/perceptron_dual.h"

#include "algorithm/learner/learner_factory.h"
#include "algorithm/perceptron/utils.h"
#include "utils/math/math_utils.h"

namespace algorithm {
namespace perceptron {

REGISTER_LEARNER(PerceptronDual, kPerceptronDualName);

PerceptronDual::PerceptronDual(const learner::LearnerOptions& options)
    : num_dim_(options.num_dim), learning_rate_(options.learning_rate) {
  alpha_.resize(options.normal_feature_list.size());

  // Compute gram_matrix;
  std::vector<std::vector<double>> gram_matrix;
  gram_matrix.resize(options.normal_feature_list.size());
  for (int i = 0; i < options.normal_feature_list.size(); i++) {
    gram_matrix[i].resize(options.normal_feature_list.size());
    for (int j = 0; j < options.normal_feature_list.size(); j++) {
      for (int k = 0; k < num_dim_; k++) {
        gram_matrix[i][j] += options.normal_feature_list[i][k] * options.normal_feature_list[j][k];
      }
    }
  }

  while (true) {
    bool all_correct = true;
    for (int i = 0; i < options.normal_feature_list.size(); i++) {
      const int internal_type = NormalTypeToInternalType(options.normal_label_list[i]);
      double value = beta_;
      for (int j = 0; j < options.normal_feature_list.size(); j++) {
        value +=
            alpha_[j] * NormalTypeToInternalType(options.normal_label_list[j]) * gram_matrix[j][i];
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
    num_iteration_++;
  }
}

PerceptronDual::NormalLabel PerceptronDual::Predict(const NormalFeature& feature) const {
  double value = beta_;
  for (int i = 0; i < num_dim_; i++) {
    value += alpha_[i] * feature[i];
  }
  return InternalTypeToNormalType(PerceptronSign(value));
}

}  // namespace perceptron
}  // namespace algorithm
