// Copyright @2020 mayf3

#include "algorithm/perceptron/perceptron.h"

#include "algorithm/perceptron/utils.h"
#include "utils/math/math_utils.h"

namespace algorithm {
namespace perceptron {

constexpr double Perceptron::kLearningRate;

Perceptron::Perceptron(const NormalFeatureList& feature_list, const NormalLabelList& label_list,
                       int dim, double learning_rate)
    : learner::LearnerBase(), dim_(dim), learning_rate_(learning_rate) {
  weight_.resize(dim_);
  while (true) {
    bool all_correct = true;
    for (int i = 0; i < feature_list.size(); i++) {
      const int internal_type = NormalTypeToInternalType(label_list[i]);
      double value = bias_;
      for (int j = 0; j < dim_; j++) {
        value += weight_[j] * feature_list[i][j];
      }
      if (value * internal_type > utils::math::kEpsilon) {
        continue;
      }
      const double factor = learning_rate_ * internal_type;
      for (int j = 0; j < dim_; j++) {
        weight_[j] += factor * feature_list[i][j];
      }
      bias_ += factor;
      all_correct = false;
      break;
    }
    if (all_correct) {
      break;
    }
  }
}

Perceptron::NormalLabel Perceptron::Predict(const NormalFeature& feature) const {
  double value = bias_;
  for (int i = 0; i < dim_; i++) {
    value += weight_[i] * feature[i];
  }
  return InternalTypeToNormalType(PerceptronSign(value));
}

}  // namespace perceptron
}  // namespace algorithm
