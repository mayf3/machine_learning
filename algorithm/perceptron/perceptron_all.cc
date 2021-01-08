// Copyright @2020 mayf3

#include "algorithm/perceptron/perceptron_all.h"

#include "algorithm/learner/learner_factory.h"
#include "algorithm/perceptron/utils.h"
#include "utils/math/math_utils.h"

namespace algorithm {
namespace perceptron {

REGISTER_LEARNER(PerceptronAll, kPerceptronAllName);

PerceptronAll::PerceptronAll(const learner::LearnerOptions& options)
    : num_dim_(options.num_dim), learning_rate_(options.learning_rate) {
  weight_.resize(num_dim_);
  while (true) {
    bool all_correct = true;
    std::vector<double> w_delta(num_dim_, 0.0);
    double b_delta = 0.0;
    for (int i = 0; i < options.normal_feature_list.size(); i++) {
      const int internal_type = NormalTypeToInternalType(options.normal_label_list[i]);
      double value = bias_;
      for (int j = 0; j < num_dim_; j++) {
        value += weight_[j] * options.normal_feature_list[i][j];
      }
      if (value * internal_type > utils::math::kEpsilon) {
        continue;
      }
      const double factor = learning_rate_ * internal_type;
      for (int j = 0; j < num_dim_; j++) {
        w_delta[j] += factor * options.normal_feature_list[i][j];
      }
      b_delta += factor;
      all_correct = false;
    }
    if (all_correct) {
      break;
    }
    for (int i = 0; i < num_dim_; i++) {
      weight_[i] += w_delta[i];
    }
    bias_ += b_delta;
    num_iteration_++;
  }
}

PerceptronAll::NormalLabel PerceptronAll::Predict(const NormalFeature& feature) const {
  double value = bias_;
  for (int i = 0; i < num_dim_; i++) {
    value += weight_[i] * feature[i];
  }
  return InternalTypeToNormalType(PerceptronSign(value));
}

}  // namespace perceptron
}  // namespace algorithm
