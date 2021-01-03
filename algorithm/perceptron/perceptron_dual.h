// Copyright @2020 mayf3

#pragma once

#include <algorithm>
#include <vector>

#include "algorithm/learner/learner_base.h"

namespace algorithm {
namespace perceptron {

class PerceptronDual : public learner::LearnerBase {
 public:
  PerceptronDual(const NormalFeatureList& feature_list, const NormalLabelList& label_list, int dim,
             double learning_rate = kLearningRate);

  const std::string Name() const override { return "PerceptronDual"; }

  NormalLabel Predict(const NormalFeature& feature) const override;

  int num_iteration() const { return num_iteration_; }

 private:
  const int dim_;
  const double learning_rate_;
  static constexpr double kLearningRate = 1.0;
  std::vector<double> alpha_;
  double beta_ = 0.0;
  int num_iteration_ = 0;
};

}  // namespace perceptron
}  // namespace algorithm
