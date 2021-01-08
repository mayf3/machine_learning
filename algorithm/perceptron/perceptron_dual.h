// Copyright @2020 mayf3

#pragma once

#include <algorithm>
#include <vector>

#include "algorithm/learner/learner_base.h"
#include "algorithm/learner/learner_name.h"
#include "algorithm/learner/learner_options.h"

namespace algorithm {
namespace perceptron {

class PerceptronDual : public learner::LearnerBase {
 public:
  PerceptronDual(const learner::LearnerOptions& options);

  const std::string Name() const override { return kPerceptronDualName; }

  NormalLabel Predict(const NormalFeature& feature) const override;

  int num_iteration() const { return num_iteration_; }

 private:
  const int num_dim_;
  const double learning_rate_;
  std::vector<double> alpha_;
  double beta_ = 0.0;
  int num_iteration_ = 0;
};

}  // namespace perceptron
}  // namespace algorithm
