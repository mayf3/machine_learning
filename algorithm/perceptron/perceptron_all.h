// Copyright @2020 mayf3

#pragma once

#include <algorithm>
#include <vector>

#include "algorithm/learner/learner_base.h"
#include "algorithm/learner/learner_name.h"
#include "algorithm/learner/learner_options.h"

namespace algorithm {
namespace perceptron {

class PerceptronAll : public learner::LearnerBase {
 public:
  PerceptronAll(const learner::LearnerOptions& options);

  const std::string Name() const override { return kPerceptronAllName; }

  NormalLabel Predict(const NormalFeature& feature) const override;

  int num_iteration() const { return num_iteration_; }

 private:
  const int num_dim_;
  const double learning_rate_;
  std::vector<double> weight_;
  double bias_ = 0.0;
  int num_iteration_ = 0;
};

}  // namespace perceptron
}  // namespace algorithm
