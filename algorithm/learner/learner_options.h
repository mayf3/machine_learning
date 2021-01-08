// Copyright @2021 mayf3

#pragma once

#include "algorithm/learner/learner_base.h"

namespace algorithm {
namespace learner {

struct LearnerOptions {
  LearnerBase::NormalFeatureList normal_feature_list;
  LearnerBase::NormalLabelList normal_label_list;
  // Dim of feature
  int num_dim = 0;
  // Class of label
  int num_class = 0;
  // Learning rate
  double learning_rate = 0.0;

  // Only for BayesEstimation
  double baye_estimation_lamda = 1.0;
};

}  // namespace learner
}  // namespace algorithm
