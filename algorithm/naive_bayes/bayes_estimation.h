// Copyright @2021 mayf3

#pragma once

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "algorithm/learner/learner_base.h"
#include "algorithm/naive_bayes/utils.h"

namespace algorithm {
namespace naive_bayes {

class BayesEstimation : public learner::LearnerBase {
 public:
  BayesEstimation(const NormalFeatureList& feature_list, const NormalLabelList& label_list,
                  int num_dim, int num_class, double lamda = kLamda);

  const std::string Name() const override { return "BayesEstimation"; }

  NormalLabel Predict(const NormalFeature& feature) const override;

 private:
  const int num_dim_;
  const int num_class_;
  const int num_data_;
  const double lamda_;
  static constexpr double kLamda = 1.0;
  // key : y, value : frequency
  std::unordered_map<NormalLabel, int> type_frequency_;
  // key : y, value : probability
  std::unordered_map<NormalLabel, double> type_probability_;

  // index : dim, key : <x, y>, value : frequency
  std::vector<std::unordered_map<std::pair<double, NormalLabel>, int>> condition_frequency_;
  // index : dim, key : <x, y>, value : probability
  std::vector<std::unordered_map<std::pair<double, NormalLabel>, double>> condition_probability_;

  // index : dim, key : x
  std::vector<std::unordered_set<double>> different_keys_;
};

}  // namespace naive_bayes
}  // namespace algorithm
