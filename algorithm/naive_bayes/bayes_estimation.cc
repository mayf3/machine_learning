// Copyright @2021 mayf3

#include "algorithm/naive_bayes/bayes_estimation.h"

#include "utils/math/math_utils.h"

namespace algorithm {
namespace naive_bayes {

constexpr double BayesEstimation::kLamda;

BayesEstimation::BayesEstimation(const NormalFeatureList& feature_list,
                                 const NormalLabelList& label_list, int num_dim, int num_class, double lamda)
    : learner::LearnerBase(), num_dim_(num_dim), num_class_(num_class), num_data_(feature_list.size()), lamda_(lamda) {
  condition_frequency_.resize(num_dim_);
  condition_probability_.resize(num_dim_);
  different_keys_.resize(num_dim_);
  for (int i = 0; i < feature_list.size(); i++) {
    type_frequency_[label_list[i]]++;
    for (int j = 0; j < feature_list[i].size(); j++) {
      condition_frequency_[j][std::make_pair(feature_list[i][j], label_list[i])]++;
      different_keys_[j].insert(feature_list[i][j]);
    }
  }

  for (const auto& pair : type_frequency_) {
    type_probability_[pair.first] = (static_cast<double>(pair.second) + lamda_) / (feature_list.size() + num_class_ * lamda_);
  }

  for (int i = 0; i < num_dim_; i++) {
    for (const auto& pair : condition_frequency_[i]) {
      condition_probability_[i][pair.first] =
          (static_cast<double>(pair.second) + lamda_) / (type_frequency_.at(pair.first.second) + different_keys_[i].size() * lamda_);
    }
  }
}

BayesEstimation::NormalLabel BayesEstimation::Predict(const NormalFeature& feature) const {
  BayesEstimation::NormalLabel result = -1;
  double max_possible = 0.0;
  for (int i = 0; i < num_class_; i++) {
    double possible = type_probability_.at(i);
    for (int j = 0; j < feature.size(); j++) {
      if (condition_probability_.at(j).count(std::make_pair(feature[j], i))) {
        possible *= condition_probability_.at(j).at(std::make_pair(feature[j], i));
      } else {
        possible *= lamda_ / (type_frequency_.at(i) + different_keys_.at(j).size() * lamda_);
      }
    }
      std::cout << std::fixed << i << " " << possible << std::endl;
    if (possible - max_possible > utils::math::kEpsilon) {
      max_possible = possible;
      result = i;
    }
  }
  return result;
}

}  // namespace naive_bayes
}  // namespace algorithm
