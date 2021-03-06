// Copyright @2021 mayf3

#include "algorithm/naive_bayes/naive_bayes.h"

#include "algorithm/learner/learner_factory.h"
#include "utils/math/math_utils.h"

namespace algorithm {
namespace naive_bayes {

REGISTER_LEARNER(NaiveBayes, kNaiveBayesName);

NaiveBayes::NaiveBayes(const learner::LearnerOptions& options)
    : num_dim_(options.num_dim),
      num_class_(options.num_class),
      num_data_(options.normal_feature_list.size()) {
  const auto& feature_list = options.normal_feature_list;
  const auto& label_list = options.normal_label_list;
  condition_frequency_.resize(num_dim_);
  condition_probability_.resize(num_dim_);
  for (int i = 0; i < feature_list.size(); i++) {
    type_frequency_[label_list[i]]++;
    for (int j = 0; j < feature_list[i].size(); j++) {
      condition_frequency_[j][std::make_pair(feature_list[i][j], label_list[i])]++;
    }
  }

  for (const auto& pair : type_frequency_) {
    type_probability_[pair.first] = static_cast<double>(pair.second) / feature_list.size();
  }

  for (int i = 0; i < num_dim_; i++) {
    for (const auto& pair : condition_frequency_[i]) {
      condition_probability_[i][pair.first] =
          static_cast<double>(pair.second) / type_frequency_.at(pair.first.second);
    }
  }
}

NaiveBayes::NormalLabel NaiveBayes::Predict(const NormalFeature& feature) const {
  NaiveBayes::NormalLabel result = -1;
  double max_possible = 0.0;
  for (int i = 0; i < num_class_; i++) {
    double possible = type_probability_.at(i);
    for (int j = 0; j < feature.size(); j++) {
      if (condition_probability_[j].count(std::make_pair(feature[j], i))) {
        possible *= condition_probability_[j].at(std::make_pair(feature[j], i));
      } else {
        possible = 0.0;
      }
    }
    if (possible - max_possible > utils::math::kEpsilon) {
      max_possible = possible;
      result = i;
    }
  }
  return result;
}

}  // namespace naive_bayes
}  // namespace algorithm
