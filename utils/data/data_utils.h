// Copyright @2020 mayf3

#pragma once

#include <fstream>
#include <random>

#include "utils/string/string_utils.h"

namespace utils {
namespace data {

template <typename FeatureListAndLabelList>
void ReadFeatureListAndLabelList(const char* filename,
                                 FeatureListAndLabelList* feature_list_and_label_list) {
  assert(feature_list_and_label_list != nullptr);
  typedef decltype(feature_list_and_label_list->feature_list) FeatureList;
  typedef decltype(feature_list_and_label_list->label_list) LabelList;
  typedef typename FeatureList::value_type Feature;
  typedef typename LabelList::value_type Label;
  typedef typename Feature::value_type FeatureType;
  FeatureList* feature_list = &feature_list_and_label_list->feature_list;
  LabelList* label_list = &feature_list_and_label_list->label_list;
  std::ifstream input(filename);
  std::string line;
  // Get feature and label
  while (getline(input, line)) {
    Feature feature;
    Label label;
    auto string_list = utils::string::Split(line);
    assert(string_list.size() > 1);
    for (int i = 0; i < string_list.size() - 1; i++) {
      feature.emplace_back(utils::string::StringToValue<FeatureType>(string_list[i]));
    }
    label = utils::string::StringToValue<Label>(string_list.back());
    if (feature_list->size()) {
      assert((*feature_list)[0].size() == feature.size());
    }
    feature_list->emplace_back(std::move(feature));
    label_list->emplace_back(label);
  }
}

// TODO(mayf3) Avoid redundant copying
// Split data to training(70%) and test(30%)
template <typename FeatureListAndLabelList>
void SplitIntoTrainingAndTesting(const FeatureListAndLabelList& feature_and_label_list,
                                 FeatureListAndLabelList* training_data,
                                 FeatureListAndLabelList* testing_data) {
  assert(training_data != nullptr);
  assert(testing_data != nullptr);
  constexpr unsigned int kRandomSeed = 20201220;
  constexpr double kRateOfNormalSplitDataMode = 0.7;
  const auto& feature_list = feature_and_label_list.feature_list;
  const auto& label_list = feature_and_label_list.label_list;
  std::vector<int> index(feature_list.size(), 0);
  for (int i = 0; i < index.size(); i++) {
    index[i] = i;
  }
  std::shuffle(index.begin(), index.end(), std::default_random_engine(kRandomSeed));

  const int end_of_training = static_cast<int>(feature_list.size() * kRateOfNormalSplitDataMode);
  for (int i = 0; i < end_of_training; i++) {
    training_data->feature_list.emplace_back(feature_list[index[i]]);
    training_data->label_list.emplace_back(label_list[index[i]]);
  }

  for (int i = end_of_training; i < feature_list.size(); i++) {
    testing_data->feature_list.emplace_back(feature_list[index[i]]);
    testing_data->label_list.emplace_back(label_list[index[i]]);
  }
}

}  // namespace data
}  // namespace utils
