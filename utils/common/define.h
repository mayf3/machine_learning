// Copyright @2020 mayf3

#pragma once

#include <vector>

namespace utils {
namespace common {

// TODO(mayf3) Move the defination to learner_base
template <typename T>
using Feature = std::vector<T>;

template <typename Feature>
using FeatureList = std::vector<Feature>;

template <typename Label>
using LabelList = std::vector<Label>;

template <typename Feature, typename Label>
struct FeatureListAndLabelList {
  FeatureList<Feature> feature_list;
  LabelList<Label> label_list;
};

}  // namespace common
}  // namespace utils
