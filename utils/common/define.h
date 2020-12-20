// Copyright @2020 mayf3

#pragma once

#include <vector>

namespace utils {
namespace common {

template <typename T>
using Feature = std::vector<T>;

template <typename Feature>
using FeatureList = std::vector<Feature>;

template <typename Label>
using LabelList = std::vector<Label>;

}  // namespace common
}  // namespace utils
