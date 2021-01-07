// Copyright @2021 mayf3

#pragma once

#include "algorithm/learner/learner_base.h"

namespace std {
template <>
class hash<std::pair<double, algorithm::learner::LearnerBase::NormalLabel>> {
 public:
  size_t operator()(const std::pair<double, algorithm::learner::LearnerBase::NormalLabel>& data) const {
    return hash<double>()(data.first) ^ hash<algorithm::learner::LearnerBase::NormalLabel>()(data.second);
  }
};
};  // namespace std
