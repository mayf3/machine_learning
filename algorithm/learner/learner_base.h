// Copyright @2020 mayf3

#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "utils/common/define.h"

namespace algorithm {
namespace learner {

class LearnerBase {
 public:
  LearnerBase() = default;
  virtual ~LearnerBase() = default;

  // Normal Data :
  // - Feature : vecotr<double>
  // - Label : int
  using NormalFeature = std::vector<double>;
  using NormalFeatureList = std::vector<NormalFeature>;
  using NormalLabel = int;
  using NormalLabelList = std::vector<NormalLabel>;
  using NormalFeatureListAndLabelList =
      utils::common::FeatureListAndLabelList<NormalFeature, NormalLabel>;

  virtual const std::string Name() const = 0;

  // Predict normal feature data which include only vector of double value.
  virtual NormalLabel Predict(const NormalFeature& feature) const {
    std::cout << "Do not implement the Predict function in class " << Name() << std::endl;
    assert(false);
    return -1;
  }
};

}  // namespace learner
}  // namespace algorithm
