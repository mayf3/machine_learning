// Copyright @2020 mayf3

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace algorithm {
namespace learner {

class LearnerBase {
 public:
  template <typename Type>
  using Feature = std::vector<Type>;

  template <typename Feature>
  using FeatureList = std::vector<Feature>;

  template <typename Type>
  using Label = Type;

  template <typename Label>
  using LabelList = std::vector<Label>;

  template <typename Feature, typename Label>
  struct FeatureListAndLabelList {
    FeatureList<Feature> feature_list;
    LabelList<Label> label_list;
  };

  // Normal Data :
  // - Feature : vecotr<double>
  // - Label : int
  using NormalFeature = Feature<double>;
  using NormalFeatureList = FeatureList<NormalFeature>;
  using NormalLabel = Label<int>;
  using NormalLabelList = LabelList<NormalLabel>;
  using NormalFeatureListAndLabelList = FeatureListAndLabelList<NormalFeature, NormalLabel>;

  LearnerBase() = default;
  virtual ~LearnerBase() = default;

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
