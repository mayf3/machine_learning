#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>

#include "algorithm/knn/knn_brute_force.h"
#include "algorithm/knn/knn_fast.h"
#include "algorithm/learner/learner_factory.h"
#include "utils/data/data_utils.h"
#include "utils/data/mnist_data_parser.h"
#include "utils/string/string_utils.h"
#include "utils/time/stop_watch.h"

using NormalFeature = algorithm::knn::KnnInterface::NormalFeature;
using NormalFeatureList = algorithm::knn::KnnInterface::NormalFeatureList;
using NormalFeatureListAndLabelList = algorithm::knn::KnnInterface::NormalFeatureListAndLabelList;
using LearnerOptions = algorithm::learner::LearnerOptions;
using LearnerBase = algorithm::learner::LearnerBase;
using LearnerFactory = algorithm::learner::LearnerFactory;

NormalFeatureList Parse(const std::vector<utils::data::MnistDataParser::Matrix>& image) {
  NormalFeatureList feature_list;
  for (auto matrix : image) {
    NormalFeature feature;
    feature.reserve(matrix.size() * matrix[0].size());
    for (int i = 0; i < matrix.size(); i++) {
      for (int j = 0; j < matrix[i].size(); j++) {
        feature.emplace_back(matrix[i][j]);
      }
    }
    feature_list.emplace_back(std::move(feature));
  }
  return feature_list;
}

int main(int argc, char** argv) {
  utils::data::MnistDataParser parser(
      "./data/multi_class_classification/mnist/train-images-idx3-ubyte",
      "./data/multi_class_classification/mnist/train-labels-idx1-ubyte",
      "./data/multi_class_classification/mnist/t10k-images-idx3-ubyte",
      "./data/multi_class_classification/mnist/t10k-labels-idx1-ubyte");

  NormalFeatureListAndLabelList training_data;
  training_data.feature_list = Parse(parser.training_image());
  training_data.label_list = parser.training_label();

  NormalFeatureListAndLabelList testing_data;
  testing_data.feature_list = Parse(parser.testing_image());
  testing_data.label_list = parser.testing_label();

  assert(training_data.feature_list.size() > 0);

  LearnerOptions learner_options;
  learner_options.normal_feature_list = training_data.feature_list;
  learner_options.normal_label_list = training_data.label_list;
  learner_options.num_dim = learner_options.normal_feature_list[0].size();
  const std::string learner_name = kKnnBruteForceName;
  // const std::string learner_name = kKnnFastName;
  std::unique_ptr<LearnerBase> learner =
      LearnerFactory::GetInstance()->Create(learner_name, learner_options);

  int correct_times = 0;
  std::unique_ptr<utils::time::StopWatch> stop_watch(new utils::time::StopWatch("mnist"));
  for (int i = 0; i < testing_data.feature_list.size(); i++) {
    if (learner->Predict(testing_data.feature_list[i]) == testing_data.label_list[i]) {
      correct_times++;
    }
    if (i % 10 == 0) {
      std::cout << i << " "
                << " Correct Rate: " << std::fixed << std::setprecision(2)
                << static_cast<double>(correct_times) / (i + 1) * 100.0 << " % (" << correct_times
                << "/" << (i + 1) << ")" << std::endl;
      stop_watch.reset(new utils::time::StopWatch("mnist"));
    }
  }
  std::cout << " Correct Rate: " << std::fixed << std::setprecision(2)
            << static_cast<double>(correct_times) / testing_data.feature_list.size() * 100.0
            << " % (" << correct_times << "/" << testing_data.feature_list.size() << ")"
            << std::endl;
  return 0;
}
