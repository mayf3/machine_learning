#include <cstdio>
#include <iostream>
#include <unordered_map>
#include <iomanip>

#include "algorithm/knn/knn_brute_force.h"
#include "algorithm/knn/knn_fast.h"
#include "utils/string/string_utils.h"
#include "utils/data/data_utils.h"
#include "utils/data/mnist_data_parser.h"

algorithm::knn::KnnInterface::FeatureList Parse(const std::vector<utils::data::MnistDataParser::Matrix>& image) {
  algorithm::knn::KnnInterface::FeatureList feature_list;
  for (auto matrix : image) {
    algorithm::knn::KnnInterface::Feature feature;
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

  algorithm::knn::KnnInterface::FeatureListAndLabelList training_data;
  training_data.feature_list = Parse(parser.training_image());
  training_data.label_list = parser.training_label();

  algorithm::knn::KnnInterface::FeatureListAndLabelList testing_data;
  testing_data.feature_list = Parse(parser.testing_image());
  testing_data.label_list = parser.testing_label();

  assert(training_data.feature_list.size() > 0);

  // algorithm::knn::KnnBruteForce knn_instance(
  //     training_data.feature_list, training_data.label_list, training_data.feature_list[0].size());
  algorithm::knn::KnnFast knn_instance(
      training_data.feature_list, training_data.label_list, training_data.feature_list[0].size());

  constexpr int kParameterOfKnn = 20;
  int correct_times = 0;
  for (int i = 0; i < testing_data.feature_list.size(); i++) {
    algorithm::knn::KnnInterface::LabelList k_labels;
    knn_instance.Search(testing_data.feature_list[i], kParameterOfKnn, &k_labels, nullptr);
    std::unordered_map<int, int> label_count;
    for (const auto& label : k_labels) {
      label_count[label]++;
    }
    int max_times = 0;
    int label_of_max_times = -1;
    for (const auto& pair : label_count) {
      if (pair.second > max_times) {
        max_times = pair.second;
        label_of_max_times = pair.first;
      }
    }
    if (label_of_max_times == testing_data.label_list[i]) {
      correct_times++;
    }
    if (i % 100 == 0) {
      std::cout << i << " "
            << " Correct Rate: " << std::fixed << std::setprecision(2)
            << static_cast<double>(correct_times) / i * 100.0 << " % ("
            << correct_times << "/" << i << ")" << std::endl;
    }
  }
  std::cout << " Correct Rate: " << std::fixed << std::setprecision(2)
            << static_cast<double>(correct_times) / testing_data.feature_list.size() * 100.0 << " % ("
            << correct_times << "/" << testing_data.feature_list.size() << ")" << std::endl;
  return 0;
}
