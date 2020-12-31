#include <cstdio>
#include <iostream>
#include <unordered_map>
#include <iomanip>
#include <memory>
#include <utility>

#include "algorithm/knn/knn_brute_force.h"
#include "algorithm/knn/knn_fast.h"
#include "utils/string/string_utils.h"
#include "utils/data/data_utils.h"
#include "utils/data/mnist_data_parser.h"
#include "utils/time/stop_watch.h"

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

  constexpr unsigned int kRandomSeed = 20201220;

  algorithm::knn::KnnInterface::FeatureListAndLabelList training_data;
  training_data.feature_list = Parse(parser.training_image());
  training_data.label_list = parser.training_label();

  algorithm::knn::KnnInterface::FeatureListAndLabelList testing_data;
  testing_data.feature_list = Parse(parser.testing_image());
  testing_data.label_list = parser.testing_label();

  constexpr int kTestCase = 20;

  testing_data.feature_list.resize(kTestCase);
  testing_data.label_list.resize(kTestCase);

  algorithm::knn::KnnInterface::FeatureListAndLabelList basic_training_data = training_data;

  std::vector<int> index;
  for (int i = 0; i < basic_training_data.feature_list.size(); i++) {
    index.emplace_back(i);
  }

  constexpr int kNumberCase = 30;
  constexpr int kNumberSample = 10000;
  constexpr int kMaxK = 1000;

  std::vector<algorithm::knn::KnnInterface::FeatureListAndLabelList> training_data_list(kNumberCase);
  std::vector<algorithm::knn::KnnBruteForce> knn_instance;
  for (int case_index = 0; case_index < kNumberCase; case_index++) {
    std::shuffle(index.begin(), index.end(), std::default_random_engine(kRandomSeed + case_index));
    for (int i = 0; i < kNumberSample; i++) {
      training_data_list[case_index].feature_list.emplace_back(basic_training_data.feature_list[index[i]]);
      training_data_list[case_index].label_list.emplace_back(basic_training_data.label_list[index[i]]);
    }
    knn_instance.emplace_back(training_data_list[case_index].feature_list, training_data_list[case_index].label_list, training_data_list[case_index].feature_list[0].size());
    std::cout << case_index << std::endl;
  }

  for (int k = 1; k <= kMaxK; k++) {
    std::vector<double> sum(testing_data.feature_list.size(), 0);
    std::vector<double> sqr_sum(testing_data.feature_list.size(), 0);
    for (int case_index = 0; case_index < kNumberCase; case_index++) {
      int correct_times = 0;
      assert(testing_data.feature_list.size() == kTestCase);
      for (int i = 0; i < testing_data.feature_list.size(); i++) {
        algorithm::knn::KnnInterface::LabelList k_labels;
        knn_instance[case_index].Search(testing_data.feature_list[i], k, &k_labels, nullptr);
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
        sum[i] += label_of_max_times;
        sqr_sum[i] += label_of_max_times * label_of_max_times;
        // result.emplace_back(label_of_max_times);
        // std::cout << k << " " << label_of_max_times << " " << testing_data.label_list[i] << std::endl;
      }
    }
    double variance_sum = 0;
    double bias_sum = 0;
    for (int i = 0; i < sum.size(); i++) {
      const double mean = sum[i] / kNumberCase;
      const double variance = (sqr_sum[i] - sum[i] * mean) / kNumberCase;
      const double bias = (mean - testing_data.label_list[i]) * (mean - testing_data.label_list[i]);
      variance_sum += variance;
      bias_sum += bias;
    }
    std::cout << k << " " << std::fixed << variance_sum / sum.size() << " " << bias_sum / sum.size() << std::endl;

    // std::cout << k << " Correct Rate: " << std::fixed << std::setprecision(2)
    //   << static_cast<double>(correct_times) / testing_data.feature_list.size() * 100.0 << " % ("
    //   << correct_times << "/" << testing_data.feature_list.size() << ")" << std::endl;
  }
  return 0;
}
