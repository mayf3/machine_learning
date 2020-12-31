#include <cstdio>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include "algorithm/knn/knn_brute_force.h"
#include "algorithm/knn/knn_fast.h"
#include "utils/data/data_utils.h"
#include "utils/math/math_utils.h"
#include "utils/string/string_utils.h"

int main(int argc, char** argv) {
  // TODO(mayf3) Use google command line to parse argc and argv.
  if (argc < 2) {
    printf("Hint : %s input_file\n", argv[0]);
    return -1;
  }
  const char* input_filename = argv[1];

  algorithm::knn::KnnInterface::FeatureListAndLabelList feature_list_and_label_list;
  utils::data::ReadFeatureListAndLabelList(input_filename, &feature_list_and_label_list);

  // utils::math::NormalizeByMinMax(&feature_list_and_label_list.feature_list);
  utils::math::NormalizeByMeanAndVariance(&feature_list_and_label_list.feature_list);

  algorithm::knn::KnnInterface::FeatureListAndLabelList training_data;
  algorithm::knn::KnnInterface::FeatureListAndLabelList testing_data;

  utils::data::SplitIntoTrainingAndTesting(feature_list_and_label_list, &training_data,
                                           &testing_data);

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
  }
  std::cout << " Correct Rate: " << std::fixed << std::setprecision(2)
            << static_cast<double>(correct_times) / testing_data.feature_list.size() * 100.0
            << " % (" << correct_times << "/" << testing_data.feature_list.size() << ")"
            << std::endl;
  return 0;
}
