#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>

#include "knn_brute_force.h"

constexpr char kDelimeter = ',';
constexpr double kEpsilon = 1e-6;

template <class OutStream>
void Split(const std::string& s, char delimeter, int delimeter_length, bool allow_empty,
           OutStream out) {
  assert(delimeter_length > 0);
  int start_pos = 0;
  int end_pos = 0;
  while (start_pos < s.length() && (end_pos = s.find(delimeter, start_pos)) != std::string::npos) {
    if (allow_empty || start_pos != end_pos) {
      *out = s.substr(start_pos, end_pos - start_pos);
    }
    start_pos = end_pos + delimeter_length;
  }
  if (allow_empty || start_pos != s.length()) {
    *out = s.substr(start_pos);
  }
}

std::vector<std::string> Split(const std::string& s, char delimiter, bool allow_empty) {
  std::vector<std::string> tokens;
  Split(s, delimiter, 1, allow_empty, std::back_inserter(tokens));
  return tokens;
}

int main(int argc, char** argv) {
  // TODO(mayf3) Use google command line to parse argc and argv.
  if (argc < 2) {
    printf("Hint : %s input_file", argv[0]);
    return -1;
  }
  const char* input_filename = argv[1];

  std::ifstream input(input_filename);
  std::string line;

  // TODO(mayf3) Move to a util.
  knn::KnnBruteForce::FeatureList feature_list;
  knn::KnnBruteForce::LabelList label_list;

  // Get feature and label
  while (getline(input, line)) {
    knn::KnnBruteForce::Feature feature;
    knn::KnnBruteForce::Label label;
    auto string_list = Split(line, kDelimeter, true);
    assert(string_list.size() > 1);
    for (int i = 0; i < string_list.size() - 1; i++) {
      feature.emplace_back(std::stod(string_list[i]));
    }
    label = std::stoi(string_list.back());
    if (feature_list.size()) {
      assert(feature_list[0].size() == feature.size());
    }
    feature_list.emplace_back(std::move(feature));
    label_list.emplace_back(label);
  }

  // Normalize Data : value = (value - min_value) / (max_value - min_value)
  /*
  for (int feature_index = 0; feature_index < feature_list[0].size(); feature_index++) {
    double min_value = feature_list[0][feature_index];
    double max_value = feature_list[0][feature_index];
    for (int i = 0; i < feature_list.size(); i++) {
      min_value = std::min(min_value, feature_list[i][feature_index]);
      max_value = std::max(max_value, feature_list[i][feature_index]);
    }
    if (max_value - min_value > kEpsilon) {
      for (int i = 0; i < feature_list.size(); i++) {
        feature_list[i][feature_index] =
            (feature_list[i][feature_index] - min_value) / (max_value - min_value);
      }
    }
  }
  */

  // Normalize Data : value = (value - mean) / variance
  /*
  for (int feature_index = 0; feature_index < feature_list[0].size(); feature_index++) {
    double sum = 0;
    double sqr_sum = 0;
    for (int i = 0; i < feature_list.size(); i++) {
      sum += feature_list[i][feature_index];
      sqr_sum += feature_list[i][feature_index] * feature_list[i][feature_index];
    }
    const double mean = sum / feature_list.size();
    const double variance = std::sqrt(sqr_sum - sum * mean);
    if (variance > kEpsilon) {
      for (int i = 0; i < feature_list.size(); i++) {
        feature_list[i][feature_index] = (feature_list[i][feature_index] - mean) / variance;
      }
    }
  }
  */

  // Split data to training(70%) and test(30%)
  std::vector<int> index(feature_list.size(), 0);
  for (int i = 0; i < index.size(); i++) {
    index[i] = i;
  }
  std::shuffle(index.begin(), index.end(), std::default_random_engine(20201220));

  constexpr double kTrainingDataRate = 0.7;

  knn::KnnBruteForce::FeatureList training_feature_list;
  knn::KnnBruteForce::LabelList training_label_list;

  const int end_of_training = static_cast<int>(feature_list.size() * kTrainingDataRate);
  for (int i = 0; i < end_of_training; i++) {
    training_feature_list.emplace_back(feature_list[index[i]]);
    training_label_list.emplace_back(label_list[index[i]]);
  }

  knn::KnnBruteForce::FeatureList test_feature_list;
  knn::KnnBruteForce::LabelList test_label_list;

  for (int i = end_of_training; i < feature_list.size(); i++) {
    test_feature_list.emplace_back(feature_list[index[i]]);
    test_label_list.emplace_back(label_list[index[i]]);
  }

  knn::KnnBruteForce knn_brute_force(training_feature_list, training_label_list,
                                     training_feature_list[0].size());

  constexpr int kParameterOfKnn = 20;
  int correct_times = 0;
  for (int i = 0; i < test_feature_list.size(); i++) {
    knn::KnnBruteForce::LabelList k_labels;
    knn_brute_force.Search(test_feature_list[i], kParameterOfKnn, &k_labels, nullptr);
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
    if (label_of_max_times == test_label_list[i]) {
      correct_times++;
    }
  }
  std::cout << " Correct Rate: " << std::fixed << std::setprecision(2)
            << static_cast<double>(correct_times) / test_feature_list.size() * 100.0 << " % ("
            << correct_times << "/" << test_feature_list.size() << ")" << std::endl;
  return 0;
}
