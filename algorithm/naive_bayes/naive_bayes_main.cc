// Copyright @2021 mayf3

#include "algorithm/naive_bayes/naive_bayes.h"
#include "algorithm/naive_bayes/bayes_estimation.h"
#include "utils/data/data_utils.h"

using NormalFeatureListAndLabelList = algorithm::learner::LearnerBase::NormalFeatureListAndLabelList;

int main(int argc, char** argv) {
  // TODO(mayf3) Use google command line to parse argc and argv.
  if (argc < 2) {
    printf("Hint : %s input_file\n", argv[0]);
    return -1;
  }
  const char* input_filename = argv[1];

  NormalFeatureListAndLabelList feature_list_and_label_list;
  utils::data::ReadFeatureListAndLabelList(input_filename, &feature_list_and_label_list);

  NormalFeatureListAndLabelList training_data;
  NormalFeatureListAndLabelList testing_data;

  utils::data::SplitIntoTrainingAndTesting(feature_list_and_label_list, &training_data,
                                           &testing_data);

  assert(training_data.feature_list.size() > 0);
  std::unique_ptr<algorithm::learner::LearnerBase> learner(
      new algorithm::naive_bayes::NaiveBayes(training_data.feature_list, training_data.label_list,
                                         training_data.feature_list[0].size(), 2));
  // std::unique_ptr<algorithm::learner::LearnerBase> learner(
  //     new algorithm::naive_bayes::BayesEstimation(training_data.feature_list, training_data.label_list,
  //                                        training_data.feature_list[0].size(), 2));

  int correct_times = 0;
  for (int i = 0; i < testing_data.feature_list.size(); i++) {
    const int predict_result = learner->Predict(testing_data.feature_list[i]);
    if (predict_result == testing_data.label_list[i]) {
      correct_times++;
    }
  }
  std::cout << " Correct Rate: " << std::fixed << std::setprecision(2)
            << static_cast<double>(correct_times) / testing_data.feature_list.size() * 100.0
            << " % (" << correct_times << "/" << testing_data.feature_list.size() << ")"
            << std::endl;
  return 0;
}
