// Copyright @2021 mayf3

#include "algorithm/perceptron/perceptron.h"
#include "algorithm/perceptron/perceptron_all.h"
#include "algorithm/perceptron/perceptron_dual.h"
#include "algorithm/perceptron/utils.h"

int main(int argc, char** argv) {
  srand(time(nullptr));
  algorithm::perceptron::Perceptron::NormalFeatureList normal_feature_list;
  algorithm::perceptron::Perceptron::NormalLabelList normal_label_list;
  normal_feature_list.emplace_back(std::vector<double>{3, 3});
  normal_feature_list.emplace_back(std::vector<double>{4, 3});
  normal_feature_list.emplace_back(std::vector<double>{1, 1});
  normal_label_list.emplace_back(1);
  normal_label_list.emplace_back(1);
  normal_label_list.emplace_back(0);
  algorithm::perceptron::Perceptron perceptron(normal_feature_list, normal_label_list, 2);
  std::cout << "perceptron: " << perceptron.num_iteration() << std::endl;
  algorithm::perceptron::PerceptronDual perceptron_dual(normal_feature_list, normal_label_list, 2);
  std::cout << "perceptron_dual: " << perceptron_dual.num_iteration() << std::endl;
  algorithm::perceptron::PerceptronAll perceptron_all(normal_feature_list, normal_label_list, 2);
  std::cout << "perceptron_all: " << perceptron_all.num_iteration() << std::endl;

  constexpr int kNumberOfTestCase = 100;
  constexpr int kNumberOfData = 30;
  constexpr int kNumberOfDim = 2;
  for (int i = 0; i < kNumberOfTestCase; i++) {
    normal_feature_list.clear();
    normal_label_list.clear();
    algorithm::perceptron::GenerateDataForPerceptron(kNumberOfData, kNumberOfDim,
                                                     &normal_feature_list, &normal_label_list);
    algorithm::perceptron::Perceptron new_perceptron(normal_feature_list, normal_label_list,
                                                     kNumberOfDim);
    algorithm::perceptron::PerceptronAll new_perceptron_all(normal_feature_list, normal_label_list,
                                                            kNumberOfDim);
    std::cout << "case(" << i << ") new_perceptron: " << new_perceptron.num_iteration()
              << " new_perceptron_all: " << new_perceptron_all.num_iteration() << std::endl;
  }
  return 0;
}
