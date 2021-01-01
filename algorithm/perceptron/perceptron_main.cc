
#include "algorithm/perceptron/perceptron.h"
#include "algorithm/perceptron/perceptron_dual.h"

int main(int argc, char** argv) {
  algorithm::perceptron::Perceptron::NormalFeatureList normal_feature_list;
  algorithm::perceptron::Perceptron::NormalLabelList normal_label_list;
  normal_feature_list.emplace_back(std::vector<double>{3, 3});
  normal_feature_list.emplace_back(std::vector<double>{4, 3});
  normal_feature_list.emplace_back(std::vector<double>{1, 1});
  normal_label_list.emplace_back(1);
  normal_label_list.emplace_back(1);
  normal_label_list.emplace_back(0);
  algorithm::perceptron::Perceptron perceptron(normal_feature_list, normal_label_list, 2);
  algorithm::perceptron::PerceptronDual perceptron_dual(normal_feature_list, normal_label_list, 2);
  return 0;
}
