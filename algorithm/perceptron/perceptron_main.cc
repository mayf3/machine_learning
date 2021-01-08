// Copyright @2021 mayf3

#include "algorithm/learner/learner_factory.h"
#include "algorithm/perceptron/perceptron.h"
#include "algorithm/perceptron/perceptron_all.h"
#include "algorithm/perceptron/perceptron_dual.h"
#include "algorithm/perceptron/utils.h"

using LearnerOptions = algorithm::learner::LearnerOptions;
using LearnerBase = algorithm::learner::LearnerBase;
using LearnerFactory = algorithm::learner::LearnerFactory;
using Perceptron = algorithm::perceptron::Perceptron;
using PerceptronAll = algorithm::perceptron::PerceptronAll;
using PerceptronDual = algorithm::perceptron::PerceptronDual;

int main(int argc, char** argv) {
  srand(time(nullptr));
  LearnerOptions options;
  options.normal_feature_list.emplace_back(std::vector<double>{3, 3});
  options.normal_feature_list.emplace_back(std::vector<double>{4, 3});
  options.normal_feature_list.emplace_back(std::vector<double>{1, 1});
  options.normal_label_list.emplace_back(1);
  options.normal_label_list.emplace_back(1);
  options.normal_label_list.emplace_back(0);
  options.num_dim = 2;
  options.learning_rate = 1.0;
  std::unique_ptr<Perceptron> perceptron =
      LearnerFactory::GetInstance()->SpecialCreate<Perceptron>(kPerceptronName, options);
  std::cout << "perceptron: " << perceptron->num_iteration() << std::endl;
  std::unique_ptr<PerceptronAll> perceptron_all =
      LearnerFactory::GetInstance()->SpecialCreate<PerceptronAll>(kPerceptronAllName, options);
  std::cout << "perceptron_all: " << perceptron_all->num_iteration() << std::endl;
  std::unique_ptr<PerceptronDual> perceptron_dual =
      LearnerFactory::GetInstance()->SpecialCreate<PerceptronDual>(kPerceptronDualName, options);
  std::cout << "perceptron_dual: " << perceptron_dual->num_iteration() << std::endl;

  constexpr int kNumberOfTestCase = 100;
  constexpr int kNumberOfData = 30;
  constexpr int kNumberOfDim = 2;
  for (int i = 0; i < kNumberOfTestCase; i++) {
    options.normal_feature_list.clear();
    options.normal_label_list.clear();
    algorithm::perceptron::GenerateDataForPerceptron(
        kNumberOfData, kNumberOfDim, &options.normal_feature_list, &options.normal_label_list);
    perceptron = LearnerFactory::GetInstance()->SpecialCreate<Perceptron>(kPerceptronName, options);
    perceptron_all =
        LearnerFactory::GetInstance()->SpecialCreate<PerceptronAll>(kPerceptronAllName, options);
    std::cout << "case(" << i << ") perceptron: " << perceptron->num_iteration()
              << " perceptron_all: " << perceptron_all->num_iteration() << std::endl;
  }
  return 0;
}
