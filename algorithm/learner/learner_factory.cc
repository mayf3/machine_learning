// Copyright @2021 mayf3

#include "algorithm/learner/learner_factory.h"

namespace algorithm {
namespace learner {

LearnerFactory* LearnerFactory::GetInstance() {
  static LearnerFactory learner_factory;
  return &learner_factory;
}

void LearnerFactory::RegisterLearner(const std::string& name, const Creator& func) {
  assert(creators_.count(name));
  creators_.emplace(name, func);
}

std::unique_ptr<LearnerBase> LearnerFactory::Create(const std::string& name, const LearnerOptions& options) const {
  if (creators_.count(name)) {
    return creators_.at(name)(options);
  }
  std::cout << "No such name : " << name << std::endl;
  return nullptr;
}

}  // namespace learner
}  // namespace algorithm
