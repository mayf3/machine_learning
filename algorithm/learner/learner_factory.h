// Copyright @2021 mayf3

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "algorithm/learner/learner_options.h"

namespace algorithm {
namespace learner {

class LearnerFactory {
 public:
  using Creator = std::function<std::unique_ptr<LearnerBase>(const LearnerOptions&)>;

  static LearnerFactory* GetInstance();

  void RegisterLearner(const std::string& name, const Creator& func);

  std::unique_ptr<LearnerBase> Create(const std::string& name, const LearnerOptions& options) const;

  ~LearnerFactory() = default;

 private:
  LearnerFactory() = default;

  std::unordered_map<std::string, Creator> creators_;
};

template <typename LearnerType>
class LearnerRegistrar {
 public:
  explicit LearnerRegistrar(const std::string& learner_name) {
    LearnerFactory::GetInstance()->RegisterLearner(
        learner_name, [](const LearnerOptions& options) -> std::unique_ptr<LearnerBase> {
          return std::make_unique<LearnerType>(options);
        });
  }
};

#define REGISTER_LEARNER(LEARNER_TYPE, LEARNER_NAME)                                            \
  static ::algorithm::learner::LearnerRegistrar<LEARNER_TYPE> learner_registrar_##LEARNER_NAME( \
      LEARNER_NAME);

}  // namespace learner
}  // namespace algorithm
