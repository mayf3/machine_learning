// Copyright @2020 mayf3

#pragma once

namespace algorithm {
namespace perceptron {

constexpr int kInternalPositiveType = 1;
constexpr int kInternalNegativeType = -1;
constexpr int kNormalPositiveType = 1;
constexpr int kNormalNegativeType = 0;

// It is not a normal Sign function.
// Return 1 if x >= 0, return -1 if x < 0;
int PerceptronSign(double x);

int InternalTypeToNormalType(int internal_type);

int NormalTypeToInternalType(int normal_type);

}  // namespace perceptron
}  // namespace algorithm
