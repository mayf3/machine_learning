package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//utils/math:math_utils",
    ],
)

cc_library(
    name = "perceptron",
    srcs = ["perceptron.cc"],
    hdrs = ["perceptron.h"],
    deps = [
        ":utils",
        "//algorithm/learner:learner_base",
    ],
)

cc_library(
    name = "perceptron_dual",
    srcs = ["perceptron_dual.cc"],
    hdrs = ["perceptron_dual.h"],
    deps = [
        ":utils",
        "//algorithm/learner:learner_base",
    ],
)

cc_binary(
    name = "perceptron_main",
    srcs = ["perceptron_main.cc"],
    deps = [
        ":perceptron",
        ":perceptron_dual",
    ],
)
