package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//algorithm/learner:learner_base",
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
    name = "perceptron_all",
    srcs = ["perceptron_all.cc"],
    hdrs = ["perceptron_all.h"],
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
        ":perceptron_all",
        ":perceptron_dual",
        ":utils",
    ],
)
