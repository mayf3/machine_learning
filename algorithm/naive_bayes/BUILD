package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//algorithm/learner:learner_base",
    ],
)

cc_library(
    name = "naive_bayes",
    srcs = ["naive_bayes.cc"],
    hdrs = ["naive_bayes.h"],
    deps = [
        ":utils",
        "//algorithm/learner:learner_base",
        "//utils/math:math_utils",
    ],
)

cc_library(
    name = "bayes_estimation",
    srcs = ["bayes_estimation.cc"],
    hdrs = ["bayes_estimation.h"],
    deps = [
        ":utils",
        "//algorithm/learner:learner_base",
        "//utils/math:math_utils",
    ],
)

cc_binary(
    name = "naive_bayes_main",
    srcs = ["naive_bayes_main.cc"],
    deps = [
        ":naive_bayes",
        ":bayes_estimation",
        "//utils/data:data_utils",
    ],
)
