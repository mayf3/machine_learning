package(default_visibility = ["//visibility:public"])

cc_library(
    name = "knn_brute_force",
    srcs = ["knn_brute_force.cc"],
    hdrs = ["knn_brute_force.h"],
    deps = [
        "//utils/common:define",
        "//utils/math:math_utils",
    ],
)

cc_binary(
    name = "knn_main",
    srcs = ["knn_main.cc"],
    deps = [
        ":knn_brute_force",
        "//utils/data:data_utils",
    ],
)
