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

cc_binary(
    name = "knn_with_mnist_main",
    srcs = ["knn_with_mnist_main.cc"],
    data = [
        "//data/multi_class_classification/mnist:mnist",
    ],
    deps = [
        ":knn_brute_force",
        "//utils/data:data_utils",
        "//utils/data:mnist_data_parser",
    ],
)
