package(default_visibility = ["//visibility:public"])

cc_library(
    name = "knn_interface",
    srcs = ["knn_interface.cc"],
    hdrs = ["knn_interface.h"],
    deps = [
        "//utils/common:define",
        "//utils/math:math_utils",
    ],
)

cc_library(
    name = "knn_brute_force",
    srcs = ["knn_brute_force.cc"],
    hdrs = ["knn_brute_force.h"],
    deps = [
        ":knn_interface",
        "//utils/common:define",
        "//utils/math:math_utils",
    ],
)

cc_library(
    name = "knn_fast",
    srcs = ["knn_fast.cc"],
    hdrs = ["knn_fast.h"],
    deps = [
        ":knn_interface",
        "//utils/common:define",
        "//utils/math:math_utils",
    ],
)

cc_binary(
    name = "knn_main",
    srcs = ["knn_main.cc"],
    deps = [
        ":knn_brute_force",
        ":knn_fast",
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
        ":knn_fast",
        "//utils/time:stop_watch",
        "//utils/data:data_utils",
        "//utils/data:mnist_data_parser",
    ],
)
