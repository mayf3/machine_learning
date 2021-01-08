package(default_visibility = ["//visibility:public"])

cc_library(
    name = "learner_name",
    srcs = ["learner_name.cc"],
    hdrs = ["learner_name.h"],
)

cc_library(
    name = "learner_base",
    srcs = ["learner_base.cc"],
    hdrs = ["learner_base.h"],
    deps = [
    ],
)

cc_library(
    name = "learner_options",
    srcs = ["learner_options.cc"],
    hdrs = ["learner_options.h"],
    deps = [
        ":learner_base",
    ],
)

cc_library(
    name = "learner_factory",
    srcs = ["learner_factory.cc"],
    hdrs = ["learner_factory.h"],
    deps = [
        ":learner_base",
        ":learner_options",
    ],
)
