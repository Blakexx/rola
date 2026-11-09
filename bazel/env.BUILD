load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

#: the interpreter the graph's actions run: the environment's base Python (a link it resolves to find its standard
#: library), standard library only
filegroup(name = "python", srcs = ["bin/python3"])


cc_library(
    name = "torch_headers",
    hdrs = glob(["site-packages/torch/include/**"]),
    includes = ["site-packages/torch/include", "site-packages/torch/include/torch/csrc/api/include"],
)

#: the torch libraries the extension links (its NEEDED)
filegroup(
    name = "torch_libraries",
    srcs = ["site-packages/torch/lib/libc10.so", "site-packages/torch/lib/libtorch_cpu.so"],
)

cc_library(
    name = "python_headers",
    hdrs = glob(["python_include/**"]),
    includes = ["python_include"],
)
