package(default_visibility = ["//visibility:public"])

filegroup(name = "clang_tidy", srcs = ["{binary}"])

#: its resource headers (clang's CUDA wrappers among them), beside the binary where clang looks for them
filegroup(name = "resources", srcs = glob(["clang_tidy/data/lib/clang/**"]))
