package(default_visibility = ["//visibility:public"])

exports_files(["bin/cuobjdump", "bin/nvdisasm", "bin/nvcc"])

#: the pinned toolkit whole, the tree a test tier hands its tests (nvcc reads nvcc.profile, include/, lib64/ and nvvm/
#: beside it); its lib64 is a link to lib, which the glob would list twice
filegroup(name = "toolkit", srcs = glob(["**"], exclude = ["BUILD.bazel", "REPO.bazel", "redistrib.json", "curand/**", "lib64/**"]))

#: cuRAND's headers alone: clang's CUDA front end includes one (its runtime wrapper's `curand_mtgp32_kernel.h`)
filegroup(name = "curand_headers", srcs = glob(["curand/include/**"]))

#: the toolkit's compute-sanitizer, its libraries beside it (the sanitizer lane's)
filegroup(name = "compute_sanitizer", srcs = glob(["compute-sanitizer/**"]))
