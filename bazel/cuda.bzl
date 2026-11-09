"""THE GRAPH'S CUDA COMPILE: rules_cuda's toolchain, flags and command line, run by the graph's own action, which drops
the headers a unit did not read from its cache key and keeps ptxas's report (docs/build-graph.md#compile)."""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", CC_ACTION_NAMES = "ACTION_NAMES")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cuda//cuda:defs.bzl", "find_cuda_toolchain")
load("@rules_cuda//cuda/private:action_names.bzl", "ACTION_NAMES")
load("@rules_cuda//cuda/private:cuda_helper.bzl", "cuda_helper")

CudaUnitInfo = provider(
    doc = "The units a target compiled or links, for ratify's census (bazel/defs.bzl's `ratify_census`).",
    fields = {"units": "depset of struct(source, report, object): each unit's source, nvcc's diagnostics, its object"},
)

#: rules_cuda's own additions to nvcc's command line, off on every unit: its extended lambdas, its compilation-mode
#: optimization and seed, the host toolchain's flags -- the command line is the unit's options alone
_RULES_CUDA_ADDITIONS = [
    "-cuda_host_use_toolchain_flags",
    "-dbg",
    "-fastbuild",
    "-nvcc_extended_lambda",
    "-nvcc_fixed_random_seed",
    "-nvcc_relaxed_constexpr",
    "-opt",
]

def _cuda_unit_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    cuda_toolchain = find_cuda_toolchain(ctx)
    common = cuda_helper.create_common(ctx)
    if cuda_helper.check_must_enforce_rdc(cuda_archs_info = common.cuda_archs_info):
        fail("%s: the build's architectures need relocatable device code, which the extension's units never are" % ctx.label)
    cc_features = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    host_compiler = cc_common.get_tool_for_action(feature_configuration = cc_features, action_name = CC_ACTION_NAMES.cpp_compile)
    cuda_features = cuda_helper.configure_features(ctx, cuda_toolchain, requested_features = [ACTION_NAMES.cuda_compile])
    category = cuda_helper.get_artifact_category_from_action(ACTION_NAMES.cuda_compile, True, False)

    #: the prunable inputs: the declared headers, never the host compiler's identity (`host_cc`, its own input) or a tool
    candidates = ctx.actions.declare_file("_objs/%s/headers.txt" % ctx.label.name)
    listed = ctx.actions.args()
    listed.set_param_file_format("multiline")
    listed.add_all(common.headers)
    ctx.actions.write(candidates, listed)

    objects, reports, units, names = [], [], [], {}
    for src in ctx.files.srcs:
        stem = src.basename.removesuffix(".cu")
        if stem in names:
            fail("%s: two units named %s (%s, %s): their objects would collide" % (ctx.label, stem, names[stem], src.path))
        names[stem] = src.path
        obj = ctx.actions.declare_file("_objs/%s/%s" % (ctx.label.name, cuda_helper.get_artifact_name(cuda_toolchain, category, stem)))
        report = ctx.actions.declare_file("_objs/%s/%s.ptxas.txt" % (ctx.label.name, stem))
        unused = ctx.actions.declare_file("_objs/%s/%s.unused.txt" % (ctx.label.name, stem))
        variables = cuda_helper.create_compile_variables(
            cuda_toolchain,
            cuda_features,
            common.cuda_archs_info,
            common.sysroot,
            source_file = src.path,
            output_file = obj.path,
            host_compiler = host_compiler,
            compile_flags = common.compile_flags,
            host_compile_flags = common.host_compile_flags,
            toolchain_host_compile_flags = common.toolchain_host_compile_flags,
            include_paths = common.includes,
            quote_include_paths = common.quote_includes,
            system_include_paths = common.system_includes,
            external_include_paths = common.external_includes,
            defines = common.local_defines + common.defines,
            host_defines = common.host_local_defines + common.host_defines,
            ptxas_flags = common.ptxas_flags,
            cpp_copts = common.cpp_copts,
            cpp_cxxopts = common.cpp_cxxopts,
            cpp_linkopts = common.cpp_linkopts,
            use_pic = True,
            use_rdc = False,
        )
        args = ctx.actions.args()
        args.add(ctx.file._script)
        args.add("--source", src)
        args.add("--candidates", candidates)
        args.add("--unused", unused)
        args.add("--report", report)
        #: the compile's own command line follows the separator: the unit lints read it from there
        args.add("--")
        args.add(cuda_toolchain.compiler_executable)
        args.add_all(cuda_helper.get_command_line(cuda_features, ACTION_NAMES.cuda_compile, variables))
        ctx.actions.run(
            executable = ctx.file._python,
            arguments = [args],
            inputs = depset([src, candidates, ctx.file._script] + ctx.files.host_cc, transitive = [common.headers, cc_toolchain.all_files, cuda_toolchain.all_files]),
            outputs = [obj, report, unused],
            unused_inputs_list = unused,
            env = cuda_helper.get_environment_variables(cuda_features, ACTION_NAMES.cuda_compile, variables),
            mnemonic = "CudaCompile",
            progress_message = "Compiling %s" % src.path,
        )
        objects.append(obj)
        reports.append(report)
        units.append(struct(source = src.path, report = report, object = obj))

    linking, _ = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = cc_features,
        cc_toolchain = cc_toolchain,
        compilation_outputs = cc_common.create_compilation_outputs(pic_objects = depset(objects)),
        alwayslink = True,
        linking_contexts = common.transitive_linking_contexts,
        disallow_dynamic_library = True,
    )
    return [
        DefaultInfo(files = depset(objects)),
        OutputGroupInfo(ptxas_reports = depset(reports)),
        CudaUnitInfo(units = depset(units)),
        cc_common.merge_cc_infos(direct_cc_infos = [CcInfo(linking_context = linking)], cc_infos = [common.transitive_cc_info]),
    ]

_cuda_unit = rule(
    implementation = _cuda_unit_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".cu"], doc = "the units, one compile each"),
        "deps": attr.label_list(providers = [CcInfo], doc = "the headers and include roots the units compile against"),
        "copts": attr.string_list(doc = "nvcc's options (tools/build_flags.py's, through @rola_env//:flags.bzl)"),
        "host_cc": attr.label(mandatory = True, allow_files = True, doc = "the host C++ compiler's identity: an input of every compile, never pruned"),
        #: what rules_cuda's helper reads off a CUDA rule, at its own defaults
        "hdrs": attr.label_list(allow_files = True),
        "includes": attr.string_list(),
        "defines": attr.string_list(),
        "local_defines": attr.string_list(),
        "host_copts": attr.string_list(),
        "host_defines": attr.string_list(),
        "host_local_defines": attr.string_list(),
        "ptxasopts": attr.string_list(),
        "archs": attr.string_list(),
        "_builtin_deps": attr.label_list(default = ["@rules_cuda//cuda:runtime"]),
        "_default_cuda_copts": attr.label(default = "@rules_cuda//cuda:copts"),
        "_default_host_copts": attr.label(default = "@rules_cuda//cuda:host_copts"),
        "_default_cuda_archs": attr.label(default = "@rules_cuda//cuda:archs"),
        "_cc_toolchain": attr.label(default = "@rules_cc//cc:current_cc_toolchain"),
        "_script": attr.label(default = "//bazel:cuda_compile.py", allow_single_file = True),
        "_python": attr.label(default = "@rola_env//:python", allow_single_file = True, cfg = "exec"),
    },
    fragments = ["cpp"],
    #: rules_cuda's `use_cuda_toolchain()` names the type relative to the rule that calls it, our repository
    toolchains = use_cc_toolchain() + [Label("@rules_cuda//cuda:toolchain_type")],
    provides = [CcInfo, CudaUnitInfo],
    doc = "CUDA units compiled as rules_cuda's cuda_library compiles them (its toolchain, its command line, a " +
          "position-independent object, always linked), each by one action whose unread headers leave its cache key.",
)

def cuda_unit(name, features = [], **kwargs):
    """CUDA units, rules_cuda's own additions to nvcc's command line off (`_RULES_CUDA_ADDITIONS`)."""
    _cuda_unit(name = name, features = _RULES_CUDA_ADDITIONS + features, **kwargs)
