// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>
#include <OSL/oslquery.h>
#include <OSL/rendererservices.h>
#include <OSL/shaderglobals.h>

#include <OpenImageIO/filesystem.h>

#include <cmath>
#include <cstdlib>
#include <string>

using namespace OSL;

namespace {

bool
check(string_view stdosl)
{
    if (!OIIO::Filesystem::exists(stdosl)) {
        print(stderr, "Installed stdosl.h does not exist: {}\n", stdosl);
        return false;
    }
    OSLCompiler compiler;
    std::string bytecode;
    if (!compiler.compile_buffer(
            "shader installed_consumer(float gain=2, output float result=0) "
            "{ result=gain*u+v+0.125; }",
            bytecode, { }, stdosl, "installed_consumer.osl")) {
        print(stderr, "Installed OSLCompiler failed\n");
        return false;
    }
    OSLQuery query;
    if (!query.open_bytecode(bytecode)) {
        print(stderr, "Installed OSLQuery failed: {}\n", query.geterror());
        return false;
    }
    const auto* gain   = query.getparam(ustring("gain"));
    const auto* result = query.getparam(ustring("result"));
    if (query.shadername() != ustring("installed_consumer")
        || query.nparams() != 2 || !gain || !result || gain->isoutput
        || gain->type != TypeFloat || !gain->validdefault
        || gain->fdefault.size() != 1 || gain->fdefault[0] != 2.0f
        || !result->isoutput || result->type != TypeFloat
        || !result->validdefault || result->fdefault.size() != 1
        || result->fdefault[0] != 0.0f) {
        print(stderr,
              "Installed OSLQuery returned incorrect parameter metadata\n");
        return false;
    }
    // The unextended services select CPU execution, even in a HART/OptiX build.
    RendererServices renderer;
    ShadingSystem ss(&renderer);
    if (!ss.attribute("optimize", 2) || !ss.attribute("llvm_optimize", 3)
        || !ss.LoadMemoryCompiledShader("installed_consumer", bytecode)) {
        print(stderr, "Cannot configure installed CPU ShadingSystem\n");
        return false;
    }
    auto group                = ss.ShaderGroupBegin("installed");
    const float override_gain = 3.0f;
    if (!group
        || !ss.Parameter(*group, "gain", TypeFloat, &override_gain,
                         ParamHints::none)
        || !ss.Shader(*group, "surface", "installed_consumer", "layer")
        || !ss.ShaderGroupEnd(*group)
        || !ss.attribute(group.get(), "renderer_outputs", "result")) {
        print(stderr, "Cannot construct installed CPU shader group\n");
        return false;
    }
    ShaderGlobals sg { };
    sg.P = sg.dPdx = sg.dPdy = sg.dPdz = Vec3(0);
    sg.I = sg.dIdx = sg.dIdy = sg.N = sg.Ng = Vec3(0);
    sg.dPdu = sg.dPdv = sg.dPdtime = sg.Ps = sg.dPsdx = sg.dPsdy = Vec3(0);
    sg.u                                                         = 0.25f;
    sg.v                                                         = 0.5f;
    auto* thread = ss.create_thread_info();
    if (!thread) {
        print(stderr, "Cannot create installed CPU thread state\n");
        return false;
    }
    auto* context = ss.get_context(thread);
    if (!context) {
        ss.destroy_thread_info(thread);
        print(stderr, "Cannot create installed CPU shading context\n");
        return false;
    }
    const bool executed = ss.execute(*context, *group, 0, 0, sg, nullptr,
                                     nullptr);
    TypeDesc type;
    const void* data = executed ? ss.get_symbol(*context, ustring("layer"),
                                                ustring("result"), type)
                                : nullptr;
    const bool valid = data && type == TypeFloat
                       && std::isfinite(*static_cast<const float*>(data))
                       && *static_cast<const float*>(data) == 1.375f;
    ss.release_context(context);
    ss.destroy_thread_info(thread);
    if (!valid) {
        print(stderr,
              "Installed CPU execution did not produce exactly 1.375\n");
        return false;
    }
    print("Installed OSL consumer: result=1.375\n");
    return true;
}

}  // namespace



int
main(int argc, const char* argv[])
{
    OIIO::Filesystem::convert_native_arguments(argc, argv);
    if (argc != 2) {
        print(stderr, "Usage: osl_install_consumer installed-stdosl.h\n");
        return EXIT_FAILURE;
    }
    return check(argv[1]) ? EXIT_SUCCESS : EXIT_FAILURE;
}
