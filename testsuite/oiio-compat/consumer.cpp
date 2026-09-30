// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause

#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>
#include <OSL/oslquery.h>
#include <OSL/rendererservices.h>
#include <OpenImageIO/unittest.h>

#include <cstring>

#if defined(EXPECT_STATIC_OSL) && !defined(OSL_STATIC_DEFINE)
#    error "The installed static targets must propagate OSL_STATIC_DEFINE"
#endif

using namespace OSL;



int
main(int argc, char* argv[])
{
    if (argc != 2) {
        print(stderr, "Usage: oiio_compat_consumer path-to-stdosl.h\n");
        return 1;
    }
    OIIO_CHECK_EQUAL(ustring("world"), Strings::world);
    OIIO_CHECK_EQUAL(ustring("common"), Strings::common);
    OIIO_CHECK_EQUAL(ustring("object"), Strings::object);
    OSLCompiler compiler;
    std::string oso;
    OIIO_CHECK_ASSERT(compiler.compile_buffer(
        "shader oiio_compat(output color Cout=color(1,2,3)) {}", oso, {},
        argv[1]));
    OSLQuery query;
    OIIO_CHECK_ASSERT(query.open_bytecode(oso));
    OIIO_CHECK_EQUAL(query.nparams(), 1);
    if (unit_test_failures)
        return 1;
    const ustring output_name("Cout");
    const auto* parameter = query.getparam(size_t(0));
    OIIO_CHECK_EQUAL(parameter->name, output_name);
    OIIO_CHECK_EQUAL(parameter->type, TypeColor);

    RendererServices renderer;
    ShadingSystem shading(&renderer);
    const char* outputs[] = { output_name.c_str() };
    OIIO_CHECK_ASSERT(shading.attribute("renderer_outputs",
                                        TypeDesc(TypeDesc::STRING, 1),
                                        outputs));
    OIIO_CHECK_ASSERT(shading.LoadMemoryCompiledShader("oiio_compat", oso));
    auto group = shading.ShaderGroupBegin("oiio_compat_group");
    OIIO_CHECK_ASSERT(group);
    OIIO_CHECK_ASSERT(shading.Shader("surface", "oiio_compat", "surface"));
    OIIO_CHECK_ASSERT(shading.ShaderGroupEnd());
    if (unit_test_failures)
        return 1;

    auto* thread = shading.create_thread_info();
    auto* ctx    = shading.get_context(thread);
    OIIO_CHECK_ASSERT(ctx);
    if (ctx) {
        ShaderGlobals globals;
        std::memset(&globals, 0, sizeof(globals));
        OIIO_CHECK_ASSERT(
            shading.execute(*ctx, *group, 0, 0, globals, nullptr, nullptr));
        const auto* symbol = shading.find_symbol(*group, output_name);
        OIIO_CHECK_ASSERT(symbol);
        OIIO_CHECK_ASSERT(symbol
                          == shading.find_symbol(*group, parameter->name));
        TypeDesc type;
        const auto* data = static_cast<const float*>(
            shading.get_symbol(*ctx, output_name, type));
        OIIO_CHECK_ASSERT(data);
        OIIO_CHECK_EQUAL(type, TypeColor);
        if (data && type == TypeColor) {
            for (int i = 0; i < 3; ++i)
                OIIO_CHECK_EQUAL(data[i], float(i + 1));
        }
        shading.release_context(ctx);
    }
    shading.destroy_thread_info(thread);
    if (!unit_test_failures)
        print(
            "OIIO {}: string identity, compiler, query and CPU output passed\n",
            OIIO_VERSION_STRING);
    return unit_test_failures ? 1 : 0;
}
