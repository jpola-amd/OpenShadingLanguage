#include "utils.hpp"
#include <iostream>

#include <OpenImageIO/filesystem.h>
#include <OSL/oslquery.h>

bool
setup_crash_stacktrace(OIIO::string_view filename)
{
    bool result = false;
#if defined(OIIO_HAS_STACKTRACE)
    result = OIIO::Sysutil::setup_crash_stacktrace(filename);
#else
    (void)filename;
    std::cerr << "setup_crash_stacktrace() OIIO_HAS_STACKTRACE is not defined\n";
#endif
    return result;
}


std::string
GetLayerNameFromFilename(const std::string& filename)
{
    auto f = OIIO::Filesystem::filename(filename);
    return OIIO::Filesystem::replace_extension(f, "");
}

void
PrintShaderGroup(OSL::ShadingSystem* shadingsys,
                 OSL::ShaderGroupRef shadergroup)
{
    std::string pickle;
    shadingsys->getattribute(shadergroup.get(), "pickle", pickle);
    fmt::print("Shader group:\n---\n{}\n---\n\n", pickle);

    OIIO::ustring groupname;
    shadingsys->getattribute(shadergroup.get(), "groupname", groupname);
    fmt::print("Shader group \"{}\" layers are:\n", groupname);

    int num_layers = 0;
    shadingsys->getattribute(shadergroup.get(), "num_layers", num_layers);
    if (num_layers > 0) {
        std::vector<const char*> layers(size_t(num_layers), nullptr);
        shadingsys->getattribute(shadergroup.get(), "layer_names",
                                 OIIO::TypeDesc(OIIO::TypeDesc::STRING,
                                                num_layers),
                                 &layers[0]);
        for (int i = 0; i < num_layers; ++i) {
            fmt::print("    {}\n", layers[i] ? layers[i] : "<unnamed>");

            OSL::OSLQuery q = shadingsys->oslquery(*shadergroup, i);
            for (size_t p = 0; p < q.nparams(); ++p) {
                const OSL::OSLQuery::Parameter* param = q.getparam(p);
                fmt::print("\t{}{} {}\n", param->isoutput ? "output " : "",
                           param->type, param->name);
            }
        }
    }
    fmt::print("\n");
}
