#ifndef UTILS_HPP
#define UTILS_HPP

#include <OpenImageIO/sysutil.h>
#include <OSL/oslexec.h>

bool setup_crash_stacktrace(OIIO::string_view filename);

std::string GetLayerNameFromFilename(const std::string& filename);

void PrintShaderGroup(OSL::ShadingSystem* shadingsys, OSL::ShaderGroupRef shadergroup);

#endif // UTILS_HPP