#include "oslcompiler.hpp"


#include <OSL/oslcomp.h>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/errorhandler.h>

#include <memory>

bool compile_shader(std::string& outputOsoBuffer, const char* oslFilename, const std::vector<std::string>& options, OIIO::string_view stdoslpath)
{

    auto errorHandler = std::make_unique<OIIO::ErrorHandler>();
    OSL::OSLCompiler compiler (errorHandler.get());
    if (!compiler.compile(oslFilename, options, stdoslpath))
    {
        return false;
    }

    // return the compiled buffer
    OIIO::ifstream is (compiler.output_filename());
    if (!is)
    {
        return false;
    }

    is.seekg(0, std::ios::end);
    outputOsoBuffer.resize(is.tellg());
    is.seekg(0, std::ios::beg);
    is.read(&outputOsoBuffer[0], outputOsoBuffer.size());
    
    return true;
}


