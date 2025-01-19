#include "argparse.hpp"
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/filesystem.h>

bool ProgramArguments::valid() const
{
    bool result {true};
    if (filenames.size() == 0)
    {
        std::cerr << "No input files specified" << std::endl;
        result = false;
    }

    if (filenames.size() > 0)
    {
        for (auto filename : filenames)
        {
            if (!OIIO::Filesystem::exists(filename))
            {
                std::cerr << "File not found: " << filename << std::endl;
                result = false;
            }
        }
    }

    return result;
}

std::optional<ProgramArguments> parse_arguments(int argc, const char* argv[])
{
    ProgramArguments programArguments;
    OIIO::ArgParse argParser;

    argParser.intro("testshade-hip -- Test Open Shading Language with HIP backend\n");
    argParser.usage("testshade-hip [options] <file.osl> <file2.osl> ...");

    argParser.arg("filename")
        .hidden()
        .action([&programArguments](OIIO::cspan<const char*> values) { programArguments.filenames = values; });

    argParser.arg("--hip", &programArguments.hip)
        .help("Use HIP backend");

    argParser.arg("--debug", &programArguments.debug)
        .help("Enable debug mode for verbose output");
    
    argParser.arg("--llvm_debug", &programArguments.llvm_debug)
        .help("Enable LLVM debug output");
    
    argParser.arg("--run_statistics", &programArguments.run_statistics)
        .help("Print run statistics");

    argParser.arg("--no_render_services_bitcode", &programArguments.no_render_services_bitcode)
        .help("Do not use renderer-supplied shadeops (rend_lib)");
    
    argParser.arg("--save_bc", &programArguments.save_bc)
        .help("Save the generated bitcode");
    
    // Different ways to handle grid size
    argParser.arg("-y %d:RES", &programArguments.yres)
        .help("Set the y resolution (default = 512)")
        .hidden();

    argParser.arg("-x %d:RES", &programArguments.xres)
        .help("Set the x resolution (default = 512)")
        .hidden();

    argParser.arg("--res %d:XRES %d:YRES", &programArguments.xres, &programArguments.yres)
        .help("Set the resolution");
    
    argParser.arg("-g %d:XRES %d:YRES", &programArguments.xres, &programArguments.yres)
        .hidden();

    argParser.arg("--output %s:FILENAME %s:VALUE", &programArguments.output.filename, &programArguments.output.value)
        .defaultval("output.png")
        .defaultval("Cout")
        .help("Set the output filename and value");

    if (argParser.parse_args(argc, argv) == -1)
    {
        return std::nullopt;
    }

    if (!programArguments.valid())
    {
        return std::nullopt;
    }

    return {programArguments};
}