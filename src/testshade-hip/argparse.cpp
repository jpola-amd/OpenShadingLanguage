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