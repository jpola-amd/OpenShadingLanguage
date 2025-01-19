#ifndef ARGPARSE_HPP
#define ARGPARSE_HPP

#include <OpenImageIO/span.h>

struct OutputDescriptor
{
    std::string filename {"output.png"};
    std::string value {"Cout"};
};

struct ProgramArguments
{
    // contains the osl file names
    OIIO::cspan<const char*> filenames;
    // use hip for rendering
    bool hip {false};
    // enable debug mode for verbose output
    bool debug {false};
    // llvm debug output
    int llvm_debug {false};
    // print statistics
    bool run_statistics {false};

    // renderer-supplied shadeops (rend_lib)
    bool no_render_services_bitcode {true};

    // save the generated bitcode
    bool save_bc {false};

    int xres {512};
    int yres {512};

    // render iterations
    int iterations {1};

    OutputDescriptor output;

    bool valid() const;
};


std::optional<ProgramArguments> parse_arguments(int argc, const char* argv[]);

#endif // ARGPARSE_HPP