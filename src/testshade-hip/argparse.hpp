#ifndef ARGPARSE_HPP
#define ARGPARSE_HPP

#include <OpenImageIO/span.h>


struct ProgramArguments
{
    OIIO::cspan<const char*> filenames;
    bool hip {false};
    bool debug {false};
    

    bool valid() const;
};


std::optional<ProgramArguments> parse_arguments(int argc, const char* argv[]);

#endif // ARGPARSE_HPP