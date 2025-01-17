#include <iostream>
#include <memory>

#include "utils.hpp"
#include "argparse.hpp"
#include "oslcompiler.hpp"
#include "simplerenderer.hpp"
#include "hiprenderer.hpp"

int main(int argc, const char *argv[])
{

    setup_crash_stacktrace("test-shade-hip-crash.log");

    auto programArguments = parse_arguments(argc, argv);

    if (!programArguments.has_value())
    {
        return EXIT_FAILURE;
    }

    ProgramArguments args = programArguments.value();

    std::vector<std::string> shaders(args.filenames.size());
    std::vector<std::string> compileOptions {};

    unsigned int index {0};
    bool compile_result = true;
    std::for_each(args.filenames.begin(), args.filenames.end(), [&shaders, &index, &compileOptions, &compile_result](const char *filename)
        {
            if (!compile_shader(shaders[index], filename, compileOptions))
            {
                std::cerr << "Could not compile shader: " << filename << std::endl;
                compile_result = false;
            }
            index++;
        }
    );

    if (!compile_result)
    {   
        std::cerr << "Compilation failed" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Compilation successful" << std::endl;


    std::unique_ptr<SimpleRenderer> renderer {nullptr};

    if (args.hip)
    {
        renderer = std::make_unique<HIPRenderer>();
    }
    else
    {
        renderer = std::make_unique<SimpleRenderer>();
    }

    
    std::cerr << "Program ended successfully" << std::endl;
    return EXIT_SUCCESS;
}