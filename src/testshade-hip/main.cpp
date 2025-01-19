#include <iostream>
#include <memory>

#include <OpenImageIO/filesystem.h>
#include <OSL/oslquery.h>

#include "utils.hpp"
#include "argparse.hpp"
#include "oslcompiler.hpp"
#include "simplerenderer.hpp"
#include "hiprenderer.hpp"


struct ShaderDesc
{
    std::string filename;
    std::string layer_name;
    std::string code;
    bool compiled {false};
};



void setup_output_images(SimpleRenderer* renderer, OSL::ShadingSystem* shadingSystem, OSL::ShaderGroupRef& shaderGroup,  const ProgramArguments& programArgs)
{
    // Get info about the number of layers in the shader group
    int num_layers { 0 };
    shadingSystem->getattribute(shaderGroup.get(), "num_layers", num_layers);

    std::vector<OIIO::ustring> layernames(num_layers);
    if (num_layers)
    {
        shadingSystem->getattribute(shaderGroup.get(), "layer_names",
                                    OIIO::TypeDesc(OIIO::TypeDesc::STRING, num_layers),
                                    &layernames[0]);
    }

    OIIO::TypeDesc vartype;
    bool found = false;

    std::cerr << "Warning assumed layer name is test_0" << std::endl;
    OIIO::string_view layer = "test_0";

    for (int lay = num_layers - 1; lay >= 0 && !found; --lay) {
            // std::cout << "   layer " << lay << " " << layernames[lay] << "\n";
            if (layer == layernames[lay] || layer.empty()) {
                OSL::OSLQuery oslquery = shadingSystem->oslquery(*shaderGroup, lay);
                for (const auto& param : oslquery) {
                    if (param.isoutput) {
                        //std::cerr << "    found param " << param.name << "\n";
                        vartype = param.type;
                        found   = true;
                        break;
                    }
                }
            }
    }
    if (found) 
    {
        std::cerr << "Found output variable" << std::endl;
        if (programArgs.output.value != "null")
        {
            std::cout << "Output " << programArgs.output.value << " to " << programArgs.output.filename << std::endl;
        }
        OIIO::TypeDesc basetype((OIIO::TypeDesc::BASETYPE)vartype.basetype);
        int number_of_channels = vartype.basevalues();

        if (renderer->add_output(programArgs.output.value, programArgs.output.value, basetype, number_of_channels))
        {
            std::cout << "Added output " << programArgs.output.value 
                      << " to " << programArgs.output.filename 
                      << " (channels " << number_of_channels << ")" << std::endl;
        }
    }
    else 
    {
        std::cerr << "Could not find output variable" << std::endl;
    }
}

int main(int argc, const char *argv[])
{

    setup_crash_stacktrace("test-shade-hip-crash.log");

    auto programArguments = parse_arguments(argc, argv);

    if (!programArguments.has_value())
    {
        return EXIT_FAILURE;
    }

    ProgramArguments programArgs = programArguments.value();

    std::vector<ShaderDesc> shaders(programArgs.filenames.size());
    std::vector<std::string> compileOptions {};

    unsigned int index {0};
    std::for_each(programArgs.filenames.begin(), programArgs.filenames.end(), [&shaders, &index, &compileOptions](const char *filename)
        {
            ShaderDesc& shader = shaders[index];
            shader.filename = filename;

            if (!compile_shader(shader.code, shader.filename.c_str(), compileOptions))
            {
                std::cerr << "Could not compile shader: " << filename << std::endl;
                
            }
            shader.compiled = true;
            shader.layer_name = GetLayerNameFromFilename(shader.filename);
            index++;
        }
    );

    // check for compilation errors with if init statement  
    bool compile_result = std::all_of(shaders.begin(), shaders.end(), [](const ShaderDesc& shader) {return shader.compiled;});

    if (compile_result == false)
    {
        std::cerr << "Compilation failed" << std::endl;
        return EXIT_FAILURE;
    }

    
    std::cout << "Compilation successful" << std::endl;


    std::unique_ptr<SimpleRenderer> renderer {nullptr};

    if (programArgs.hip)
    {
        renderer = std::make_unique<HIPRenderer>();
    }
    else
    {
        renderer = std::make_unique<SimpleRenderer>();
    }

    if (programArgs.debug)
    {
        renderer->errhandler().verbosity(OIIO::ErrorHandler::VERBOSE);
    }

    // sets some renderer options / attributes
    // the interface is fucked because without looking into the code
    // you can't know what the function does set or get 
    renderer->attribute("savebc", (int)programArgs.save_bc);
    renderer->attribute("no_rend_lib_bitcode", (int)programArgs.no_render_services_bitcode);

    std::shared_ptr<OIIO::TextureSystem> textureSystem = OIIO::TextureSystem::create();
    
    // init shading system
    OIIO::ErrorHandler shadingSystemErrorHandler {};
    std::unique_ptr<OSL::ShadingSystem> shadingSystem = std::make_unique<OSL::ShadingSystem>(renderer.get(), textureSystem.get(), &shadingSystemErrorHandler);
    renderer->init_shadingsys(shadingSystem.get());

    register_closures(shadingSystem.get());

    // set the shading system options
    OSL_DEV_ONLY(shadingSystem->attribute("clearmemory", 1));
    // Always generate llvm debugging info
    shadingSystem->attribute("llvm_debugging_symbols", 1);
 // Always emit llvm Intel profiling events
    shadingSystem->attribute("llvm_profiling_events", 1);

    OSL_DEV_ONLY(llvm_debug = true);
    shadingSystem->attribute("llvm_debug", (programArgs.llvm_debug ? 2 : 0));
    shadingSystem->attribute("compile_report", programArgs.llvm_debug);

    const int optimization_level {2};
    shadingSystem->attribute("optimize", optimization_level);

    // Instead of playing with multiple args and options we simply load the shader here:
    OSL::ShaderGroupRef shaderGroup = shadingSystem->ShaderGroupBegin("jpa_group");
    for (const ShaderDesc& shader : shaders)
    {
        // compile shader
        shadingSystem->LoadMemoryCompiledShader(shader.layer_name, shader.code);
        // add it to the shading system
        shadingSystem->Shader(*shaderGroup, "surface", shader.layer_name, "");
    }
    shadingSystem->ShaderGroupEnd(*shaderGroup);

    PrintShaderGroup(shadingSystem.get(), shaderGroup);

    // adds the shader to the renderer
    renderer->shaders().push_back(shaderGroup);

    // Adjust rendering parameters
    OSL::Matrix44 world_to_camera;
    world_to_camera.makeIdentity();
    renderer->camera_params(world_to_camera, OSL::ustring("perspective"), 90.0f, 0.1f, 1000.0f, programArgs.xres, programArgs.yres);

    // Make a "shader" space that is translated one unit in x and rotated
    // 45deg about the z axis.
    OSL::Matrix44 Mshad;
    Mshad.makeIdentity();
    Mshad.translate(OSL::Vec3(1.0, 0.0, 0.0));
    Mshad.rotate(OSL::Vec3(0.0, 0.0, M_PI_4));

    OSL::Matrix44 Mobj;
    Mobj.makeIdentity();
    Mobj.translate(OSL::Vec3(0.0, 1.0, 0.0));
    Mobj.rotate(OSL::Vec3(0.0, 0.0, M_PI_2));
    // std::cout << "object-to-common matrix: " << Mobj << "\n";

    OSL::Matrix44 Mmyspace;
    Mmyspace.scale(OSL::Vec3(13.0, 2.0, 1.0));
    // std::cout << "myspace-to-common matrix: " << Mmyspace << "\n";
    renderer->name_transform("myspace", Mmyspace);

    if (programArgs.hip)
    {
        reinterpret_cast<HIPRenderer*>(renderer.get())->set_transforms(Mobj, Mshad);
        reinterpret_cast<HIPRenderer*>(renderer.get())->register_named_transforms();
        // from this point the named transforms are visible and can be queried by the get_matrix methods 
        // {
        //     OSL::Matrix44 matrix;
        //     auto hipRenderer = reinterpret_cast<HIPRenderer*>(renderer.get());
        //     hipRenderer->get_matrix(nullptr, matrix, OSL::ustring("myspace"), 0.f);

        //     std::cout << "myspace matrix: " << matrix << std::endl;
        //     std::cout << "object-to-common matrix: " << Mmyspace << std::endl;
        // }
    }

    // this is crap
    setup_output_images(renderer.get(), shadingSystem.get(), shaderGroup, programArgs);

    // prepare render <--- compile components, setup globals and other stuff on the device
    /*
     // we must use function pointers to get the names of the functions for shader group
     // the order of shaders must be taken from the connection list?
     // let's assume first that we have only one shader
     // the trampoline is not required for the simple renderer
     // pas the shaders entrypoints as a list of the device functions like in hiprt  
     std::string init_name, entry_name;
    if (!shadingSystem.getattribute(shaderGroup.get(), "group_init_name", init_name)) {
        std::cerr << "Error getting shader group init name" << std::endl;
        return 1;
    }

    if (!shadingSystem.getattribute(shaderGroup.get(), "group_entry_name", entry_name)) {
        std::cerr << "Error getting shader group entry name" << std::endl;
        return 1;
    }


    */


    // warmup and run the render 

    // get back the data from the device (finalize_pixel_buffer)

    // clear the renderer reset release 





    
    std::cerr << "Program ended successfully" << std::endl;
    return EXIT_SUCCESS;
}