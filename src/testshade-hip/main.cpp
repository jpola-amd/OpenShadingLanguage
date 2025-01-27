#include <iostream>
#include <memory>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imagecache.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/timer.h> // to be used later

#include <OSL/oslquery.h>
#include <OSL/journal.h>

#include "utils.hpp"
#include "argparse.hpp"
#include "oslcompiler.hpp"
#include "simplerenderer.hpp"
#include "hiprenderer.hpp"
#include "renderstate.hpp"

// let's make it simple for now
static RenderState g_renderstate;

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
            std::cout << "   layer " << lay << " " << layernames[lay] << "\n";
            if (layer == layernames[lay] || layer.empty()) {
                OSL::OSLQuery oslquery = shadingSystem->oslquery(*shaderGroup, lay);
                for (const auto& param : oslquery) {
                     std::cout << "    param " << param.type << " " << param.name
                              << " isoutput=" << param.isoutput << "\n";
                    if (param.isoutput) {
                        std::cerr << "    found param " << param.name << "\n";
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

        if (renderer->add_output(programArgs.output.value, programArgs.output.filename, basetype, number_of_channels))
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

    std::cout << "Render outputs: " << renderer->noutputs() << std::endl;

    // if (renderer->noutputs() > 0)
    // {
    //     std::vector<OSL::SymLocationDesc> symlocs;
    //     for (size_t i = 0; i < renderer->noutputs(); ++i)
    //     {
    //         OIIO::ImageBuf* ib = renderer->outputbuf(i);
    //         char* outptr = static_cast<char*>(ib->pixeladdr(0, 0));
    //         if (i == 0)
    //         {
    //             // The output arena is the start of the first output jbuffer
    //             g_renderstate.output_base_ptr = outptr;
    //         }


    //         ptrdiff_t offset = outptr - g_renderstate.output_base_ptr;
    //         OIIO::TypeDesc t = vartype;
    //         auto outputname = renderer->outputname(i);
    //         symlocs.emplace_back(outputname, t, false, OSL::SymArena::Outputs, offset, t.size());
    //         std::cout.flush();
    //         OIIO::Strutil::print("Output buffer - symloc {} {} off={} size={}\n",
    //                               outputname, t, offset, t.size());
    //     }
    //     shadingSystem->add_symlocs(shaderGroup.get(), symlocs);
    // }

    
    {
        // Old fashined way -- tell the shading system which outputs we want
        std::vector<std::string> outputvars {programArgs.output.value};

        std::vector<const char*> aovnames(outputvars.size());
        for (size_t i = 0; i < outputvars.size(); ++i) {
            OSL::ustring varname(outputvars[i]);
            aovnames[i] = varname.c_str();
            size_t dot  = varname.find('.');
            if (dot != OSL::ustring::npos) {
                // If the name contains a dot, it's intended to be layer.symbol
                varname = OSL::ustring(varname, dot + 1);
            }
        }
        // shadingsys->attribute(use_group_outputs ? shadergroup.get() : NULL,
        //                       "renderer_outputs",
        //                       TypeDesc(TypeDesc::STRING, (int)aovnames.size()),
        //                       &aovnames[0]);
        // Group outputs generates the memcpy in the kernel code. Let's avoid that for a moment;
        shadingSystem->attribute(NULL, "renderer_outputs",
                                  OIIO::TypeDesc(OIIO::TypeDesc::STRING, (int)aovnames.size()),
                                  &aovnames[0]);
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
    shadingSystem->attribute("debug", programArgs.debug);
    shadingSystem->attribute("compile_report", programArgs.llvm_debug);


    const int optimization_level {2};
    shadingSystem->attribute("optimize", optimization_level);

    // Instead of playing with multiple args and options we simply load the shader here:
    OSL::ShaderGroupRef shaderGroup = shadingSystem->ShaderGroupBegin("");
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
        renderer->set_transforms(Mobj, Mshad);
        renderer->register_named_transforms();
        renderer->initialize_render_parameters();
        
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

    // set the noumber of threads to 1 for now
    const int num_threads = 1; // we can do OIIO::Sysutil::hardware_concurrency();

    // init the journal to record the render
    constexpr int journal_size = 1 * 1024 * 1024;
    std::unique_ptr<uint8_t[]> journal_buffer = std::make_unique<uint8_t[]>(journal_size);
    if (!OSL::journal::initialize_buffer (journal_buffer.get(), journal_size, 1024, num_threads))
    {
        std::cerr << "Error initializing journal buffer" << std::endl;
        return 1;
    }


    {
        renderer->export_state(g_renderstate);
        // bleah
        g_renderstate.shaderGroup = shaderGroup.get();
        g_renderstate.shadingSystem = shadingSystem.get();
        g_renderstate.num_threads = num_threads;
        g_renderstate.iteration = 0;
        g_renderstate.max_iterations = programArgs.iterations;
        g_renderstate.journal_buffer = journal_buffer.get();

        g_renderstate.shader2common = OSL::TransformationPtr(&Mshad);
        g_renderstate.object2common = OSL::TransformationPtr(&Mobj);
    }


    // does nothing on a cpu
    // on gpu compiles the shaders and sets the global variables


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

    renderer->prepare_render(g_renderstate);

    for(int i = 0; i < programArgs.iterations; i++)
    {
        renderer->render(programArgs.xres, programArgs.yres, g_renderstate);
    }



    


    // get back the data from the device (finalize_pixel_buffer)
    renderer->finalize_pixel_buffer(); // does nothing on CPU

    {
        for (size_t i = 0; i < renderer->noutputs(); ++i) 
        {
            if (OIIO::ImageBuf* outputimg = renderer->outputbuf(i)) 
            {
                std::string filename = outputimg->name();
                OIIO::TypeDesc datatype = outputimg->spec().format;
                
                // JPEG, GIF, and PNG images should be automatically saved
                // as sRGB because they are almost certainly supposed to
                // be displayed on web pages.
                std::vector<std::string> valid_extensions = {".jpg", ".jpeg", ".gif", ".png"};
                if (std::any_of(valid_extensions.begin(), valid_extensions.end(), 
                    [&filename](const std::string& ext) { return OIIO::Strutil::iends_with(filename, ext); })) 
                {
                    OIIO::ImageBuf ccbuf = OIIO::ImageBufAlgo::colorconvert(*outputimg, "linear", "sRGB");
                    ccbuf.write(filename, datatype);
                }
                else
                {
                    outputimg->write(filename, datatype);
                }    
            }
        }
    }

    // Print some debugging info
    // Timings for setup, warmup, run, write

    // clear the renderer reset release 

    renderer.reset();
    shaderGroup.reset();
    shadingSystem.reset();

     int retcode = EXIT_SUCCESS;
    // Double check that there were no uncaught errors in the texture
    // system and image cache.
    std::string err = textureSystem->geterror();
    if (!err.empty()) {
        std::cout << "ERRORS left in TextureSystem:\n" << err << "\n";
        retcode = EXIT_FAILURE;
    }
    auto ic = textureSystem->imagecache();
    err     = ic ? ic->geterror() : std::string();
    if (!err.empty()) {
        std::cout << "ERRORS left in ImageCache:\n" << err << "\n";
        retcode = EXIT_FAILURE;
    }




    
    std::cerr << "Program ended successfully" << std::endl;
    return retcode;
}