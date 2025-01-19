#include "simplerenderer.hpp"

#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imagebufalgo_util.h>
#include <OSL/genclosure.h>
#include <OSL/encodedtypes.h>
#include <OSL/hashes.h>
#include <OSL/journal.h>


// Create ustrings for all strings used by the free function renderer services.
// Required to allow the reverse mapping of hash->string to work when processing messages
namespace RS {
namespace Strings {
#define RS_STRDECL(str, var_name) const OSL::ustring var_name { str };
#include "renderservices_strdecls.hpp"
#undef RS_STRDECL
}  // namespace Strings
}  // namespace RS


using namespace OSL;

// anonymous namespace
namespace {

// unique identifier for each closure supported by testshade
enum ClosureIDs {
    EMISSION_ID = 1,
    BACKGROUND_ID,
    DIFFUSE_ID,
    OREN_NAYAR_ID,
    TRANSLUCENT_ID,
    PHONG_ID,
    WARD_ID,
    MICROFACET_ID,
    REFLECTION_ID,
    FRESNEL_REFLECTION_ID,
    REFRACTION_ID,
    TRANSPARENT_ID,
    DEBUG_ID,
    HOLDOUT_ID,
    PARAMETER_TEST_ID,
};

// these structures hold the parameters of each closure type
// they will be contained inside ClosureComponent
struct EmptyParams {};
struct DiffuseParams {
    Vec3 N;
    ustring label;
};
struct OrenNayarParams {
    Vec3 N;
    float sigma;
};
struct PhongParams {
    Vec3 N;
    float exponent;
    ustring label;
};
struct WardParams {
    Vec3 N, T;
    float ax, ay;
};
struct ReflectionParams {
    Vec3 N;
    float eta;
};
struct RefractionParams {
    Vec3 N;
    float eta;
};
struct MicrofacetParams {
    ustring dist;
    Vec3 N, U;
    float xalpha, yalpha, eta;
    int refract;
};
struct DebugParams {
    ustring tag;
};
struct ParameterTestParams {
    int int_param;
    float float_param;
    Color3 color_param;
    Vec3 vector_param;
    ustring string_param;
    int int_array[5];
    Vec3 vector_array[5];
    Color3 color_array[5];
    float float_array[5];
    ustring string_array[5];
    int int_key;
    float float_key;
    Color3 color_key;
    Vec3 vector_key;
    ustring string_key;
};

}  // anonymous namespace

static TypeDesc TypeFloatArray2(TypeDesc::FLOAT, 2);
static TypeDesc TypeFloatArray4(TypeDesc::FLOAT, 4);
static TypeDesc TypeIntArray2(TypeDesc::INT, 2);

void register_closures(OSL::ShadingSystem* shadingsys)
{
    // Describe the memory layout of each closure type to the OSL runtime
    enum { MaxParams = 32 };
    struct BuiltinClosures {
        const char* name;
        int id;
        ClosureParam params[MaxParams];  // upper bound
    };
    BuiltinClosures builtins[] = {
        { "emission", EMISSION_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "background", BACKGROUND_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "diffuse",
          DIFFUSE_ID,
          { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_STRING_KEYPARAM(DiffuseParams, label,
                                    "label"),  // example of custom key param
            CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "oren_nayar",
          OREN_NAYAR_ID,
          { CLOSURE_VECTOR_PARAM(OrenNayarParams, N),
            CLOSURE_FLOAT_PARAM(OrenNayarParams, sigma),
            CLOSURE_FINISH_PARAM(OrenNayarParams) } },
        { "translucent",
          TRANSLUCENT_ID,
          { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "phong",
          PHONG_ID,
          { CLOSURE_VECTOR_PARAM(PhongParams, N),
            CLOSURE_FLOAT_PARAM(PhongParams, exponent),
            CLOSURE_STRING_KEYPARAM(PhongParams, label,
                                    "label"),  // example of custom key param
            CLOSURE_FINISH_PARAM(PhongParams) } },
        { "ward",
          WARD_ID,
          { CLOSURE_VECTOR_PARAM(WardParams, N),
            CLOSURE_VECTOR_PARAM(WardParams, T),
            CLOSURE_FLOAT_PARAM(WardParams, ax),
            CLOSURE_FLOAT_PARAM(WardParams, ay),
            CLOSURE_FINISH_PARAM(WardParams) } },
        { "microfacet",
          MICROFACET_ID,
          { CLOSURE_STRING_PARAM(MicrofacetParams, dist),
            CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
            CLOSURE_VECTOR_PARAM(MicrofacetParams, U),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, xalpha),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, yalpha),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, eta),
            CLOSURE_INT_PARAM(MicrofacetParams, refract),
            CLOSURE_FINISH_PARAM(MicrofacetParams) } },
        { "reflection",
          REFLECTION_ID,
          { CLOSURE_VECTOR_PARAM(ReflectionParams, N),
            CLOSURE_FINISH_PARAM(ReflectionParams) } },
        { "reflection",
          FRESNEL_REFLECTION_ID,
          { CLOSURE_VECTOR_PARAM(ReflectionParams, N),
            CLOSURE_FLOAT_PARAM(ReflectionParams, eta),
            CLOSURE_FINISH_PARAM(ReflectionParams) } },
        { "refraction",
          REFRACTION_ID,
          { CLOSURE_VECTOR_PARAM(RefractionParams, N),
            CLOSURE_FLOAT_PARAM(RefractionParams, eta),
            CLOSURE_FINISH_PARAM(RefractionParams) } },
        { "transparent", TRANSPARENT_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "debug",
          DEBUG_ID,
          { CLOSURE_STRING_PARAM(DebugParams, tag),
            CLOSURE_FINISH_PARAM(DebugParams) } },
        { "holdout", HOLDOUT_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "parameter_test",
          PARAMETER_TEST_ID,
          { CLOSURE_INT_PARAM(ParameterTestParams, int_param),
            CLOSURE_FLOAT_PARAM(ParameterTestParams, float_param),
            CLOSURE_COLOR_PARAM(ParameterTestParams, color_param),
            CLOSURE_VECTOR_PARAM(ParameterTestParams, vector_param),
            CLOSURE_STRING_PARAM(ParameterTestParams, string_param),
            CLOSURE_INT_ARRAY_PARAM(ParameterTestParams, int_array, 5),
            CLOSURE_VECTOR_ARRAY_PARAM(ParameterTestParams, vector_array, 5),
            CLOSURE_COLOR_ARRAY_PARAM(ParameterTestParams, color_array, 5),
            CLOSURE_FLOAT_ARRAY_PARAM(ParameterTestParams, float_array, 5),
            CLOSURE_STRING_ARRAY_PARAM(ParameterTestParams, string_array, 5),
            CLOSURE_INT_KEYPARAM(ParameterTestParams, int_key, "int_key"),
            CLOSURE_FLOAT_KEYPARAM(ParameterTestParams, float_key, "float_key"),
            CLOSURE_COLOR_KEYPARAM(ParameterTestParams, color_key, "color_key"),
            CLOSURE_VECTOR_KEYPARAM(ParameterTestParams, vector_key,
                                    "vector_key"),
            CLOSURE_STRING_KEYPARAM(ParameterTestParams, string_key,
                                    "string_key"),
            CLOSURE_FINISH_PARAM(ParameterTestParams) } }
    };

    for (const auto& b : builtins)
        shadingsys->register_closure(b.name, b.id, b.params, nullptr, nullptr);
}


SimpleRenderer::SimpleRenderer()
{
    Matrix44 M;
    M.makeIdentity();
    // RS::Hashes::perspective is the hash of the string "perspective" 
    // created in renderservices_strdecls.hpp
    camera_params(M, RS::Hashes::perspective, 90.0f, 0.1f, 1000.0f, 256, 256);
     // Set up getters
    m_attr_getters[RS::Hashes::osl_version] = &SimpleRenderer::get_osl_version;
    m_attr_getters[RS::Hashes::camera_resolution]
        = &SimpleRenderer::get_camera_resolution;
    m_attr_getters[RS::Hashes::camera_projection]
        = &SimpleRenderer::get_camera_projection;
    m_attr_getters[RS::Hashes::camera_pixelaspect]
        = &SimpleRenderer::get_camera_pixelaspect;
    m_attr_getters[RS::Hashes::camera_screen_window]
        = &SimpleRenderer::get_camera_screen_window;
    m_attr_getters[RS::Hashes::camera_fov]  = &SimpleRenderer::get_camera_fov;
    m_attr_getters[RS::Hashes::camera_clip] = &SimpleRenderer::get_camera_clip;
    m_attr_getters[RS::Hashes::camera_clip_near]
        = &SimpleRenderer::get_camera_clip_near;
    m_attr_getters[RS::Hashes::camera_clip_far]
        = &SimpleRenderer::get_camera_clip_far;
    m_attr_getters[RS::Hashes::camera_shutter]
        = &SimpleRenderer::get_camera_shutter;
    m_attr_getters[RS::Hashes::camera_shutter_open]
        = &SimpleRenderer::get_camera_shutter_open;
    m_attr_getters[RS::Hashes::camera_shutter_close]
        = &SimpleRenderer::get_camera_shutter_close;
}


SimpleRenderer::~SimpleRenderer() {}

int SimpleRenderer::supports(string_view feature) const
{
    if (m_use_rs_bitcode && feature == "build_attribute_getter")
        return true;
    else if (m_use_rs_bitcode && feature == "build_interpolated_getter")
        return true;
    return false;
}

static void
setup_shaderglobals(ShaderGlobals& sg, ShadingSystem* shadingsys, int x, int y, RenderState& renderState)
{
    // Just zero the whole thing out to start
    memset((char*)&sg, 0, sizeof(ShaderGlobals));

    // Any state data needed by SimpleRenderer or its free function equivalent
    // will need to be passed here the ShaderGlobals.
    sg.renderstate = &renderState;

    // Set "shader" space to be Mshad.  In a real renderer, this may be
    // different for each shader group.
    sg.shader2common = renderState.shader2common;

    // Set "object" space to be Mobj.  In a real renderer, this may be
    // different for each object.
    sg.object2common = renderState.object2common;

    // Just make it look like all shades are the result of 'raytype' rays.
    sg.raytype = renderState.raytype_bit;

    // Set up u,v to vary across the "patch", and also their derivatives.
    // Note that since u & x, and v & y are aligned, we only need to set
    // values for dudx and dvdy, we can use the memset above to have set
    // dvdx and dudy to 0.
    auto uscale = renderState.uscale;
    auto vscale = renderState.vscale;
    auto pixelcenters = renderState.pixel_centers;
    auto xres = renderState.xres;
    auto yres = renderState.yres;
    auto uoffset = renderState.uoffset;
    auto voffset = renderState.voffset;
    auto vary_udxdy = renderState.vary_udxdy;
    auto vary_vdxdy = renderState.vary_vdxdy;
    auto vary_Pdxdy = renderState.vary_Pdxdy;
    
    if (pixelcenters) {
        // Our patch is like an "image" with shading samples at the
        // centers of each pixel.
        sg.u = uscale * (float)(x + 0.5f) / xres + uoffset;
        sg.v = vscale * (float)(y + 0.5f) / yres + voffset;
        if (vary_udxdy) {
            sg.dudx = 1.0f - sg.u;
            sg.dudy = sg.u;
        } else {
            sg.dudx = uscale / xres;
        }
        if (vary_vdxdy) {
            sg.dvdx = 1.0f - sg.v;
            sg.dvdy = sg.v;
        } else {
            sg.dvdy = vscale / yres;
        }
    } else {
        // Our patch is like a Reyes grid of points, with the border
        // samples being exactly on u,v == 0 or 1.
        sg.u = uscale * ((xres == 1) ? 0.5f : (float)x / (xres - 1)) + uoffset;
        sg.v = vscale * ((yres == 1) ? 0.5f : (float)y / (yres - 1)) + voffset;
        if (vary_udxdy) {
            sg.dudx = 1.0f - sg.u;
            sg.dudy = sg.u;
        } else {
            sg.dudx = uscale / std::max(1, xres - 1);
        }
        if (vary_vdxdy) {
            sg.dvdx = 1.0f - sg.v;
            sg.dvdy = sg.v;
        } else {
            sg.dvdy = vscale / std::max(1, yres - 1);
        }
    }

    // Assume that position P is simply (u,v,1), that makes the patch lie
    // on [0,1] at z=1.
    sg.P = Vec3(sg.u, sg.v, 1.0f);
    // Derivatives with respect to x,y
    if (vary_Pdxdy) {
        sg.dPdx = Vec3(1.0f - sg.u, 1.0f - sg.v, sg.u * 0.5);
        sg.dPdy = Vec3(1.0f - sg.v, 1.0f - sg.u, sg.v * 0.5);
    } else {
        sg.dPdx = Vec3(uscale / std::max(1, xres - 1), 0.0f, 0.0f);
        sg.dPdy = Vec3(0.0f, vscale / std::max(1, yres - 1), 0.0f);
    }
    sg.dPdz = Vec3(0.0f, 0.0f, 0.0f);  // just use 0 for volume tangent
    // Tangents of P with respect to surface u,v
    sg.dPdu = Vec3(1.0f, 0.0f, 0.0f);
    sg.dPdv = Vec3(0.0f, 1.0f, 0.0f);
    // That also implies that our normal points to (0,0,1)
    sg.N  = Vec3(0, 0, 1);
    sg.Ng = Vec3(0, 0, 1);

    // Set the surface area of the patch to 1 (which it is).  This is
    // only used for light shaders that call the surfacearea() function.
    sg.surfacearea = 1;
}

// Testshade thread tracking and assignment.
// Not recommended for production renderer but fine for testshade
std::atomic<uint32_t> next_thread_index { 0 };
constexpr uint32_t uninitialized_thread_index = -1;
thread_local uint32_t this_threads_index = uninitialized_thread_index;

static void
save_outputs(SimpleRenderer* rend, ShadingSystem* shadingsys,
             ShadingContext* ctx, int x, int y)
{

    // For each output requested on the command line...
    for (size_t i = 0, e = rend->noutputs(); i < e; ++i) {
        OIIO::ImageBuf* output_bffer_ptr = rend->outputbuf(i);
        if (!output_bffer_ptr)
            continue;

        // Ask for a pointer to the symbol's data, as computed by this shader.
        TypeDesc t;
        auto outputName = rend->outputname(i);
        const void* data = shadingsys->get_symbol(*ctx, outputName, t);
        if (!data)
            continue;

        int nchans = output_bffer_ptr->nchannels();
        if (t.basetype == TypeDesc::FLOAT) {
            output_bffer_ptr->setpixel(x, y, (const float*)data);
            // if (print_outputs) {
            //     print("  {} :", outputvarnames[i]);
            //     for (int c = 0; c < nchans; ++c)
            //         print(" {:g}", ((const float*)data)[c]);
            //     print("\n");
        
        } 
        else if (t.basetype == TypeDesc::INT) 
        {
            // We are outputting an integer variable, so we need to
            float* pixel = OSL_ALLOCA(float, nchans);
            OIIO::convert_pixel_values(TypeDesc::BASETYPE(t.basetype), data,
                                       TypeDesc::FLOAT, pixel, nchans);
            output_bffer_ptr->setpixel(x, y, &pixel[0]);
        }
        // N.B. Drop any outputs that aren't float- or int-based
    }
}

static void shade_region(SimpleRenderer* renderer, RenderState& renderstate, OIIO::ROI roi)
{
    // Request an OSL::PerThreadInfo for this thread.
    auto shadingsys = renderstate.shadingSystem;
    auto shadergroup = renderstate.shaderGroup;

    OSL::PerThreadInfo* thread_info = shadingsys->create_thread_info();

    // Request a shading context so that we can execute the shader.
    // We could get_context/release_context for each shading point,
    // but to save overhead, it's more efficient to reuse a context
    // within a thread.
    ShadingContext* ctx = shadingsys->get_context(thread_info);

    // Set up shader globals and a little test grid of points to shade.
    ShaderGlobals shaderglobals;

    renderstate.raytype_bit = shadingsys->raytype_bit(ustring("camera"));

    for (int y = roi.ybegin; y < roi.yend; ++y)
    {
        int shadeindex = y * renderstate.xres + roi.xbegin;
        for (int x = roi.xbegin; x < roi.xend; ++x, ++shadeindex) 
        {
            // Set up shader globals
            setup_shaderglobals(shaderglobals, shadingsys, x, y, renderstate);

            if (this_threads_index == uninitialized_thread_index) 
            {
                this_threads_index = next_thread_index.fetch_add(1u); 
            }
            
            int thread_index = this_threads_index;

            shadingsys->execute(*ctx, *shadergroup, thread_index,
                                shadeindex, shaderglobals,
                                renderstate.userdata_base_ptr, renderstate.output_base_ptr);

            // Actually run the shader for this point
            /*if (entrylayer_index.empty()) {
                Sole entry point for whole group, default behavior
                shadingsys->execute(*ctx, *shadergroup, thread_index,
                                    shadeindex, shaderglobals,
                                    renderstate.userdata_base_ptr, renderstate.output_base_ptr);
            } else {
                // Explicit list of entries to call in order
                shadingsys->execute_init(*ctx, *shadergroup, thread_index,
                                         shadeindex, shaderglobals,
                                         userdata_base_ptr, output_base_ptr);
                if (entrylayer_symbols.size()) {
                    for (size_t i = 0, e = entrylayer_symbols.size(); i < e;
                         ++i)
                        shadingsys->execute_layer(*ctx, thread_index,
                                                  shadeindex, shaderglobals,
                                                  userdata_base_ptr,
                                                  output_base_ptr,
                                                  entrylayer_symbols[i]);
                } else {
                    for (size_t i = 0, e = entrylayer_index.size(); i < e; ++i)
                        shadingsys->execute_layer(*ctx, thread_index,
                                                  shadeindex, shaderglobals,
                                                  userdata_base_ptr,
                                                  output_base_ptr,
                                                  entrylayer_index[i]);
                }
                shadingsys->execute_cleanup(*ctx);
            }*/
            
            if (renderstate.iteration == renderstate.max_iterations - 1)
            {
                save_outputs(renderer, shadingsys, ctx, x, y);
            }
        }
    }

    shadingsys->release_context(ctx);
    shadingsys->destroy_thread_info(thread_info);
}



void SimpleRenderer::render(int xres, int yres, RenderState& renderState)
{
     OIIO::ROI roi(0, xres, 0, yres);
     OIIO::ImageBufAlgo::parallel_image(roi, renderState.num_threads,
            [this, &renderState](OIIO::ROI sub_roi) -> void { shade_region(this, renderState, sub_roi);} );  
}

void SimpleRenderer::camera_params(const Matrix44& world_to_camera,
                              ustringhash projection, 
                              float hfov, float hither, float yon,
                              int xres, int yres)
{
    m_world_to_camera  = world_to_camera;
    m_projection       = projection;
    m_fov              = hfov;
    m_pixelaspect      = 1.0f;  // hard-coded
    m_hither           = hither;
    m_yon              = yon;
    m_shutter[0]       = 0.0f;
    m_shutter[1]       = 1.0f;  // hard-coded
    float frame_aspect = float(xres) / float(yres) * m_pixelaspect;
    m_screen_window[0] = -frame_aspect;
    m_screen_window[1] = -1.0f;
    m_screen_window[2] = frame_aspect;
    m_screen_window[3] = 1.0f;
    m_xres             = xres;
    m_yres             = yres;
}


/*
    Matrix functions
*/

bool SimpleRenderer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                           TransformationPtr xform, float /*time*/)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *reinterpret_cast<const Matrix44*>(xform);
    return true;
}


bool SimpleRenderer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                           ustringhash from, float /*time*/)
{
    TransformMap::const_iterator found = m_named_xforms.find(from);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}

bool SimpleRenderer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                           TransformationPtr xform)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *(OSL::Matrix44*)xform;
    return true;
}

bool SimpleRenderer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                           ustringhash from)
{
    // SimpleRenderer doesn't understand motion blur, so we never fail
    // on account of time-varying transformations.
    TransformMap::const_iterator found = m_named_xforms.find(from);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}

bool SimpleRenderer::get_inverse_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                                   ustringhash to, float /*time*/)
{
    if (to == OSL::Hashes::camera || to == OSL::Hashes::screen
        || to == OSL::Hashes::NDC || to == RS::Hashes::raster) {
        Matrix44 M = m_world_to_camera;
        if (to == OSL::Hashes::screen || to == OSL::Hashes::NDC
            || to == RS::Hashes::raster) {
            float depthrange = (double)m_yon - (double)m_hither;
            if (m_projection == RS::Hashes::perspective) {
                float tanhalffov = tanf(0.5f * m_fov * M_PI / 180.0);
                Matrix44 camera_to_screen(1 / tanhalffov, 0, 0, 0, 0,
                                          1 / tanhalffov, 0, 0, 0, 0,
                                          m_yon / depthrange, 1, 0, 0,
                                          -m_yon * m_hither / depthrange, 0);
                M = M * camera_to_screen;
            } else {
                Matrix44 camera_to_screen(1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                          1 / depthrange, 0, 0, 0,
                                          -m_hither / depthrange, 1);
                M = M * camera_to_screen;
            }
            if (to == OSL::Hashes::NDC || to == RS::Hashes::raster) {
                float screenleft = -1.0, screenwidth = 2.0;
                float screenbottom = -1.0, screenheight = 2.0;
                Matrix44 screen_to_ndc(1 / screenwidth, 0, 0, 0, 0,
                                       1 / screenheight, 0, 0, 0, 0, 1, 0,
                                       -screenleft / screenwidth,
                                       -screenbottom / screenheight, 0, 1);
                M = M * screen_to_ndc;
                if (to == RS::Hashes::raster) {
                    Matrix44 ndc_to_raster(m_xres, 0, 0, 0, 0, m_yres, 0, 0, 0,
                                           0, 1, 0, 0, 0, 0, 1);
                    M = M * ndc_to_raster;
                }
            }
        }
        result = M;
        return true;
    }

    TransformMap::const_iterator found = m_named_xforms.find(to);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        result.invert();
        return true;
    } else {
        return false;
    }
}




void SimpleRenderer::name_transform(const char* name, const OSL::Matrix44& xform)
{
    std::shared_ptr<Transformation> M(new OSL::Matrix44(xform));
    m_named_xforms[ustringhash(name)] = M;
}

/*
    Attributes
*/
bool SimpleRenderer::get_array_attribute(ShaderGlobals* sg, bool derivatives,
                                    ustringhash object, TypeDesc type,
                                    ustringhash name, int index, void* val)
{
    AttrGetterMap::const_iterator g = m_attr_getters.find(name);
    if (g != m_attr_getters.end()) {
        AttrGetter getter = g->second;
        return (this->*(getter))(sg, derivatives, object, type, name, val);
    }

    // In order to test getattribute(), respond positively to
    // "options"/"blahblah"
    if (object == RS::Hashes::options && name == RS::Hashes::blahblah
        && type == TypeFloat) {
        *(float*)val = 3.14159;
        return true;
    }

    if (object.empty() && name == RS::Hashes::shading_index
        && type == TypeInt) {
        *(int*)val = OSL::get_shade_index(sg);
        return true;
    }

    // If no named attribute was found, allow userdata to bind to the
    // attribute request.
    if (object.empty() && index == -1)
        return get_userdata(derivatives, name, type, sg, val);

    return false;
}



bool
SimpleRenderer::get_attribute(ShaderGlobals* sg, bool derivatives,
                              ustringhash object, TypeDesc type,
                              ustringhash name, void* val)
{
    return get_array_attribute(sg, derivatives, object, type, name, -1, val);
}



bool
SimpleRenderer::get_userdata(bool derivatives, ustringhash name, TypeDesc type,
                             ShaderGlobals* sg, void* val)
{
    // Just to illustrate how this works, respect s and t userdata, filled
    // in with the uv coordinates.  In a real renderer, it would probably
    // look up something specific to the primitive, rather than have hard-
    // coded names.

    if (name == RS::Hashes::s && type == TypeFloat) {
        ((float*)val)[0] = sg->u;
        if (derivatives) {
            ((float*)val)[1] = sg->dudx;
            ((float*)val)[2] = sg->dudy;
        }
        return true;
    }
    if (name == RS::Hashes::t && type == TypeFloat) {
        ((float*)val)[0] = sg->v;
        if (derivatives) {
            ((float*)val)[1] = sg->dvdx;
            ((float*)val)[2] = sg->dvdy;
        }
        return true;
    }
    if (name == RS::Hashes::red && type == TypeFloat && sg->P.x > 0.5f) {
        ((float*)val)[0] = sg->u;
        if (derivatives) {
            ((float*)val)[1] = sg->dudx;
            ((float*)val)[2] = sg->dudy;
        }
        return true;
    }
    if (name == RS::Hashes::green && type == TypeFloat && sg->P.x < 0.5f) {
        ((float*)val)[0] = sg->v;
        if (derivatives) {
            ((float*)val)[1] = sg->dvdx;
            ((float*)val)[2] = sg->dvdy;
        }
        return true;
    }
    if (name == RS::Hashes::blue && type == TypeFloat
        && ((static_cast<int>(sg->P.y * 12) % 2) == 0)) {
        ((float*)val)[0] = 1.0f - sg->u;
        if (derivatives) {
            ((float*)val)[1] = -sg->dudx;
            ((float*)val)[2] = -sg->dudy;
        }
        return true;
    }

    if (const OIIO::ParamValue* p = userdata.find_pv(ustring_from(name), type)) 
    {
        size_t size = p->type().size();

        if (p->type() == TypeDesc::STRING) {
            const ustringhash* uh_data = reinterpret_cast<const ustringhash*>(p->data());
            memcpy(val, uh_data, size);
        } else {
            memcpy(val, p->data(), size);
        }
        if (derivatives)
            memset((char*)val + size, 0, 2 * size);
        return true;
    }

    return false;
}


void
SimpleRenderer::build_attribute_getter(
    const ShaderGroup& group, bool is_object_lookup, const ustring* object_name,
    const ustring* attribute_name, bool is_array_lookup, const int* array_index,
    TypeDesc type, bool derivatives, AttributeGetterSpec& spec)
{
    static const OIIO::ustring rs_get_attribute_constant_int(
        "rs_get_attribute_constant_int");
    static const OIIO::ustring rs_get_attribute_constant_int2(
        "rs_get_attribute_constant_int2");
    static const OIIO::ustring rs_get_attribute_constant_int3(
        "rs_get_attribute_constant_int3");
    static const OIIO::ustring rs_get_attribute_constant_int4(
        "rs_get_attribute_constant_int4");

    static const OIIO::ustring rs_get_attribute_constant_float(
        "rs_get_attribute_constant_float");
    static const OIIO::ustring rs_get_attribute_constant_float2(
        "rs_get_attribute_constant_float2");
    static const OIIO::ustring rs_get_attribute_constant_float3(
        "rs_get_attribute_constant_float3");
    static const OIIO::ustring rs_get_attribute_constant_float4(
        "rs_get_attribute_constant_float4");

    static const OIIO::ustring rs_get_shade_index("rs_get_shade_index");

    static const OIIO::ustring rs_get_attribute("rs_get_attribute");

    if (m_use_rs_bitcode) {
        // For demonstration purposes we show how to build functions taking
        // advantage of known compile time information. Here we simply select
        // which function to call based on what we know at this point.

        if (object_name && object_name->empty() && attribute_name) {
            if (const OIIO::ParamValue* p = userdata.find_pv(*attribute_name,
                                                             type)) {
                if (p->type().basetype == OIIO::TypeDesc::INT) {
                    if (p->type().aggregate == 1) {
                        spec.set(rs_get_attribute_constant_int,
                                 ((int*)p->data())[0]);
                        return;
                    } else if (p->type().aggregate == 2) {
                        spec.set(rs_get_attribute_constant_int2,
                                 ((int*)p->data())[0], ((int*)p->data())[1]);
                        return;
                    } else if (p->type().aggregate == 3) {
                        spec.set(rs_get_attribute_constant_int3,
                                 ((int*)p->data())[0], ((int*)p->data())[1],
                                 ((int*)p->data())[2]);
                        return;
                    } else if (p->type().aggregate == 4) {
                        spec.set(rs_get_attribute_constant_int4,
                                 ((int*)p->data())[0], ((int*)p->data())[1],
                                 ((int*)p->data())[2], ((int*)p->data())[3]);
                        return;
                    }
                } else if (p->type().basetype == OIIO::TypeDesc::FLOAT) {
                    if (p->type().aggregate == 1) {
                        spec.set(rs_get_attribute_constant_float,
                                 ((float*)p->data())[0],
                                 AttributeSpecBuiltinArg::Derivatives);
                        return;
                    } else if (p->type().aggregate == 2) {
                        spec.set(rs_get_attribute_constant_float2,
                                 ((float*)p->data())[0], ((float*)p->data())[1],
                                 AttributeSpecBuiltinArg::Derivatives);
                        return;
                    } else if (p->type().aggregate == 3) {
                        spec.set(rs_get_attribute_constant_float3,
                                 ((float*)p->data())[0], ((float*)p->data())[1],
                                 ((float*)p->data())[2],
                                 AttributeSpecBuiltinArg::Derivatives);
                        return;
                    } else if (p->type().aggregate == 4) {
                        spec.set(rs_get_attribute_constant_float4,
                                 ((float*)p->data())[0], ((float*)p->data())[1],
                                 ((float*)p->data())[2], ((float*)p->data())[3],
                                 AttributeSpecBuiltinArg::Derivatives);
                        return;
                    }
                }
            }
        }

        if (object_name && *object_name == ustring("options") && attribute_name
            && *attribute_name == ustring("blahblah")
            && type == OSL::TypeFloat) {
            spec.set(rs_get_attribute_constant_float, 3.14159f,
                     AttributeSpecBuiltinArg::Derivatives);
        } else if (!is_object_lookup && attribute_name
                   && *attribute_name == ustring("shading:index")
                   && type == OSL::TypeInt) {
            spec.set(rs_get_shade_index,
                     AttributeSpecBuiltinArg::OpaqueExecutionContext);
        } else {
            spec.set(rs_get_attribute,
                     AttributeSpecBuiltinArg::OpaqueExecutionContext,
                     AttributeSpecBuiltinArg::ObjectName,
                     AttributeSpecBuiltinArg::AttributeName,
                     AttributeSpecBuiltinArg::Type,
                     AttributeSpecBuiltinArg::Derivatives,
                     AttributeSpecBuiltinArg::ArrayIndex);
        }
    }
}


void
SimpleRenderer::build_interpolated_getter(const ShaderGroup& group,
                                          const ustring& param_name,
                                          TypeDesc type, bool derivatives,
                                          InterpolatedGetterSpec& spec)
{
    static const OIIO::ustring rs_get_interpolated_s("rs_get_interpolated_s");
    static const OIIO::ustring rs_get_interpolated_t("rs_get_interpolated_t");
    static const OIIO::ustring rs_get_interpolated_red(
        "rs_get_interpolated_red");
    static const OIIO::ustring rs_get_interpolated_green(
        "rs_get_interpolated_green");
    static const OIIO::ustring rs_get_interpolated_blue(
        "rs_get_interpolated_blue");
    static const OIIO::ustring rs_get_interpolated_test(
        "rs_get_interpolated_test");

    static const OIIO::ustring rs_get_attribute_constant_int(
        "rs_get_attribute_constant_int");
    static const OIIO::ustring rs_get_attribute_constant_int2(
        "rs_get_attribute_constant_int2");
    static const OIIO::ustring rs_get_attribute_constant_int3(
        "rs_get_attribute_constant_int3");
    static const OIIO::ustring rs_get_attribute_constant_int4(
        "rs_get_attribute_constant_int4");

    static const OIIO::ustring rs_get_attribute_constant_float(
        "rs_get_attribute_constant_float");
    static const OIIO::ustring rs_get_attribute_constant_float2(
        "rs_get_attribute_constant_float2");
    static const OIIO::ustring rs_get_attribute_constant_float3(
        "rs_get_attribute_constant_float3");
    static const OIIO::ustring rs_get_attribute_constant_float4(
        "rs_get_attribute_constant_float4");

    if (param_name == RS::Hashes::s && type == OIIO::TypeFloat) {
        spec.set(rs_get_interpolated_s,
                 InterpolatedSpecBuiltinArg::OpaqueExecutionContext,
                 InterpolatedSpecBuiltinArg::Derivatives);
    } else if (param_name == RS::Hashes::t && type == OIIO::TypeFloat) {
        spec.set(rs_get_interpolated_t,
                 InterpolatedSpecBuiltinArg::OpaqueExecutionContext,
                 InterpolatedSpecBuiltinArg::Derivatives);
    } else if (param_name == RS::Hashes::red && type == OIIO::TypeFloat) {
        spec.set(rs_get_interpolated_red,
                 InterpolatedSpecBuiltinArg::OpaqueExecutionContext,
                 InterpolatedSpecBuiltinArg::Derivatives);
    } else if (param_name == RS::Hashes::green && type == OIIO::TypeFloat) {
        spec.set(rs_get_interpolated_green,
                 InterpolatedSpecBuiltinArg::OpaqueExecutionContext,
                 InterpolatedSpecBuiltinArg::Derivatives);
    } else if (param_name == RS::Hashes::blue && type == OIIO::TypeFloat) {
        spec.set(rs_get_interpolated_blue,
                 InterpolatedSpecBuiltinArg::OpaqueExecutionContext,
                 InterpolatedSpecBuiltinArg::Derivatives);
    } else if (param_name == RS::Hashes::test && type == OIIO::TypeFloat) {
        spec.set(rs_get_interpolated_test);
    } else if (const OIIO::ParamValue* p = userdata.find_pv(param_name, type)) {
        if (p->type().basetype == OIIO::TypeDesc::INT) {
            if (p->type().aggregate == 1) {
                spec.set(rs_get_attribute_constant_int, ((int*)p->data())[0]);
                return;
            } else if (p->type().aggregate == 2) {
                spec.set(rs_get_attribute_constant_int2, ((int*)p->data())[0],
                         ((int*)p->data())[1]);
                return;
            } else if (p->type().aggregate == 3) {
                spec.set(rs_get_attribute_constant_int3, ((int*)p->data())[0],
                         ((int*)p->data())[1], ((int*)p->data())[2]);
                return;
            } else if (p->type().aggregate == 4) {
                spec.set(rs_get_attribute_constant_int4, ((int*)p->data())[0],
                         ((int*)p->data())[1], ((int*)p->data())[2],
                         ((int*)p->data())[3]);
                return;
            }
        } else if (p->type().basetype == OIIO::TypeDesc::FLOAT) {
            if (p->type().aggregate == 1) {
                spec.set(rs_get_attribute_constant_float,
                         ((float*)p->data())[0],
                         InterpolatedSpecBuiltinArg::Derivatives);
                return;
            } else if (p->type().aggregate == 2) {
                spec.set(rs_get_attribute_constant_float2,
                         ((float*)p->data())[0], ((float*)p->data())[1],
                         InterpolatedSpecBuiltinArg::Derivatives);
                return;
            } else if (p->type().aggregate == 3) {
                spec.set(rs_get_attribute_constant_float3,
                         ((float*)p->data())[0], ((float*)p->data())[1],
                         ((float*)p->data())[2],
                         InterpolatedSpecBuiltinArg::Derivatives);
                return;
            } else if (p->type().aggregate == 4) {
                spec.set(rs_get_attribute_constant_float4,
                         ((float*)p->data())[0], ((float*)p->data())[1],
                         ((float*)p->data())[2], ((float*)p->data())[3],
                         InterpolatedSpecBuiltinArg::Derivatives);
                return;
            }
        }
    }
}

/*
    Trace
*/

bool SimpleRenderer::trace(TraceOpt& options, ShaderGlobals* sg, const OSL::Vec3& P,
                      const OSL::Vec3& dPdx, const OSL::Vec3& dPdy,
                      const OSL::Vec3& R, const OSL::Vec3& dRdx,
                      const OSL::Vec3& dRdy)
{
    // Don't do real ray tracing, just
    // use source and direction to alter hit results
    // so they are repeatable values for testsuite.
    float dot_val = P.dot(R);

    if ((sg->u) / dot_val > 0.5) {
        return true;  //1 in batched
    } else {
        return false;
    }
}

bool
SimpleRenderer::getmessage(ShaderGlobals* sg, ustringhash source_,
                           ustringhash name_, TypeDesc type, void* val,
                           bool derivatives)
{
    ustring source = ustring_from(source_);
    ustring name   = ustring_from(name_);
    OSL_ASSERT(source == ustring("trace"));
    // Don't have any real ray tracing results
    // so just fill in some repeatable values for testsuite
    if (sg->u > 0.5) {
        if (name == ustring("hitdist")) {
            if (type == TypeFloat) {
                *reinterpret_cast<float*>(val) = 0.5f;
            }
        }
        if (name == ustring("hit")) {
            if (type == TypeInt) {
                *reinterpret_cast<int*>(val) = 1;
            }
        }
        if (name == ustring("geom:name")) {
            if (type == TypeString) {
                *reinterpret_cast<ustring*>(val) = ustringhash("teapot");
            }
        }
        if (name == ustring("N")) {
            if (type == TypeNormal) {
                *reinterpret_cast<Vec3*>(val) = Vec3(1.0 - sg->v, 0.25,
                                                     1.0 - sg->u);
            } else {
                OSL_ASSERT(0 && "Oops");
            }
        }
        return true;  //1 in batched

    } else {
        if (name == ustring("hit")) {
            if (type == TypeInt) {
                *reinterpret_cast<int*>(val) = 0;
            }
        }
        return false;
    }
}

/*
    IO
*/
bool SimpleRenderer::add_output(string_view varname_, string_view filename,
                           TypeDesc datatype, int nchannels)
{
    // FIXME: use name to figure out
    ustring varname_us(varname_);
    OIIO::ImageSpec spec(m_xres, m_yres, nchannels, datatype);
    m_outputvars.emplace_back(varname_us);
    m_outputbufs.emplace_back(
        new OIIO::ImageBuf(filename, spec, OIIO::InitializePixels::Yes));
    return true;
}

void SimpleRenderer::export_state(RenderState& state) const
{
    state.xres   = m_xres;
    state.yres   = m_yres;
    state.fov    = m_fov;
    state.hither = m_hither;
    state.yon    = m_yon;

    state.world_to_camera = OSL::Matrix44(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                          0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                          0.0, 1.0);
    //perspective is not  a member of StringParams (i.e not in strdecls.h)
    state.projection  = RS::Hashes::perspective;
    state.pixelaspect = m_pixelaspect;
    std::copy_n(m_screen_window, 4, state.screen_window);
    std::copy_n(m_shutter, 2, state.shutter);
}

void SimpleRenderer::errorfmt(OSL::ShaderGlobals* sg,
                         OSL::ustringhash fmt_specification, int32_t arg_count,
                         const EncodedType* arg_types, uint32_t arg_values_size,
                         uint8_t* argValues)
{
    RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_errorfmt(OSL::get_thread_index(sg), OSL::get_shade_index(sg),
                       fmt_specification, arg_count, arg_types, arg_values_size,
                       argValues);
}

void SimpleRenderer::warningfmt(OSL::ShaderGlobals* sg,
                           OSL::ustringhash fmt_specification,
                           int32_t arg_count, const EncodedType* arg_types,
                           uint32_t arg_values_size, uint8_t* argValues)
{
    RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_warningfmt(OSL::get_max_warnings_per_thread(sg),
                         OSL::get_thread_index(sg), OSL::get_shade_index(sg),
                         fmt_specification, arg_count, arg_types,
                         arg_values_size, argValues);
}


void SimpleRenderer::printfmt(OSL::ShaderGlobals* sg,
                         OSL::ustringhash fmt_specification, int32_t arg_count,
                         const EncodedType* arg_types, uint32_t arg_values_size,
                         uint8_t* argValues)
{
    RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_printfmt(OSL::get_thread_index(sg), OSL::get_shade_index(sg),
                       fmt_specification, arg_count, arg_types, arg_values_size,
                       argValues);
}

void SimpleRenderer::filefmt(OSL::ShaderGlobals* sg, OSL::ustringhash filename_hash,
                        OSL::ustringhash fmt_specification, int32_t arg_count,
                        const EncodedType* arg_types, uint32_t arg_values_size,
                        uint8_t* argValues)
{
    RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_filefmt(OSL::get_thread_index(sg), OSL::get_shade_index(sg),
                      filename_hash, fmt_specification, arg_count, arg_types,
                      arg_values_size, argValues);
}



/*
    Utilities
*/
bool SimpleRenderer::get_osl_version(ShaderGlobals* /*sg*/, bool /*derivs*/,
                                ustringhash /*object*/, TypeDesc type,
                                ustringhash /*name*/, void* val)
{
    if (type == TypeInt) {
        ((int*)val)[0] = OSL_VERSION;
        return true;
    }
    return false;
}


bool SimpleRenderer::get_camera_resolution(ShaderGlobals* /*sg*/, bool /*derivs*/,
                                      ustringhash /*object*/, TypeDesc type,
                                      ustringhash /*name*/, void* val)
{
    if (type == TypeIntArray2) {
        ((int*)val)[0] = m_xres;
        ((int*)val)[1] = m_yres;
        return true;
    }
    return false;
}


bool SimpleRenderer::get_camera_projection(ShaderGlobals* /*sg*/, bool /*derivs*/,
                                      ustringhash /*object*/, TypeDesc type,
                                      ustringhash /*name*/, void* val)
{
    if (type == TypeString) {
        ((ustringhash*)val)[0] = m_projection;
        return true;
    }
    return false;
}


bool SimpleRenderer::get_camera_fov(ShaderGlobals* /*sg*/, bool derivs,
                               ustringhash /*object*/, TypeDesc type,
                               ustringhash /*name*/, void* val)
{
    // N.B. in a real renderer, this may be time-dependent
    if (type == TypeFloat) {
        ((float*)val)[0] = m_fov;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool SimpleRenderer::get_camera_pixelaspect(ShaderGlobals* /*sg*/, bool derivs,
                                       ustringhash /*object*/, TypeDesc type,
                                       ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_pixelaspect;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool SimpleRenderer::get_camera_clip(ShaderGlobals* /*sg*/, bool derivs,
                                ustringhash /*object*/, TypeDesc type,
                                ustringhash /*name*/, void* val)
{
    if (type == TypeFloatArray2) {
        ((float*)val)[0] = m_hither;
        ((float*)val)[1] = m_yon;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool SimpleRenderer::get_camera_clip_near(ShaderGlobals* /*sg*/, bool derivs,
                                     ustringhash /*object*/, TypeDesc type,
                                     ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_hither;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_clip_far(ShaderGlobals* /*sg*/, bool derivs,
                                    ustringhash /*object*/, TypeDesc type,
                                    ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_yon;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}



bool SimpleRenderer::get_camera_shutter(ShaderGlobals* /*sg*/, bool derivs,
                                   ustringhash /*object*/, TypeDesc type,
                                   ustringhash /*name*/, void* val)
{
    if (type == TypeFloatArray2) {
        ((float*)val)[0] = m_shutter[0];
        ((float*)val)[1] = m_shutter[1];
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool SimpleRenderer::get_camera_shutter_open(ShaderGlobals* /*sg*/, bool derivs,
                                        ustringhash /*object*/, TypeDesc type,
                                        ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_shutter[0];
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool SimpleRenderer::get_camera_shutter_close(ShaderGlobals* /*sg*/, bool derivs,
                                         ustringhash /*object*/, TypeDesc type,
                                         ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_shutter[1];
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool SimpleRenderer::get_camera_screen_window(ShaderGlobals* /*sg*/, bool derivs,
                                         ustringhash /*object*/, TypeDesc type,
                                         ustringhash /*name*/, void* val)
{
    // N.B. in a real renderer, this may be time-dependent
    if (type == TypeFloatArray4) {
        ((float*)val)[0] = m_screen_window[0];
        ((float*)val)[1] = m_screen_window[1];
        ((float*)val)[2] = m_screen_window[2];
        ((float*)val)[3] = m_screen_window[3];
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}

OIIO::ParamValue*
SimpleRenderer::find_attribute(string_view name, TypeDesc searchtype,
                               bool casesensitive)
{
    auto iter = options.find(name, searchtype, casesensitive);
    if (iter != options.end())
        return &(*iter);
    return nullptr;
}



const OIIO::ParamValue*
SimpleRenderer::find_attribute(string_view name, TypeDesc searchtype,
                               bool casesensitive) const
{
    auto iter = options.find(name, searchtype, casesensitive);
    if (iter != options.end())
        return &(*iter);
    return nullptr;
}


void SimpleRenderer::attribute(OIIO::string_view name, OIIO::TypeDesc type, const void* value)
{
    if (name.empty())  // Guard against bogus empty names
        return;
    // Don't allow duplicates
    auto f = find_attribute(name);
    if (!f) {
        options.resize(options.size() + 1);
        f = &options.back();
    }
    f->init(name, type, 1, value);
}