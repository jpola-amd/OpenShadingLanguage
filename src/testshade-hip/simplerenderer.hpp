#ifndef SIMPLERENDERER_HPP
#define SIMPLERENDERER_HPP

#include <vector>

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/ustring.h>

#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>

#include "renderstate.hpp"


void
register_closures(OSL::ShadingSystem* shadingsys);



class SimpleRenderer : public OSL::RendererServices {
    // Keep implementation in sync with rs_simplerend.cpp
    //template<int> friend class BatchedSimpleRenderer;

public:
    typedef OSL::Matrix44 Transformation;

    SimpleRenderer();
    virtual ~SimpleRenderer();

    int supports(OIIO::string_view feature) const override;
    bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::TransformationPtr xform, float time) override;
    bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::ustringhash from, float time) override;
    bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::TransformationPtr xform) override;
    bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::ustringhash from) override;
    bool get_inverse_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::ustringhash to, float time) override;

    void name_transform(const char* name, const Transformation& xform);

    bool get_array_attribute(OSL::ShaderGlobals* sg, bool derivatives,
                             OSL::ustringhash object, OIIO::TypeDesc type,
                             OSL::ustringhash name, int index,
                             void* val) override;
    bool get_attribute(OSL::ShaderGlobals* sg, bool derivatives,
                       OSL::ustringhash object, OIIO::TypeDesc type,
                       OSL::ustringhash name, void* val) override;
    bool get_userdata(bool derivatives, OSL::ustringhash name,
                      OIIO::TypeDesc type, OSL::ShaderGlobals* sg,
                      void* val) override;

    void build_attribute_getter(const OSL::ShaderGroup& group,
                                bool is_object_lookup,
                                const OIIO::ustring* object_name,
                                const OIIO::ustring* attribute_name,
                                bool is_array_lookup, const int* array_index,
                                OIIO::TypeDesc type, bool derivatives,
                                OSL::AttributeGetterSpec& spec) override;

    void build_interpolated_getter(const OSL::ShaderGroup& group,
                                   const OIIO::ustring& param_name,
                                   OIIO::TypeDesc type, bool derivatives,
                                   OSL::InterpolatedGetterSpec& spec) override;

    bool trace(TraceOpt& options, OSL::ShaderGlobals* sg, const OSL::Vec3& P,
               const OSL::Vec3& dPdx, const OSL::Vec3& dPdy, const OSL::Vec3& R,
               const OSL::Vec3& dRdx, const OSL::Vec3& dRdy) override;

    bool getmessage(OSL::ShaderGlobals* sg, OSL::ustringhash source,
                    OSL::ustringhash name, OIIO::TypeDesc type, void* val,
                    bool derivatives) override;

    void errorfmt(OSL::ShaderGlobals* sg, OSL::ustringhash fmt_specification,
                  int32_t count, const OSL::EncodedType* argTypes,
                  uint32_t argValuesSize, uint8_t* argValues) override;
    void warningfmt(OSL::ShaderGlobals* sg, OSL::ustringhash fmt_specification,
                    int32_t count, const OSL::EncodedType* argTypes,
                    uint32_t argValuesSize, uint8_t* argValues) override;
    void printfmt(OSL::ShaderGlobals* sg, OSL::ustringhash fmt_specification,
                  int32_t count, const OSL::EncodedType* argTypes,
                  uint32_t argValuesSize, uint8_t* argValues) override;
    void filefmt(OSL::ShaderGlobals* sg, OSL::ustringhash filename_hash,
                 OSL::ustringhash fmt_specification, int32_t arg_count,
                 const OSL::EncodedType* argTypes, uint32_t argValuesSize,
                 uint8_t* argValues) override;

    // Set and get renderer attributes/options
    void attribute(OIIO::string_view name, OIIO::TypeDesc type,
                   const void* value);
    void attribute(OIIO::string_view name, int value)
    {
        attribute(name, OIIO::TypeDesc::INT, &value);
    }
    void attribute(OIIO::string_view name, float value)
    {
        attribute(name, OIIO::TypeDesc::FLOAT, &value);
    }
    void attribute(OIIO::string_view name, OIIO::string_view value)
    {
        std::string valstr(value);
        const char* s = valstr.c_str();
        attribute(name, OIIO::TypeDesc::STRING, &s);
    }
    OIIO::ParamValue*
    find_attribute(OIIO::string_view name,
                   OIIO::TypeDesc searchtype = OIIO::TypeUnknown,
                   bool casesensitive        = false);
    const OIIO::ParamValue*
    find_attribute(OIIO::string_view name,
                   OIIO::TypeDesc searchtype = OIIO::TypeUnknown,
                   bool casesensitive        = false) const;

    // Super simple camera and display parameters.  Many options not
    // available, no motion blur, etc.
    void camera_params(const OSL::Matrix44& world_to_camera,
                       OSL::ustringhash projection, float hfov, float hither,
                       float yon, int xres, int yres);

    virtual bool add_output(OIIO::string_view varname,
                            OIIO::string_view filename,
                            OIIO::TypeDesc datatype = OIIO::TypeFloat,
                            int nchannels           = 3);

    // Get the output ImageBuf by index
    OIIO::ImageBuf* outputbuf(int index)
    {
        return index < (int)m_outputbufs.size() ? m_outputbufs[index].get()
                                                : nullptr;
    }
    // Get the output ImageBuf by name
    OIIO::ImageBuf* outputbuf(OIIO::string_view name)
    {
        for (size_t i = 0; i < m_outputbufs.size(); ++i)
            if (m_outputvars[i] == name)
                return m_outputbufs[i].get();
        return nullptr;
    }
    OIIO::ustring outputname(int index) const { return m_outputvars[index]; }
    size_t noutputs() const { return m_outputbufs.size(); }

    virtual void init_shadingsys(OSL::ShadingSystem* ss) { shadingsys = ss; }
    virtual void export_state(RenderState&) const;
    virtual void prepare_render() {}
    virtual void warmup() {}
    virtual void render(int /*xres*/, int /*yres*/) {}
    virtual void clear() { m_shaders.clear(); }

    // After render, get the pixel data into the output buffers, if
    // they aren't already.
    virtual void finalize_pixel_buffer() {}

    void use_rs_bitcode(bool enabled) { m_use_rs_bitcode = enabled; }

    static void register_JIT_Global_Variables();

    // ShaderGroupRef storage
    std::vector<OSL::ShaderGroupRef>& shaders() { return m_shaders; }

    OIIO::ErrorHandler& errhandler() const { return *m_errhandler; }

    OSL::ShadingSystem* shadingsys = nullptr;
    OIIO::ParamValueList options;
    OIIO::ParamValueList userdata;


private:
    OSL::Matrix44 m_world_to_camera;
    OSL::ustringhash m_projection;
    float m_fov;
    float m_pixelaspect { 1.0f };
    // near clipplane
    float m_hither;
    // far clipplane
    float m_yon;
    float m_shutter[2] = { 0.0f, 1.0f };
    float m_screen_window[4];
    int m_xres;
    int m_yres;

    std::vector<OSL::ShaderGroupRef> m_shaders;
    std::vector<OIIO::ustring> m_outputvars;
    std::vector<std::shared_ptr<OIIO::ImageBuf>> m_outputbufs;
    std::unique_ptr<OIIO::ErrorHandler> m_errhandler { new OIIO::ErrorHandler };
    bool m_use_rs_bitcode = false;

    // Named transforms
    typedef std::map<OSL::ustringhash, std::shared_ptr<Transformation>>
        TransformMap;
    TransformMap m_named_xforms;

    // Attribute and userdata retrieval -- for fast dispatch, use a hash
    // table to map attribute names to functions that retrieve them. We
    // imagine this to be fairly quick, but for a performance-critical
    // renderer, we would encourage benchmarking various methods and
    // alternate data structures.
    typedef bool (SimpleRenderer::*AttrGetter)(
        OSL::ShaderGlobals* sg, bool derivs, OSL::ustringhash object,
        OIIO::TypeDesc type, OSL::ustringhash name, void* val);
    typedef std::unordered_map<OSL::ustringhash, AttrGetter> AttrGetterMap;
    AttrGetterMap m_attr_getters;

    // Attribute getters
    bool get_osl_version(OSL::ShaderGlobals* sg, bool derivs,
                         OSL::ustringhash object, OIIO::TypeDesc type,
                         OSL::ustringhash name, void* val);
    bool get_camera_resolution(OSL::ShaderGlobals* sg, bool derivs,
                               OSL::ustringhash object, OIIO::TypeDesc type,
                               OSL::ustringhash name, void* val);
    bool get_camera_projection(OSL::ShaderGlobals* sg, bool derivs,
                               OSL::ustringhash object, OIIO::TypeDesc type,
                               OSL::ustringhash name, void* val);
    bool get_camera_fov(OSL::ShaderGlobals* sg, bool derivs,
                        OSL::ustringhash object, OIIO::TypeDesc type,
                        OSL::ustringhash name, void* val);
    bool get_camera_pixelaspect(OSL::ShaderGlobals* sg, bool derivs,
                                OSL::ustringhash object, OIIO::TypeDesc type,
                                OSL::ustringhash name, void* val);
    bool get_camera_clip(OSL::ShaderGlobals* sg, bool derivs,
                         OSL::ustringhash object, OIIO::TypeDesc type,
                         OSL::ustringhash name, void* val);
    bool get_camera_clip_near(OSL::ShaderGlobals* sg, bool derivs,
                              OSL::ustringhash object, OIIO::TypeDesc type,
                              OSL::ustringhash name, void* val);
    bool get_camera_clip_far(OSL::ShaderGlobals* sg, bool derivs,
                             OSL::ustringhash object, OIIO::TypeDesc type,
                             OSL::ustringhash name, void* val);
    bool get_camera_shutter(OSL::ShaderGlobals* sg, bool derivs,
                            OSL::ustringhash object, OIIO::TypeDesc type,
                            OSL::ustringhash name, void* val);
    bool get_camera_shutter_open(OSL::ShaderGlobals* sg, bool derivs,
                                 OSL::ustringhash object, OIIO::TypeDesc type,
                                 OSL::ustringhash name, void* val);
    bool get_camera_shutter_close(OSL::ShaderGlobals* sg, bool derivs,
                                  OSL::ustringhash object, OIIO::TypeDesc type,
                                  OSL::ustringhash name, void* val);
    bool get_camera_screen_window(OSL::ShaderGlobals* sg, bool derivs,
                                  OSL::ustringhash object, OIIO::TypeDesc type,
                                  OSL::ustringhash name, void* val);
};



#endif  // SIMPLERENDERER_HPP