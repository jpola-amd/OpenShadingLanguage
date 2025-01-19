#ifndef RENDERSTATE_HPP
#define RENDERSTATE_HPP

#pragma once

#include <OSL/hashes.h>
#include <OSL/oslconfig.h>

// All the the state free functions in rs_simplerend.cpp will need to do their job
// NOTE:  Additional data is here that will be used by rs_simplerend.cpp in future PR's
//        procedurally generating ShaderGlobals.
struct RenderState {
    int xres;
    int yres;
    OSL::Matrix44 world_to_camera;
    OSL::ustringhash projection;
    float pixelaspect;
    float screen_window[4];
    float shutter[2];
    float fov;
    float hither;
    float yon;
    void* journal_buffer;
    int max_iterations;
    int iteration;
    int num_threads;
    OSL::ShaderGroup* shaderGroup;
    OSL::ShadingSystem* shadingSystem;
    int raytype_bit {0};
    bool pixel_centers {false};
    OSL::TransformationPtr object2common;
    OSL::TransformationPtr shader2common;

    float uscale {1.0f};
    float uoffset {0.0f};
    float vscale {1.0f};
    float voffset {0.0f};

    bool vary_Pdxdy {false };
    bool vary_udxdy {false };
    bool vary_vdxdy {false };

    char* output_base_ptr {nullptr};
    char* userdata_base_ptr {nullptr};
};


// Create constexpr hashes for all strings used by the free function renderer services.
// NOTE:  Actually ustring's should also be instantiated in host code someplace as well
// to allow the reverse mapping of hash->string to work when processing messages
namespace RS {
namespace {
namespace Hashes {
#define RS_STRDECL(str, var_name) \
    constexpr OSL::ustringhash var_name(OSL::strhash(str));
#include "renderservices_strdecls.hpp"
#undef RS_STRDECL
};  //namespace Hashes
}  // unnamed namespace
};  //namespace RS


#endif // RENDERSTATE_HPP