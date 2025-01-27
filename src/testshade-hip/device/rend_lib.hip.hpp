#ifndef REND_LIB_HIP_HPP
#define REND_LIB_HIP_HPP

#include <OSL/oslconfig.h>
#include <OSL/hashes.h>
#include <OSL/oslexec.h>

#define HDSTR(cstr) (*((OSL::ustringhash*)&cstr))
// device side definition of the ShaderGlobals struct

struct ShadingContextHIP {};

namespace  {

struct ShaderGlobals {
    OSL::Vec3 P, dPdx, dPdy;
    OSL::Vec3 dPdz;
    OSL::Vec3 I, dIdx, dIdy;
    OSL::Vec3 N;
    OSL::Vec3 Ng;
    float u, dudx, dudy;
    float v, dvdx, dvdy;
    OSL::Vec3 dPdu, dPdv;
    float time;
    float dtime;
    OSL::Vec3 dPdtime;
    OSL::Vec3 Ps, dPsdx, dPsdy;
    void* renderstate;
    void* tracedata;
    void* objdata;
    void* context;
    void* shadingStateUniform;
    int thread_index;
    int shade_index;
    void* renderer;
    void* object2common;
    void* shader2common;
    void* Ci;
    float surfacearea;
    int raytype;
    int flipHandedness;
    int backfacing;
};


} // namespace OSL_HIP


#endif // REND_LIB_HIP_HPP