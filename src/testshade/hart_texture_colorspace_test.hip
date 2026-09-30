// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <amd/hart/hart_device.h>
#include <hip/hip_runtime.h>

#include <OSL/rs_free_function.h>

#include "hart_texture_colorspace_params.h"

#if !OSL_ARNOLD_COMPAT
#    error This fixture requires the Arnold texture colorspace ABI
#endif

using namespace OSL;
using namespace texture_colorspace_test;

extern "C" __constant__ Params osl_hart_render_params;

namespace {

struct State {
    Result* result;
    uint64_t texture;
};

}  // namespace



OSL_RSOP OSL_HOSTDEVICE void
rs_hart_range_error(OpaqueExecContextPtr ec, int, int)
{
    const auto* sg = static_cast<const ShaderGlobals*>(ec);
    auto* state    = static_cast<State*>(sg->renderstate);
    if (state)
        state->result->errors |= invalid_call;
}



// Deliberately a bounded test renderer, not an OCIO implementation. One HIP
// texture is sampled with a runtime (source handle, colorspace hash) selection.
OSL_RSOP OSL_HOSTDEVICE bool
rs_texture(OpaqueExecContextPtr ec, ustringhash filename,
           ustringhash color_space, TextureSystem::TextureHandle* handle,
           TextureSystem::Perthread*, TextureOpt& options, float s, float t,
           float dsdx, float dtdx, float dsdy, float dtdy, int nchannels,
           float* result, float* dresultds, float* dresultdt,
           ustringhash* errormessage)
{
    constexpr ustringhash raw_space(strhash("raw"));
    constexpr ustringhash srgb_space(strhash("sRGB"));
    constexpr ustringhash source(strhash("hart-colorspace-fixed"));
    constexpr ustringhash bad_source(
        strhash("HART texture test: unknown source colorspace"));
    const auto* sg = static_cast<const ShaderGlobals*>(ec);
    auto* state    = static_cast<State*>(sg->renderstate);
    if (errormessage)
        *errormessage = ustringhash();
    for (int c = 0; c < nchannels && c < 4; ++c) {
        result[c] = 0;
        if (dresultds)
            dresultds[c] = 0;
        if (dresultdt)
            dresultdt[c] = 0;
    }
    if (!state)
        return false;
    auto& audit = *state->result;
    if (!isfinite(s) || s < 0 || s >= float(slots) || !isfinite(t)
        || !isfinite(dsdx) || !isfinite(dtdx) || !isfinite(dsdy)
        || !isfinite(dtdy)) {
        audit.errors |= invalid_call;
        return false;
    }
    const unsigned slot = unsigned(s);
    ++audit.calls[slot];
    audit.colorspaces[slot] = color_space.hash();
    audit.filenames[slot]   = filename.hash();
    audit.handles[slot]     = reinterpret_cast<uintptr_t>(handle);
    audit.gradients[slot]   = unsigned(dresultds != nullptr)
                              | (unsigned(dresultdt != nullptr) << 1);
    if (!state->texture || audit.handles[slot] != texture_id
        || filename != source || nchannels != 4 || options.firstchannel != 0
        || options.swrap != TextureOpt::WrapClamp
        || options.twrap != TextureOpt::WrapClamp
        || options.interpmode != TextureOpt::InterpClosest) {
        audit.errors |= invalid_call;
        return false;
    }
    if (color_space != ustringhash() && color_space != raw_space
        && color_space != srgb_space) {
        audit.errors |= invalid_source;
        audit.diagnostics[slot] = bad_source.hash();
        if (errormessage)
            *errormessage = bad_source;
        return false;
    }
    const float4 texel = tex2D<float4>(hipTextureObject_t(state->texture), s,
                                       t);
    const float channels[] = { texel.x, texel.y, texel.z, texel.w };
    for (int c = 0; c < nchannels; ++c) {
        const float value = channels[c];
        result[c]         = color_space == srgb_space && c < 3
                                ? (value <= 0.04045f
                                       ? value / 12.92f
                                       : powf((value + 0.055f) / 1.055f, 2.4f))
                                : value;
    }
    return true;
}



extern "C" __global__ void
__raygen__texture_colorspaces()
{
    const unsigned index = hartGetLaunchIndex().x;
    if (index >= points)
        return;
    const auto& params = osl_hart_render_params;
    Result result {};
    State state { &result, params.texture };
    ShaderGlobals sg {};
    sg.P = sg.dPdx = sg.dPdy = sg.dPdz = sg.I = sg.dIdx = sg.dIdy = Vec3(0);
    sg.dPdu = sg.dPdv = sg.dPdtime = sg.Ps = sg.dPsdx = sg.dPsdy = Vec3(0);
    sg.N = sg.Ng   = Vec3(0, 0, 1);
    sg.u           = index ? 0.75f : 0.25f;
    sg.v           = 0.5f;
    sg.dudx        = 0.125f;
    sg.dudy        = -0.0625f;
    sg.dvdx        = 0.25f;
    sg.dvdy        = 0.125f;
    sg.renderstate = &state;
    void* group    = params.groupdata + index * params.group_stride;
    hartDirectCall<void, ShaderGlobals*, void*, void*, void*, int, void*>(
        0u, &sg, group, nullptr, params.outputs, int(index),
        params.interactive);
    hartDirectCall<void, ShaderGlobals*, void*, void*, void*, int, void*>(
        1u, &sg, group, nullptr, params.outputs, int(index),
        params.interactive);
    result.done           = completed;
    params.results[index] = result;
}



// HartContext binds these entries even though this fixture traces no rays.
extern "C" __global__ void
__miss__osl_hart()
{
}



extern "C" __global__ void
__closesthit__osl_hart()
{
}
