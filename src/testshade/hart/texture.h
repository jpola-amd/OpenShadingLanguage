// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <hip/hip_runtime.h>

#include <OSL/rs_free_function.h>
#include <OSL/shaderglobals.h>

#include "../harttextureparams.h"

namespace {

struct TextureSample {
    float value[4] { };
    float ds[4] { };
    float dt[4] { };
};



__device__ int
texture_index(int index, int extent, OSL::TextureOpt::Wrap wrap)
{
    if (wrap == OSL::TextureOpt::WrapPeriodic) {
        index %= extent;
        return index < 0 ? index + extent : index;
    }
    if (wrap == OSL::TextureOpt::WrapClamp)
        return index < 0 ? 0 : (index >= extent ? extent - 1 : index);
    return index >= 0 && index < extent ? index : -1;
}



__device__ float4
texture_texel(const testshade::HartTextureDesc& texture, int level, int x,
              int y, int width, int height, OSL::TextureOpt::Wrap swrap,
              OSL::TextureOpt::Wrap twrap)
{
    x = texture_index(x, width, swrap);
    y = texture_index(y, height, twrap);
    if (x < 0 || y < 0)
        return make_float4(0, 0, 0, 0);
    return tex2DLod<float4>(hipTextureObject_t(texture.object),
                            (float(x) + 0.5f) / float(width),
                            (float(y) + 0.5f) / float(height), float(level));
}



__device__ TextureSample
texture_level(const testshade::HartTextureDesc& texture, int level, float s,
              float t, const OSL::TextureOpt& options)
{
    TextureSample result;
    const int width  = max(1, texture.width >> level);
    const int height = max(1, texture.height >> level);
    if ((options.swrap == OSL::TextureOpt::WrapBlack && (s < -1 || s > 2))
        || (options.twrap == OSL::TextureOpt::WrapBlack && (t < -1 || t > 2)))
        return result;
    if (options.swrap == OSL::TextureOpt::WrapPeriodic)
        s -= floorf(s);
    else if (options.swrap == OSL::TextureOpt::WrapClamp)
        s = fminf(fmaxf(s, 0), 1);
    if (options.twrap == OSL::TextureOpt::WrapPeriodic)
        t -= floorf(t);
    else if (options.twrap == OSL::TextureOpt::WrapClamp)
        t = fminf(fmaxf(t, 0), 1);

    if (options.interpmode == OSL::TextureOpt::InterpClosest) {
        const float4 texel
            = texture_texel(texture, level, int(floorf(s * width)),
                            int(floorf(t * height)), width, height,
                            options.swrap, options.twrap);
        result.value[0] = texel.x;
        result.value[1] = texel.y;
        result.value[2] = texel.z;
        result.value[3] = texel.w;
        return result;
    }

    const float x = s * width - 0.5f, y = t * height - 0.5f;
    const int ix = int(floorf(x)), iy = int(floorf(y));
    const float fx = x - float(ix), fy = y - float(iy);
    const float4 a   = texture_texel(texture, level, ix, iy, width, height,
                                     options.swrap, options.twrap);
    const float4 b   = texture_texel(texture, level, ix + 1, iy, width, height,
                                     options.swrap, options.twrap);
    const float4 c   = texture_texel(texture, level, ix, iy + 1, width, height,
                                     options.swrap, options.twrap);
    const float4 d   = texture_texel(texture, level, ix + 1, iy + 1, width,
                                     height, options.swrap, options.twrap);
    const float av[] = { a.x, a.y, a.z, a.w };
    const float bv[] = { b.x, b.y, b.z, b.w };
    const float cv[] = { c.x, c.y, c.z, c.w };
    const float dv[] = { d.x, d.y, d.z, d.w };
    for (int channel = 0; channel < 4; ++channel) {
        const float top       = av[channel] + fx * (bv[channel] - av[channel]);
        const float bottom    = cv[channel] + fx * (dv[channel] - cv[channel]);
        result.value[channel] = top + fy * (bottom - top);
        result.ds[channel]    = width
                                * ((bv[channel] - av[channel]) * (1 - fy)
                                   + (dv[channel] - cv[channel]) * fy);
        result.dt[channel]    = height
                                * ((cv[channel] - av[channel]) * (1 - fx)
                                   + (dv[channel] - bv[channel]) * fx);
    }
    return result;
}



__device__ bool
texture_wrap_supported(OSL::TextureOpt::Wrap wrap)
{
    return wrap == OSL::TextureOpt::WrapBlack
           || wrap == OSL::TextureOpt::WrapClamp
           || wrap == OSL::TextureOpt::WrapPeriodic;
}

}  // namespace



OSL_RSOP OSL_HOSTDEVICE bool
rs_texture(OSL::OpaqueExecContextPtr ec, OSL::ustringhash,
           OSL::TextureSystem::TextureHandle* handle,
           OSL::TextureSystem::Perthread*, OSL::TextureOpt& options, float s,
           float t, float dsdx, float dtdx, float dsdy, float dtdy,
           int nchannels, float* result, float* dresultds, float* dresultdt,
           OSL::ustringhash*)
{
    const auto* sg    = static_cast<const OSL::ShaderGlobals*>(ec);
    const auto* state = static_cast<const testshade::HartTextureState*>(
        sg->renderstate);
    const uint64_t id  = reinterpret_cast<uintptr_t>(handle);
    unsigned int error = 0;
    if (!state || !state->textures || !id || id > state->count)
        error |= testshade::HartTextureInvalidHandle;
    if (!isfinite(s) || !isfinite(t) || !isfinite(dsdx) || !isfinite(dtdx)
        || !isfinite(dsdy) || !isfinite(dtdy))
        error |= testshade::HartTextureNonfiniteCoordinates;
    if (nchannels < 1 || nchannels > 4
        || (options.interpmode != OSL::TextureOpt::InterpClosest
            && options.interpmode != OSL::TextureOpt::InterpBilinear)
        || !texture_wrap_supported(options.swrap)
        || !texture_wrap_supported(options.twrap))
        error |= testshade::HartTextureInvalidOptions;
    float lod = 0;
    if (!error) {
        const auto& texture = state->textures[id - 1];
        const float rho
            = fmaxf(hypotf(texture.width * dsdx, texture.height * dtdx),
                    hypotf(texture.width * dsdy, texture.height * dtdy));
        if (!isfinite(rho))
            error |= testshade::HartTextureNonfiniteCoordinates;
        else
            lod = fminf(log2f(fmaxf(rho, 1)), float(texture.levels - 1));
    }
    if (error) {
        if (state && state->errors)
            atomicOr(state->errors, error);
        for (int channel = 0; channel < min(nchannels, 4); ++channel) {
            result[channel] = __builtin_nanf("");
            if (dresultds)
                dresultds[channel] = __builtin_nanf("");
            if (dresultdt)
                dresultdt[channel] = __builtin_nanf("");
        }
        return false;
    }

    const auto& texture = state->textures[id - 1];
    const bool linear   = options.interpmode == OSL::TextureOpt::InterpBilinear;
    const int first     = int(floorf(lod + (linear ? 0 : 0.5f)));
    TextureSample sample = texture_level(texture, first, s, t, options);
    if (linear && first + 1 < texture.levels) {
        const auto next   = texture_level(texture, first + 1, s, t, options);
        const float blend = lod - float(first);
        for (int channel = 0; channel < 4; ++channel) {
            sample.value[channel]
                += blend * (next.value[channel] - sample.value[channel]);
            sample.ds[channel] += blend
                                  * (next.ds[channel] - sample.ds[channel]);
            sample.dt[channel] += blend
                                  * (next.dt[channel] - sample.dt[channel]);
        }
    }
    // Differentiate the reconstruction with a fixed footprint, not the LOD.
    // osl_texture applies the existing coordinate-to-screen chain rule.
    for (int channel = 0; channel < nchannels; ++channel) {
        result[channel] = sample.value[channel];
        if (dresultds)
            dresultds[channel] = sample.ds[channel];
        if (dresultdt)
            dresultdt[channel] = sample.dt[channel];
    }
    return true;
}
