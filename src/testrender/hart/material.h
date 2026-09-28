// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "../../testshade/hartrenderstate.h"
#include "../shading.cpp"

#include <cmath>

OSL_NAMESPACE_BEGIN

namespace {

OSL_HOSTDEVICE size_t
hart_closure_size(int id)
{
    switch (id) {
    case BACKGROUND_ID:
    case EMISSION_ID: return sizeof(EmptyParams);
    case DIFFUSE_ID:
    case TRANSLUCENT_ID: return sizeof(DiffuseParams);
    case OREN_NAYAR_ID: return sizeof(OrenNayarParams);
    case PHONG_ID: return sizeof(PhongParams);
    case WARD_ID: return sizeof(WardParams);
    case MICROFACET_ID: return sizeof(MicrofacetParams);
    case REFLECTION_ID:
    case FRESNEL_REFLECTION_ID: return sizeof(ReflectionParams);
    case REFRACTION_ID: return sizeof(RefractionParams);
    case TRANSPARENT_ID:
    case MX_TRANSPARENT_ID: return sizeof(EmptyParams);
    case MX_OREN_NAYAR_DIFFUSE_ID: return sizeof(MxOrenNayarDiffuse::Data);
    case MX_BURLEY_DIFFUSE_ID: return sizeof(MxBurleyDiffuse::Data);
    case MX_DIELECTRIC_ID: return sizeof(MxDielectric::Data);
    case MX_CONDUCTOR_ID: return sizeof(MxConductor::Data);
    case MX_GENERALIZED_SCHLICK_ID: return sizeof(MxGeneralizedSchlick::Data);
    case MX_TRANSLUCENT_ID: return sizeof(MxTranslucent::Data);
    case MX_SUBSURFACE_ID: return sizeof(MxSubsurfaceParams);
    case MX_SHEEN_ID: return sizeof(MxSheen::Data);
    case MX_UNIFORM_EDF_ID: return sizeof(MxUniformEdfParams);
    case MX_LAYER_ID: return sizeof(MxLayerParams);
    case MX_ANISOTROPIC_VDF_ID: return sizeof(MxAnisotropicVdfParams);
    case MX_MEDIUM_VDF_ID: return sizeof(MxMediumVdfParams);
    case SPI_THINLAYER: return sizeof(SpiThinLayer::Data);
    default: return 0;
    }
}



template<class T>
OSL_HOSTDEVICE bool
hart_closure_contains(const testshade::HartClosurePool& pool, const void* ptr,
                      size_t extra = 0)
{
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    const auto base    = reinterpret_cast<uintptr_t>(pool.data);
    return address % alignof(T) == 0 && address >= base
           && address - base <= pool.used
           && sizeof(T) + extra <= pool.used - (address - base);
}



OSL_HOSTDEVICE bool
hart_valid_closure(const ClosureColor* root,
                   const testshade::HartClosurePool& pool,
                   bool background = false)
{
    if (pool.failed || pool.used > pool.capacity || (pool.used && !pool.data))
        return false;
    // The reference traversals hold at most sixteen deferred branches.
    const ClosureColor* pending[17];
    unsigned count = 0, visited = 0;
    if (root)
        pending[count++] = root;
    while (count) {
        const auto* node = pending[--count];
        if (!node)
            continue;
        if (++visited > 128 || !hart_closure_contains<ClosureColor>(pool, node))
            return false;
        if (node->id == ClosureColor::ADD) {
            if (count + 2 > 17
                || !hart_closure_contains<ClosureAdd>(pool, node))
                return false;
            pending[count++] = node->as_add()->closureB;
            pending[count++] = node->as_add()->closureA;
        } else if (node->id == ClosureColor::MUL) {
            if (!hart_closure_contains<ClosureMul>(pool, node))
                return false;
            const auto* mul = node->as_mul();
            if (!std::isfinite(mul->weight.x) || !std::isfinite(mul->weight.y)
                || !std::isfinite(mul->weight.z))
                return false;
            pending[count++] = mul->closure;
        } else {
            if (background ? node->id != BACKGROUND_ID
                           : node->id == BACKGROUND_ID)
                return false;
            const size_t bytes = hart_closure_size(node->id);
            if (!bytes
                || !hart_closure_contains<ClosureComponent>(pool, node, bytes))
                return false;
            const auto* comp = node->as_comp();
            if (!std::isfinite(comp->w.x) || !std::isfinite(comp->w.y)
                || !std::isfinite(comp->w.z))
                return false;
            if (comp->id == MX_ANISOTROPIC_VDF_ID
                && !valid_medium_params(*comp->as<MxAnisotropicVdfParams>()))
                return false;
            if (comp->id == MX_MEDIUM_VDF_ID
                && !valid_medium_params(*comp->as<MxMediumVdfParams>()))
                return false;
            if (comp->id == MX_LAYER_ID) {
                if (count + 2 > 17)
                    return false;
                const auto* layer = comp->as<MxLayerParams>();
                pending[count++]  = layer->base;
                pending[count++]  = layer->top;
            }
        }
    }
    return true;
}

}  // namespace

OSL_NAMESPACE_END
