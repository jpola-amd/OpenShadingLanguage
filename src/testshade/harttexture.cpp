// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// HIP must precede OIIO on MSVC (HIP's vector-header feature detection).
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imagecache.h>
#include <OpenImageIO/strutil.h>

#include "harttexture.h"

OSL_NAMESPACE_BEGIN
namespace {

bool
hip_check(OIIO::ErrorHandler& err, hipError_t result, string_view operation)
{
    if (result == hipSuccess)
        return true;
    err.errorfmt("HART texture {} failed: {} ({})", operation,
                 hipGetErrorName(result), hipGetErrorString(result));
    return false;
}



struct DeviceBuffer {
    explicit DeviceBuffer(OIIO::ErrorHandler& handler) : err(handler) { }
    ~DeviceBuffer() { clear(); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    bool allocate(size_t bytes)
    { return hip_check(err, hipMalloc(&data, bytes), "hipMalloc"); }

    bool clear()
    {
        const bool ok = !data || hip_check(err, hipFree(data), "hipFree");
        data          = nullptr;
        return ok;
    }

    OIIO::ErrorHandler& err;
    void* data = nullptr;
};



struct Texture {
    explicit Texture(OIIO::ErrorHandler& handler) : err(handler) { }
    ~Texture() { clear(); }
    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;

    bool clear()
    {
        bool ok = true;
        if (object)
            ok = hip_check(err, hipDestroyTextureObject(object),
                           "hipDestroyTextureObject")
                 && ok;
        object = nullptr;
        if (array)
            ok = hip_check(err, hipFreeMipmappedArray(array),
                           "hipFreeMipmappedArray")
                 && ok;
        array = nullptr;
        return ok;
    }

    OIIO::ErrorHandler& err;
    OIIO::ustring filename;
    testshade::HartTextureDesc desc { };
    hipMipmappedArray_t array = nullptr;
    hipTextureObject_t object = nullptr;
};



bool
valid_spec(const OIIO::ImageSpec& spec, OIIO::ustring filename,
           OIIO::ErrorHandler& err)
{
    if (spec.deep || spec.width <= 0 || spec.height <= 0 || spec.depth != 1
        || spec.nchannels < 1 || spec.nchannels > 4) {
        err.errorfmt("HART texture '{}': expected a non-deep 2D image with "
                     "positive dimensions and 1-4 channels",
                     filename);
        return false;
    }
    if (spec.x != spec.full_x || spec.y != spec.full_y || spec.z != spec.full_z
        || spec.width != spec.full_width || spec.height != spec.full_height
        || spec.depth != spec.full_depth) {
        err.errorfmt("HART texture '{}': cropped/data-window images are not "
                     "supported",
                     filename);
        return false;
    }
    const auto max_int = std::numeric_limits<int>::max();
    const auto max_bytes
        = std::min(size_t(std::numeric_limits<OIIO::stride_t>::max()),
                   std::numeric_limits<size_t>::max());
    if (int64_t(spec.x) + spec.width > max_int
        || int64_t(spec.y) + spec.height > max_int
        || int64_t(spec.z) + spec.depth > max_int
        || size_t(spec.width)
               > max_bytes / (4 * sizeof(float)) / size_t(spec.height)) {
        err.errorfmt("HART texture '{}': image dimensions overflow", filename);
        return false;
    }
    return true;
}



bool
udim_pattern(OIIO::ustring filename)
{
    const auto lower = OIIO::Strutil::lower(filename.string());
    for (const char* pattern : { "<udim>", "<uvtile>", "%(udim)d", "<u>", "<v>",
                                 "%(u)d", "%(v)d", "_u##", "_v##" })
        if (lower.find(pattern) != std::string::npos)
            return true;
    return false;
}

}  // namespace



struct HartTextureStore::Impl {
    explicit Impl(OIIO::ErrorHandler& handler)
        : err(handler), descriptors(handler), state(handler), errors(handler)
    {
    }

    OIIO::ErrorHandler& err;
    std::vector<std::unique_ptr<Texture>> textures;
    DeviceBuffer descriptors, state, errors;
    bool dirty = true;
};



HartTextureStore::HartTextureStore(OIIO::ErrorHandler& err)
    : m_impl(std::make_unique<Impl>(err))
{
}



HartTextureStore::~HartTextureStore()
{ clear(); }



uint64_t
HartTextureStore::load(OIIO::ustring filename)
{
    auto& impl = *m_impl;
    for (size_t i = 0; i < impl.textures.size(); ++i)
        if (impl.textures[i]->filename == filename)
            return uint64_t(i) + 1;
    if (udim_pattern(filename)) {
        impl.err.errorfmt("HART texture '{}': UDIM patterns are not supported",
                          filename);
        return 0;
    }
    if (impl.textures.size() >= std::numeric_limits<size_t>::max()
                                    / sizeof(testshade::HartTextureDesc)) {
        impl.err.errorfmt("HART texture descriptor table size overflows");
        return 0;
    }

    OIIO::ImageSpec config;
    config.attribute("oiio:UnassociatedAlpha", 1);
    config.attribute("oiio:RawColor", 1);
    // Discover stored mips through OIIO, without cache-generated mip levels.
    auto cache = OIIO::ImageCache::create(false);
#if OIIO_VERSION < 30000
    const auto destroy_cache = [](OIIO::ImageCache* p) {
        OIIO::ImageCache::destroy(p);
    };
    std::unique_ptr<OIIO::ImageCache, decltype(destroy_cache)> cache_owner(
        cache, destroy_cache);
#endif
    cache->attribute("automip", 0);
    OIIO::ImageBuf base(filename.string(), 0, 0, cache, &config);
    if (!base.init_spec(filename.string(), 0, 0)) {
        impl.err.errorfmt("Cannot open HART texture '{}': {}", filename,
                          base.geterror());
        return 0;
    }
    const auto spec = base.spec();
    if (!valid_spec(spec, filename, impl.err))
        return 0;
    int levels = 1;
    for (int w = spec.width, h = spec.height; w > 1 || h > 1; ++levels) {
        w = std::max(1, w / 2);
        h = std::max(1, h / 2);
    }
    const int file_levels = base.nmiplevels();
    if (file_levels < 1 || file_levels > levels) {
        impl.err.errorfmt("HART texture '{}': invalid mip level count {}",
                          filename, file_levels);
        return 0;
    }

    auto texture       = std::make_unique<Texture>(impl.err);
    texture->filename  = filename;
    const auto channel = hipCreateChannelDesc(32, 32, 32, 32,
                                              hipChannelFormatKindFloat);
    if (!hip_check(impl.err,
                   hipMallocMipmappedArray(
                       &texture->array, &channel,
                       make_hipExtent(spec.width, spec.height, 0), levels, 0),
                   "hipMallocMipmappedArray"))
        return 0;

    OIIO::ImageBuf previous;
    int width = spec.width, height = spec.height;
    for (int level = 0; level < levels; ++level) {
        OIIO::ImageBuf image;
        if (level < file_levels) {
            OIIO::ImageBuf source(filename.string(), 0, level, cache, &config);
            if (!source.init_spec(filename.string(), 0, level)) {
                impl.err.errorfmt("Cannot read HART texture '{}' mip {}: {}",
                                  filename, level, source.geterror());
                return 0;
            }
            const auto& mip = source.spec();
            if (!valid_spec(mip, filename, impl.err))
                return 0;
            if (mip.width != width || mip.height != height
                || mip.nchannels != spec.nchannels) {
                impl.err.errorfmt("HART texture '{}': invalid mip {} "
                                  "dimensions/channels (expected {}x{}x{})",
                                  filename, level, width, height,
                                  spec.nchannels);
                return 0;
            }
            if (!source.read(0, level, true, OIIO::TypeDesc::FLOAT)) {
                impl.err.errorfmt("Cannot read HART texture '{}' mip {}: {}",
                                  filename, level, source.geterror());
                return 0;
            }
            source.set_origin(0, 0, 0);
            source.set_full(0, width, 0, height, 0, 1);
            std::array<int, 4> order { -1, -1, -1, -1 };
            for (int c = 0; c < spec.nchannels; ++c)
                order[c] = c;
            const std::array<float, 4> zero { };
            if (!OIIO::ImageBufAlgo::channels(image, source, 4, order, zero)) {
                impl.err.errorfmt("Cannot expand HART texture '{}' channels: {}",
                                  filename, image.geterror());
                return 0;
            }
        } else {
            image.reset(
                OIIO::ImageSpec(width, height, 4, OIIO::TypeDesc::FLOAT));
            if (!OIIO::ImageBufAlgo::resize(image, previous, "box", 1.0f)) {
                impl.err.errorfmt("Cannot resize HART texture '{}' mip {}: {}",
                                  filename, level, image.geterror());
                return 0;
            }
        }
        hipArray_t array       = nullptr;
        const size_t row_bytes = size_t(width) * 4 * sizeof(float);
        if (!hip_check(impl.err,
                       hipGetMipmappedArrayLevel(&array, texture->array, level),
                       "hipGetMipmappedArrayLevel")
            || !hip_check(impl.err,
                          hipMemcpy2DToArray(array, 0, 0, image.localpixels(),
                                             row_bytes, row_bytes, height,
                                             hipMemcpyHostToDevice),
                          "hipMemcpy2DToArray"))
            return 0;
        previous = std::move(image);
        width    = std::max(1, width / 2);
        height   = std::max(1, height / 2);
    }
    hipResourceDesc resource { };
    resource.resType           = hipResourceTypeMipmappedArray;
    resource.res.mipmap.mipmap = texture->array;
    hipTextureDesc sampler { };
    sampler.addressMode[0] = sampler.addressMode[1] = sampler.addressMode[2]
        = hipAddressModeClamp;
    sampler.filterMode          = hipFilterModePoint;
    sampler.mipmapFilterMode    = hipFilterModePoint;
    sampler.readMode            = hipReadModeElementType;
    sampler.normalizedCoords    = 1;
    sampler.maxMipmapLevelClamp = float(levels - 1);
    if (!hip_check(impl.err,
                   hipCreateTextureObject(&texture->object, &resource, &sampler,
                                          nullptr),
                   "hipCreateTextureObject"))
        return 0;
    texture->desc = { uint64_t(reinterpret_cast<uintptr_t>(texture->object)),
                      spec.width, spec.height, levels, spec.nchannels };
    impl.textures.push_back(std::move(texture));
    impl.dirty = true;
    return uint64_t(impl.textures.size());
}



bool
HartTextureStore::prepare()
{
    auto& impl = *m_impl;
    if (!impl.dirty)
        return true;
    std::vector<testshade::HartTextureDesc> host;
    host.reserve(impl.textures.size());
    for (const auto& texture : impl.textures)
        host.push_back(texture->desc);

    DeviceBuffer descriptors(impl.err), state(impl.err), errors(impl.err);
    const size_t bytes = host.size() * sizeof(testshade::HartTextureDesc);
    if ((!host.empty()
         && (!descriptors.allocate(bytes)
             || !hip_check(impl.err,
                           hipMemcpy(descriptors.data, host.data(), bytes,
                                     hipMemcpyHostToDevice),
                           "hipMemcpy descriptors")))
        || !errors.allocate(sizeof(unsigned int))
        || !hip_check(impl.err, hipMemset(errors.data, 0, sizeof(unsigned int)),
                      "hipMemset errors")
        || !state.allocate(sizeof(testshade::HartTextureState)))
        return false;
    const testshade::HartTextureState host_state {
        static_cast<const testshade::HartTextureDesc*>(descriptors.data),
        uint64_t(host.size()), static_cast<unsigned int*>(errors.data)
    };
    if (!hip_check(impl.err,
                   hipMemcpy(state.data, &host_state, sizeof(host_state),
                             hipMemcpyHostToDevice),
                   "hipMemcpy state"))
        return false;
    std::swap(descriptors.data, impl.descriptors.data);
    std::swap(state.data, impl.state.data);
    std::swap(errors.data, impl.errors.data);
    impl.dirty = false;
    bool ok    = state.clear();
    ok         = descriptors.clear() && ok;
    return errors.clear() && ok;
}



bool
HartTextureStore::reset_errors()
{
    auto& impl = *m_impl;
    return !impl.errors.data
           || hip_check(impl.err,
                        hipMemset(impl.errors.data, 0, sizeof(unsigned int)),
                        "hipMemset errors");
}



bool
HartTextureStore::check_errors()
{
    auto& impl          = *m_impl;
    unsigned int errors = 0;
    if (impl.errors.data
        && !hip_check(impl.err,
                      hipMemcpy(&errors, impl.errors.data, sizeof(errors),
                                hipMemcpyDeviceToHost),
                      "hipMemcpy errors"))
        return false;
    if (!errors)
        return true;
    impl.err.errorfmt(
        "HART device services failed (error bits {}): {}{}{}{}{}", errors,
        errors & testshade::HartTextureInvalidHandle ? "invalid handle; " : "",
        errors & testshade::HartTextureNonfiniteCoordinates
            ? "nonfinite coordinates/gradients; "
            : "",
        errors & testshade::HartTextureInvalidOptions ? "invalid options; "
                                                      : "",
        errors & testshade::HartInvalidTransform
            ? "invalid transform binding/space; "
            : "",
        errors & ~15u ? "unknown error; " : "");
    return false;
}



bool
HartTextureStore::clear()
{
    auto& impl = *m_impl;
    bool ok    = impl.state.clear();
    ok         = impl.descriptors.clear() && ok;
    ok         = impl.errors.clear() && ok;
    for (auto& texture : impl.textures)
        ok = texture->clear() && ok;
    impl.textures.clear();
    impl.dirty = true;
    return ok;
}



const testshade::HartTextureState*
HartTextureStore::device_state() const
{ return static_cast<const testshade::HartTextureState*>(m_impl->state.data); }

OSL_NAMESPACE_END
