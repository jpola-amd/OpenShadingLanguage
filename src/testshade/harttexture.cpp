// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// HIP must precede OIIO on MSVC (HIP's vector-header feature detection).
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <vector>

#include <OSL/oslexec.h>

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imagecache.h>
#include <OpenImageIO/strutil.h>

#include "render_state.h"
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
        : err(handler)
        , descriptors(handler)
        , state(handler)
        , errors(handler)
        , colorsystem(handler)
        , diagnostics(handler)
        , userdata_entries(handler)
        , userdata_data(handler)
        , userdata_state(handler)
        , attributes(handler)
        , transform_entries(handler)
        , transform_state(handler)
        , diagnostic_host(std::make_unique<HartDiagnosticBuffer>())
    {
    }

    OIIO::ErrorHandler& err;
    std::vector<std::unique_ptr<Texture>> textures;
    bool report_diagnostics()
    {
        if (!diagnostics.data)
            return true;
        uint32_t count = 0;
        if (!hip_check(err,
                       hipMemcpy(&count, diagnostics.data, sizeof(count),
                                 hipMemcpyDeviceToHost),
                       "hipMemcpy diagnostic count"))
            return false;
        if (count > HartDiagnosticCapacity) {
            err.errorfmt("HART invalid diagnostic record count {}", count);
            return false;
        }
        if (!count)
            return true;
        const size_t bytes = offsetof(HartDiagnosticBuffer, records)
                             + count * sizeof(HartDiagnosticRecord);
        if (!hip_check(err,
                       hipMemcpy(diagnostic_host.get(), diagnostics.data, bytes,
                                 hipMemcpyDeviceToHost),
                       "hipMemcpy diagnostics"))
            return false;
        std::array<uint32_t, HartDiagnosticCapacity> order;
        for (uint32_t i = 0; i < count; ++i)
            order[i] = i;
        std::sort(order.begin(), order.begin() + count,
                  [&](uint32_t a, uint32_t b) {
                      const auto left = diagnostic_host->records[a].shade_index;
                      const auto right = diagnostic_host->records[b].shade_index;
                      return left < right || (left == right && a < b);
                  });
        bool success = true;
        for (uint32_t position = 0; position < count; ++position) {
            const uint32_t i   = order[position];
            const auto& record = diagnostic_host->records[i];
            auto fail          = [&](string_view reason) {
                err.errorfmt("HART diagnostic record {}: {}", i, reason);
                return false;
            };
            if (record.arg_count > HartDiagnosticMaxArgs
                || record.arg_bytes > HartDiagnosticMaxValues
                || record.severity < 0 || record.severity > 2)
                return fail("invalid header");
            const ustring format = ustring::from_hash(record.format);
            const ustring shader = ustring::from_hash(record.shader);
            const ustring source = ustring::from_hash(record.source);
            if (format.hash() != record.format || shader.hash() != record.shader
                || source.hash() != record.source
                || format.length() > HartDiagnosticMaxFormat
                || shader.length() > HartDiagnosticMaxFormat
                || source.length() > HartDiagnosticMaxFormat)
                return fail("unknown string hash or oversized format/context");
            uint32_t offset = 0;
            for (uint32_t arg = 0; arg < record.arg_count; ++arg) {
                const auto type     = EncodedType(record.arg_types[arg]);
                const uint32_t size = type == EncodedType::kUstringHash ? 8 : 4;
                if ((type != EncodedType::kUstringHash
                     && type != EncodedType::kInt32
                     && type != EncodedType::kUInt32
                     && type != EncodedType::kFloat)
                    || offset > record.arg_bytes
                    || size > record.arg_bytes - offset)
                    return fail("invalid argument types or byte count");
                if (type == EncodedType::kUstringHash) {
                    uint64_t hash;
                    memcpy(&hash, record.arg_values + offset, sizeof(hash));
                    const ustring text = ustring::from_hash(hash);
                    if (text.hash() != hash
                        || text.length() > HartDiagnosticMaxField)
                        return fail(
                            "unknown string argument or field exceeds 1024 bytes");
                }
                offset += size;
            }
            if (offset != record.arg_bytes)
                return fail("argument byte count mismatch");
            std::string message;
            bool valid = record.arg_count == 0;
            const int consumed
                = format.empty()
                      ? 0
                      : decode_message(record.format, int(record.arg_count),
                                       reinterpret_cast<const EncodedType*>(
                                           record.arg_types),
                                       record.arg_values, message, valid);
            if (!valid || consumed != int(record.arg_bytes))
                return fail("invalid formatting or truncated field");
            if (message.size() > HartDiagnosticMaxMessage)
                return fail("message exceeds 4096 bytes");
            auto context = fmtformat("HART shader '{}' ({}:{}, point {}): {}",
                                     shader,
                                     source.empty() ? string_view("<unknown>")
                                                    : string_view(source),
                                     record.line, record.shade_index, message);
            if (context.back() != '\n')
                context += '\n';
            if (record.severity == int(HartDiagnosticSeverity::Error)) {
                err.errorfmt("{}", context);
                success = false;
            } else if (record.severity == int(HartDiagnosticSeverity::Warning))
                err.warningfmt("{}", context);
            else
                err.message(context);
        }
        return success;
    }

    DeviceBuffer descriptors, state, errors, colorsystem, diagnostics;
    DeviceBuffer userdata_entries, userdata_data, userdata_state;
    DeviceBuffer attributes;
    DeviceBuffer transform_entries, transform_state;
    std::unique_ptr<HartDiagnosticBuffer> diagnostic_host;
    size_t colorsystem_bytes = 0;
    bool dirty               = true;
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



void*
HartTextureStore::device_alloc(int device, size_t size)
{
    auto& err = m_impl->err;
    void* ptr = nullptr;
    if (!hip_check(err, hipSetDevice(device), "hipSetDevice interactive")
        || !hip_check(err, hipMalloc(&ptr, size), "hipMalloc interactive"))
        return nullptr;
    return ptr;
}



void
HartTextureStore::device_free(int device, void* ptr)
{
    auto& err = m_impl->err;
    if (hip_check(err, hipSetDevice(device), "hipSetDevice interactive"))
        hip_check(err, hipFree(ptr), "hipFree interactive");
}



void*
HartTextureStore::copy_to_device(int device, void* dst, const void* src,
                                 size_t size)
{
    auto& err = m_impl->err;
    if (!hip_check(err, hipSetDevice(device), "hipSetDevice interactive")
        || !hip_check(err, hipMemcpy(dst, src, size, hipMemcpyHostToDevice),
                      "hipMemcpy interactive"))
        return nullptr;
    return dst;
}



bool
HartTextureStore::prepare_userdata(cspan<HartUserdataBinding> bindings,
                                   size_t points, bool grid_defaults)
{
    auto& impl = *m_impl;
    auto fail  = [&](string_view message) {
        impl.err.errorfmt("HART userdata: {}", message);
        return false;
    };
    if (!points || points > size_t(std::numeric_limits<int>::max()))
        return fail("invalid point count");
    std::vector<testshade::HartUserdataDesc> entries;
    std::vector<unsigned char> data;
    for (const auto& binding : bindings) {
        const auto type = binding.type;
        if (binding.name.empty() || type.arraylen < 0 || !type.aggregate
            || (type.basetype != TypeDesc::INT
                && type.basetype != TypeDesc::FLOAT
                && type.basetype != TypeDesc::STRING)
            || (binding.derivatives && type.basetype != TypeDesc::FLOAT))
            return fail("unsupported name, type, or derivatives");
        const size_t size = type.size();
        if (!size || size > size_t(std::numeric_limits<int>::max()) / 3)
            return fail("unsupported size");
        const size_t record = size * (binding.derivatives ? 3 : 1);
        const size_t count  = binding.stride ? points : 1;
        if (!binding.data.data() || binding.data.size() < record
            || (binding.stride
                && (binding.stride < record
                    || count - 1
                           > (binding.data.size() - record) / binding.stride))
            || (!binding.present.empty() && binding.present.size() != points))
            return fail("invalid data extent, stride, or presence count");
        if (std::any_of(binding.present.begin(), binding.present.end(),
                        [](uint8_t v) { return v > 1; }))
            return fail("presence values must be zero or one");
        const uint64_t name = ustring(binding.name).hash();
        for (const auto& existing : entries)
            if (existing.name == name)
                return fail("duplicate userdata name or hash");
        if (data.size() > data.max_size() - 7)
            return fail("data size overflow");
        const size_t offset = (data.size() + 7) & ~size_t(7);
        if (count > (data.max_size() - offset) / record)
            return fail("data size overflow");
        data.resize(offset + count * record);
        for (size_t point = 0; point < count; ++point) {
            if (binding.stride && !binding.present.empty()
                && !binding.present[point])
                continue;
            const auto* source = binding.data.data() + point * binding.stride;
            auto* destination  = data.data() + offset + point * record;
            if (type.basetype == TypeDesc::STRING) {
                for (size_t element = 0; element < size / sizeof(ustring);
                     ++element) {
                    ustring text;
                    memcpy(&text, source + element * sizeof(ustring),
                           sizeof(text));
                    const auto hash = ustringhash_from(text);
                    memcpy(destination + element * sizeof(hash), &hash,
                           sizeof(hash));
                }
            } else {
                memcpy(destination, source, record);
            }
        }
        uint64_t presence = UINT64_MAX;
        if (!binding.present.empty()) {
            if (binding.present.size() > data.max_size() - data.size())
                return fail("presence size overflow");
            presence = data.size();
            data.insert(data.end(), binding.present.begin(),
                        binding.present.end());
        }
        uint64_t encoded_type;
        static_assert(sizeof(encoded_type) == sizeof(type));
        memcpy(&encoded_type, &type, sizeof(type));
        entries.push_back({ name, encoded_type, offset,
                            binding.stride ? record : 0, presence,
                            uint32_t(size), uint32_t(binding.derivatives) });
    }
    DeviceBuffer device_entries(impl.err), device_data(impl.err),
        state(impl.err);
    const size_t entry_bytes = entries.size()
                               * sizeof(testshade::HartUserdataDesc);
    auto upload = [&](DeviceBuffer& buffer, const void* source, size_t size) {
        return !size
               || (buffer.allocate(size)
                   && hip_check(impl.err,
                                hipMemcpy(buffer.data, source, size,
                                          hipMemcpyHostToDevice),
                                "hipMemcpy userdata"));
    };
    if (!upload(device_entries, entries.data(), entry_bytes)
        || !upload(device_data, data.data(), data.size()))
        return false;
    const testshade::HartUserdataState host {
        static_cast<const testshade::HartUserdataDesc*>(device_entries.data),
        uint64_t(entries.size()),
        static_cast<const unsigned char*>(device_data.data),
        uint64_t(data.size()),
        uint64_t(points),
        uint32_t(grid_defaults),
        0
    };
    if (!upload(state, &host, sizeof(host)))
        return false;
    std::swap(device_entries.data, impl.userdata_entries.data);
    std::swap(device_data.data, impl.userdata_data.data);
    std::swap(state.data, impl.userdata_state.data);
    impl.dirty = true;
    bool ok    = state.clear();
    ok         = device_entries.clear() && ok;
    return device_data.clear() && ok;
}



bool
HartTextureStore::prepare_attributes(const RenderContext& context)
{
    auto& impl                    = *m_impl;
    RenderContext device_context  = context;
    device_context.journal_buffer = nullptr;
    DeviceBuffer attributes(impl.err);
    if (!attributes.allocate(sizeof(context))
        || !hip_check(impl.err,
                      hipMemcpy(attributes.data, &device_context,
                                sizeof(device_context), hipMemcpyHostToDevice),
                      "hipMemcpy attributes"))
        return false;
    std::swap(attributes.data, impl.attributes.data);
    impl.dirty = true;
    return attributes.clear();
}



bool
HartTextureStore::prepare_transforms(cspan<HartTransformBinding> bindings,
                                     ustringhash commonspace,
                                     bool unknown_error)
{
    auto& impl = *m_impl;
    if (bindings.size() > std::numeric_limits<size_t>::max()
                              / sizeof(testshade::HartTransformDesc)) {
        impl.err.errorfmt("HART transforms: too many named bindings");
        return false;
    }
    std::vector<testshade::HartTransformDesc> entries;
    entries.reserve(bindings.size());
    for (const auto& binding : bindings) {
        testshade::HartTransformDesc entry { };
        entry.name       = binding.name.hash();
        entry.directions = (binding.has_forward ? 1u : 0u)
                           | (binding.has_inverse ? 2u : 0u);
        for (int i = 0; i < 16; ++i) {
            entry.forward[i] = binding.forward[i / 4][i % 4];
            entry.inverse[i] = binding.inverse[i / 4][i % 4];
        }
        entries.push_back(entry);
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.name < b.name; });
    if (std::adjacent_find(entries.begin(), entries.end(),
                           [](const auto& a, const auto& b) {
                               return a.name == b.name;
                           })
        != entries.end()) {
        impl.err.errorfmt("HART transforms: duplicate named-space hash");
        return false;
    }
    DeviceBuffer device_entries(impl.err), state(impl.err);
    const size_t bytes = entries.size() * sizeof(testshade::HartTransformDesc);
    if ((bytes
         && (!device_entries.allocate(bytes)
             || !hip_check(impl.err,
                           hipMemcpy(device_entries.data, entries.data(), bytes,
                                     hipMemcpyHostToDevice),
                           "hipMemcpy named transforms")))
        || !state.allocate(sizeof(testshade::HartTransformState)))
        return false;
    const testshade::HartTransformState host {
        static_cast<const testshade::HartTransformDesc*>(device_entries.data),
        uint64_t(entries.size()), commonspace.hash(), uint32_t(unknown_error), 0
    };
    if (!hip_check(impl.err,
                   hipMemcpy(state.data, &host, sizeof(host),
                             hipMemcpyHostToDevice),
                   "hipMemcpy transform state"))
        return false;
    std::swap(device_entries.data, impl.transform_entries.data);
    std::swap(state.data, impl.transform_state.data);
    impl.dirty = true;
    bool ok    = state.clear();
    return device_entries.clear() && ok;
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

    DeviceBuffer descriptors(impl.err), state(impl.err), errors(impl.err),
        diagnostics(impl.err);
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
        || !diagnostics.allocate(sizeof(HartDiagnosticBuffer))
        || !hip_check(impl.err,
                      hipMemset(diagnostics.data, 0, sizeof(uint32_t)),
                      "hipMemset diagnostic count")
        || !state.allocate(sizeof(testshade::HartTextureState)))
        return false;
    const testshade::HartTextureState host_state {
        static_cast<const testshade::HartTextureDesc*>(descriptors.data),
        uint64_t(host.size()),
        static_cast<unsigned int*>(errors.data),
        impl.colorsystem.data,
        static_cast<HartDiagnosticBuffer*>(diagnostics.data),
        static_cast<const testshade::HartUserdataState*>(
            impl.userdata_state.data),
        static_cast<const RenderContext*>(impl.attributes.data),
        static_cast<const testshade::HartTransformState*>(
            impl.transform_state.data)
    };
    if (!hip_check(impl.err,
                   hipMemcpy(state.data, &host_state, sizeof(host_state),
                             hipMemcpyHostToDevice),
                   "hipMemcpy state"))
        return false;
    std::swap(descriptors.data, impl.descriptors.data);
    std::swap(state.data, impl.state.data);
    std::swap(errors.data, impl.errors.data);
    std::swap(diagnostics.data, impl.diagnostics.data);
    impl.dirty = false;
    bool ok    = state.clear();
    ok         = descriptors.clear() && ok;
    ok         = diagnostics.clear() && ok;
    return errors.clear() && ok;
}



bool
HartTextureStore::prepare(ShadingSystem& shadingsys)
{
    auto& impl         = *m_impl;
    void* data         = nullptr;
    long long sizes[2] = { };
    if (!shadingsys.getattribute("colorsystem", TypeDesc::PTR, &data)
        || !shadingsys.getattribute("colorsystem:sizes",
                                    TypeDesc(TypeDesc::LONGLONG, 2), sizes)
        || !data || sizes[0] <= 0 || sizes[1] < 0
        || uint64_t(sizes[0]) > std::numeric_limits<size_t>::max()
        || uint64_t(sizes[1]) > uint64_t(sizes[0]) / sizeof(ustringhash)) {
        impl.err.errorfmt("HART cannot retrieve a valid color-system payload");
        return false;
    }
    static_assert(sizeof(ustringhash) == sizeof(uint64_t),
                  "HART color-system strings require 64-bit hashes");
    const size_t bytes     = size_t(sizes[0]);
    const size_t strings   = size_t(sizes[1]);
    const size_t pod_bytes = bytes - strings * sizeof(ustringhash);
    std::vector<unsigned char> host(bytes);
    std::memcpy(host.data(), data, pod_bytes);
    const auto* hashes = reinterpret_cast<const ustringhash*>(
        static_cast<const unsigned char*>(data) + pod_bytes);
    for (size_t i = 0; i < strings; ++i) {
        const uint64_t hash = hashes[i].hash();
        std::memcpy(host.data() + pod_bytes + i * sizeof(hash), &hash,
                    sizeof(hash));
    }
    if (impl.colorsystem.data && impl.colorsystem_bytes != bytes) {
        impl.err.errorfmt("HART color-system payload size changed from {} to {}",
                          impl.colorsystem_bytes, bytes);
        return false;
    }
    if (!impl.colorsystem.data) {
        if (!impl.colorsystem.allocate(bytes))
            return false;
        impl.colorsystem_bytes = bytes;
        impl.dirty             = true;
    }
    if (!hip_check(impl.err,
                   hipMemcpy(impl.colorsystem.data, host.data(), bytes,
                             hipMemcpyHostToDevice),
                   "hipMemcpy color system"))
        return false;
    return prepare();
}



bool
HartTextureStore::reset_errors()
{
    auto& impl = *m_impl;
    const bool errors_ok
        = !impl.errors.data
          || hip_check(impl.err,
                       hipMemset(impl.errors.data, 0, sizeof(unsigned int)),
                       "hipMemset errors");
    const bool diagnostics_ok
        = !impl.diagnostics.data
          || hip_check(impl.err,
                       hipMemset(impl.diagnostics.data, 0, sizeof(uint32_t)),
                       "hipMemset diagnostic count");
    return errors_ok && diagnostics_ok;
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
    const bool diagnostics_ok = impl.report_diagnostics();
    if (!errors)
        return diagnostics_ok;
    impl.err.errorfmt(
        "HART device services failed (error bits {}): {}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        errors,
        errors & testshade::HartTextureInvalidHandle ? "invalid handle; " : "",
        errors & testshade::HartTextureNonfiniteCoordinates
            ? "nonfinite coordinates/gradients; "
            : "",
        errors & testshade::HartTextureInvalidOptions ? "invalid options; "
                                                      : "",
        errors & testshade::HartInvalidTransform
            ? "invalid transform binding/space; "
            : "",
        errors & testshade::HartClosureAllocationFailed
            ? "closure pool allocation failed; "
            : "",
        errors & testshade::HartClosureInvalidTree ? "invalid closure tree; "
                                                   : "",
        errors & testshade::HartInvalidRayHit
            ? "invalid or repeated ray hit/material; "
            : "",
        errors & testshade::HartArrayIndexOutOfBounds
            ? "array/component index out of range; "
            : "",
        errors & testshade::HartInvalidSpline ? "invalid spline arguments; "
                                              : "",
        errors & testshade::HartUnsupportedColorTransform
            ? "unsupported color transform; "
            : "",
        errors & testshade::HartInvalidNoiseArguments
            ? "invalid noise arguments; "
            : "",
        errors & testshade::HartDiagnosticOverflow
            ? "diagnostic buffer overflow (256 records per launch); "
            : "",
        errors & testshade::HartInvalidDiagnostic
            ? "invalid diagnostic payload; "
            : "",
        errors & testshade::HartShaderError ? "shader error; " : "",
        errors & testshade::HartInvalidUserdata ? "invalid userdata binding; "
                                                : "",
        errors & ~32767u ? "unknown error; " : "");
    return false;
}



bool
HartTextureStore::clear()
{
    auto& impl             = *m_impl;
    bool ok                = impl.state.clear();
    ok                     = impl.descriptors.clear() && ok;
    ok                     = impl.errors.clear() && ok;
    ok                     = impl.diagnostics.clear() && ok;
    ok                     = impl.colorsystem.clear() && ok;
    ok                     = impl.userdata_state.clear() && ok;
    ok                     = impl.userdata_entries.clear() && ok;
    ok                     = impl.userdata_data.clear() && ok;
    ok                     = impl.attributes.clear() && ok;
    ok                     = impl.transform_state.clear() && ok;
    ok                     = impl.transform_entries.clear() && ok;
    impl.colorsystem_bytes = 0;
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
