// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <hip/hip_runtime_api.h>

#include <array>
#include <vector>

#include <OpenImageIO/deepdata.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/unittest.h>

#include "harttexture.h"

using namespace OSL;

namespace {

class TestErrorHandler final : public OIIO::ErrorHandler {
public:
    void operator()(int code, const std::string& message) override
    {
        OIIO_CHECK_EQUAL(code, EH_ERROR);
        ++errors;
        last_message = message;
        OIIO::print("HART texture diagnostic: {}\n", message);
    }

    int errors = 0;
    std::string last_message;
};



struct Fixtures {
    ~Fixtures()
    {
        for (const auto& file : files)
            OIIO_CHECK_ASSERT(OIIO::Filesystem::remove(file));
    }

    bool write(string_view name, cspan<OIIO::ImageBuf> images,
               bool mips = false)
    {
        files.emplace_back(name);
        auto output = OIIO::ImageOutput::create(name);
        if (!output)
            return false;
        for (size_t i = 0; i < images.size(); ++i) {
            auto spec = images[i].spec();
            if (mips) {
                spec.tile_width = spec.tile_height = 16;
                spec.tile_depth                    = 1;
                spec.attribute("textureformat", "Plain Texture");
            }
            const auto mode = i == 0 ? OIIO::ImageOutput::Create
                              : mips && output->supports("mipmap")
                                  ? OIIO::ImageOutput::AppendMIPLevel
                                  : OIIO::ImageOutput::AppendSubimage;
            if (!output->open(name, spec, mode)
                || !output->write_image(OIIO::TypeDesc::FLOAT,
                                        images[i].localpixels())) {
                OIIO::print(stderr, "Fixture {}: {}\n", name,
                            output->geterror());
                return false;
            }
        }
        return output->close();
    }

    bool deep(string_view name)
    {
        files.emplace_back(name);
        auto output = OIIO::ImageOutput::create(name);
        OIIO::ImageSpec spec(1, 1, 1, OIIO::TypeDesc::FLOAT);
        spec.deep         = true;
        spec.channelnames = { "Z" };
        OIIO::DeepData pixels(spec);
        return output && output->open(name, spec)
               && output->write_deep_image(pixels) && output->close();
    }

    std::vector<std::string> files;
};



OIIO::ImageBuf
image(int width, int height, int channels, float value = 0.0f,
      bool constant = false, int xorigin = 0, int yorigin = 0)
{
    OIIO::ImageSpec spec(width, height, channels, OIIO::TypeDesc::FLOAT);
    spec.x = spec.full_x = xorigin;
    spec.y = spec.full_y = yorigin;
    OIIO::ImageBuf result(spec);
    for (OIIO::ImageBuf::Iterator<float> p(result); !p.done(); ++p)
        for (int c = 0; c < channels; ++c)
            p[c] = value + 100.0f * c
                   + (constant
                          ? 0.0f
                          : float(p.x() - xorigin + 10 * (p.y() - yorigin)));
    return result;
}



bool
hip_ok(hipError_t result)
{
    if (result != hipSuccess)
        OIIO::print(stderr, "HIP failure: {} ({})\n", hipGetErrorName(result),
                    hipGetErrorString(result));
    OIIO_CHECK_EQUAL(result, hipSuccess);
    return result == hipSuccess;
}



bool
read_state(const HartTextureStore& store, testshade::HartTextureState& state)
{
    OIIO_CHECK_ASSERT(store.device_state());
    return store.device_state()
           && hip_ok(hipMemcpy(&state, store.device_state(), sizeof(state),
                               hipMemcpyDeviceToHost));
}



bool
read_desc(const HartTextureStore& store, uint64_t id,
          testshade::HartTextureDesc& desc)
{
    testshade::HartTextureState state { };
    if (!read_state(store, state))
        return false;
    OIIO_CHECK_ASSERT(id > 0 && id <= state.count);
    return id > 0 && id <= state.count
           && hip_ok(hipMemcpy(&desc, state.textures + id - 1, sizeof(desc),
                               hipMemcpyDeviceToHost));
}



bool
inspect(const HartTextureStore& store, uint64_t id,
        cspan<OIIO::ImageBuf> original)
{
    testshade::HartTextureDesc desc { };
    if (!read_desc(store, id, desc))
        return false;
    OIIO_CHECK_EQUAL(desc.width, original[0].spec().width);
    OIIO_CHECK_EQUAL(desc.height, original[0].spec().height);
    OIIO_CHECK_EQUAL(desc.channels, original[0].spec().nchannels);
    int levels = 1;
    for (int n = std::max(desc.width, desc.height); n > 1; n /= 2)
        ++levels;
    OIIO_CHECK_EQUAL(desc.levels, levels);
    const auto object = reinterpret_cast<hipTextureObject_t>(
        uintptr_t(desc.object));
    hipResourceDesc resource { };
    hipTextureDesc sampler { };
    if (!hip_ok(hipGetTextureObjectResourceDesc(&resource, object))
        || !hip_ok(hipGetTextureObjectTextureDesc(&sampler, object)))
        return false;
    OIIO_CHECK_EQUAL(resource.resType, hipResourceTypeMipmappedArray);
    OIIO_CHECK_EQUAL(sampler.filterMode, hipFilterModePoint);
    OIIO_CHECK_EQUAL(sampler.mipmapFilterMode, hipFilterModePoint);
    OIIO_CHECK_EQUAL(sampler.readMode, hipReadModeElementType);
    OIIO_CHECK_EQUAL(sampler.normalizedCoords, 1);
    OIIO_CHECK_EQUAL(sampler.minMipmapLevelClamp, 0.0f);
    OIIO_CHECK_EQUAL(sampler.maxMipmapLevelClamp, float(levels - 1));

    OIIO::ImageBuf previous;
    int width = desc.width, height = desc.height;
    for (int level = 0; level < levels; ++level) {
        OIIO::ImageBuf expected(
            OIIO::ImageSpec(width, height, 4, OIIO::TypeDesc::FLOAT));
        if (size_t(level) < original.size()) {
            const auto& src = original[level];
            for (OIIO::ImageBuf::Iterator<float> p(expected); !p.done(); ++p)
                for (int c = 0; c < src.nchannels(); ++c)
                    p[c] = src.getchannel(p.x() + src.spec().x,
                                          p.y() + src.spec().y, 0, c);
        } else if (!OIIO::ImageBufAlgo::resize(expected, previous, "box",
                                               1.0f)) {
            OIIO_CHECK_ASSERT(false);
            return false;
        }
        hipArray_t array = nullptr;
        hipChannelFormatDesc channel { };
        hipExtent extent { };
        unsigned int flags = 0;
        if (!hip_ok(hipGetMipmappedArrayLevel(&array,
                                              resource.res.mipmap.mipmap, level))
            || !hip_ok(hipArrayGetInfo(&channel, &extent, &flags, array)))
            return false;
        OIIO_CHECK_EQUAL(extent.width, size_t(width));
        OIIO_CHECK_EQUAL(extent.height, size_t(height));
        OIIO_CHECK_EQUAL(channel.f, hipChannelFormatKindFloat);
        OIIO_CHECK_EQUAL(channel.x, 32);
        OIIO_CHECK_EQUAL(channel.y, 32);
        OIIO_CHECK_EQUAL(channel.z, 32);
        OIIO_CHECK_EQUAL(channel.w, 32);
        const size_t row_bytes = size_t(width) * 4 * sizeof(float);
        std::vector<float> pixels(size_t(width) * height * 4);
        if (!hip_ok(hipMemcpy2DFromArray(pixels.data(), row_bytes, array, 0, 0,
                                         row_bytes, height,
                                         hipMemcpyDeviceToHost)))
            return false;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 4; ++c)
                    OIIO_CHECK_EQUAL_THRESH(
                        pixels[(size_t(y) * width + x) * 4 + c],
                        expected.getchannel(x, y, 0, c), 1.0e-6f);
        previous = std::move(expected);
        width    = std::max(1, width / 2);
        height   = std::max(1, height / 2);
    }
    return true;
}



void
reject(HartTextureStore& store, TestErrorHandler& errors, string_view filename,
       string_view diagnostic)
{
    const int before = errors.errors;
    OIIO_CHECK_EQUAL(store.load(ustring(filename)), uint64_t(0));
    OIIO_CHECK_EQUAL(errors.errors, before + 1);
    OIIO_CHECK_ASSERT(OIIO::Strutil::contains(errors.last_message, diagnostic));
}



void
test_resources()
{
    Fixtures fixtures;
    std::array<OIIO::ImageBuf, 1> gray { image(5, 3, 1, 1.0f) };
    std::array<OIIO::ImageBuf, 1> two { image(4, 4, 2, 2.0f) };
    two[0].specmod().channelnames  = { "R", "A" };
    two[0].specmod().alpha_channel = 1;
    two[0].specmod().attribute("oiio:UnassociatedAlpha", 1);
    two[0].specmod().attribute("oiio:ColorSpace", "sRGB");
    std::array<OIIO::ImageBuf, 1> shifted { image(3, 2, 4, 3.0f, false, -7, 9) };
    std::array<OIIO::ImageBuf, 3> mips { image(4, 4, 3, 3.0f, true),
                                         image(2, 2, 3, 17.0f, true),
                                         image(1, 1, 3, 91.0f, true) };
    std::array<OIIO::ImageBuf, 2> subimages { image(4, 2, 1, 5.0f, true),
                                              image(4, 2, 1, 77.0f, true) };
    OIIO_CHECK_ASSERT(fixtures.write("hart-texture-gray.exr", gray));
    OIIO_CHECK_ASSERT(fixtures.write("hart-texture-two.tif", two));
    OIIO_CHECK_ASSERT(fixtures.write("hart-texture-shifted.exr", shifted));
    OIIO_CHECK_ASSERT(fixtures.write("hart-texture-mips.tx", mips, true));
    OIIO_CHECK_ASSERT(fixtures.write("hart-texture-partial.tx",
                                     cspan<OIIO::ImageBuf>(mips.data(), 2),
                                     true));
    OIIO_CHECK_ASSERT(fixtures.write("hart-texture-subimages.tif", subimages));

    TestErrorHandler errors;
    HartTextureStore store(errors);
    OIIO_CHECK_ASSERT(!store.device_state());
    OIIO_CHECK_ASSERT(store.reset_errors());
    OIIO_CHECK_ASSERT(store.check_errors());
    OIIO_CHECK_ASSERT(store.prepare());
    testshade::HartTextureState state { };
    if (!read_state(store, state))
        return;
    OIIO_CHECK_EQUAL(state.count, uint64_t(0));
    OIIO_CHECK_ASSERT(!state.textures);

    const auto first = store.load(ustring("hart-texture-gray.exr"));
    OIIO_CHECK_EQUAL(first, uint64_t(1));
    OIIO_CHECK_ASSERT(store.prepare());
    OIIO_CHECK_ASSERT(inspect(store, first, gray));
    testshade::HartTextureDesc before { }, after { };
    if (!read_desc(store, first, before))
        return;
    const auto* old_state = store.device_state();
    OIIO_CHECK_EQUAL(store.load(ustring("hart-texture-gray.exr")), first);
    OIIO_CHECK_ASSERT(store.prepare());
    OIIO_CHECK_EQUAL(store.device_state(), old_state);

    const auto second = store.load(ustring("hart-texture-two.tif"));
    OIIO_CHECK_EQUAL(second, uint64_t(2));
    OIIO_CHECK_ASSERT(store.prepare());
    if (!read_state(store, state) || !read_desc(store, first, after))
        return;
    OIIO_CHECK_EQUAL(state.count, uint64_t(2));
    OIIO_CHECK_EQUAL(before.object, after.object);
    OIIO_CHECK_EQUAL(store.load(ustring("hart-texture-gray.exr")), first);
    OIIO_CHECK_ASSERT(inspect(store, first, gray));
    OIIO_CHECK_ASSERT(inspect(store, second, two));

    const auto shifted_id  = store.load(ustring("hart-texture-shifted.exr"));
    const auto mip_id      = store.load(ustring("hart-texture-mips.tx"));
    const auto partial_id  = store.load(ustring("hart-texture-partial.tx"));
    const auto subimage_id = store.load(ustring("hart-texture-subimages.tif"));
    OIIO_CHECK_EQUAL(shifted_id, uint64_t(3));
    OIIO_CHECK_EQUAL(mip_id, uint64_t(4));
    OIIO_CHECK_EQUAL(partial_id, uint64_t(5));
    OIIO_CHECK_EQUAL(subimage_id, uint64_t(6));
    OIIO_CHECK_ASSERT(store.prepare());
    OIIO_CHECK_ASSERT(inspect(store, shifted_id, shifted));
    OIIO_CHECK_ASSERT(inspect(store, mip_id, mips));
    OIIO_CHECK_ASSERT(
        inspect(store, partial_id, cspan<OIIO::ImageBuf>(mips.data(), 2)));
    OIIO_CHECK_ASSERT(inspect(store, subimage_id,
                              cspan<OIIO::ImageBuf>(subimages.data(), 1)));
    OIIO_CHECK_EQUAL(errors.errors, 0);

    reject(store, errors, "", "Cannot open");
    reject(store, errors, "hart-texture-no-such-file.exr", "Cannot open");
    for (const char* name :
         { "tex.<UDIM>.exr", "tex.<UVTILE>.exr", "tex.%(UDIM)d.exr",
           "tex.<u>_<v>.exr", "tex_u##v##.exr" })
        reject(store, errors, name, "UDIM");
    std::array<OIIO::ImageBuf, 1> many { image(1, 1, 5) };
    std::array<OIIO::ImageBuf, 1> cropped { image(2, 2, 3) };
    cropped[0].set_full(0, 4, 0, 4, 0, 1);
    std::array<OIIO::ImageBuf, 2> invalid_mips { image(4, 4, 1),
                                                 image(3, 2, 1) };
    OIIO_CHECK_ASSERT(fixtures.write("hart-texture-many.exr", many));
    OIIO_CHECK_ASSERT(fixtures.write("hart-texture-cropped.exr", cropped));
    OIIO_CHECK_ASSERT(
        fixtures.write("hart-texture-bad-mips.tx", invalid_mips, true));
    OIIO_CHECK_ASSERT(fixtures.deep("hart-texture-deep.exr"));
    reject(store, errors, "hart-texture-many.exr", "1-4 channels");
    reject(store, errors, "hart-texture-cropped.exr", "data-window");
    reject(store, errors, "hart-texture-bad-mips.tx", "invalid mip");
    reject(store, errors, "hart-texture-deep.exr", "non-deep");
    OIIO_CHECK_EQUAL(store.load(ustring("hart-texture-gray.exr")), first);
    OIIO_CHECK_ASSERT(store.prepare());
    if (!read_state(store, state))
        return;
    OIIO_CHECK_EQUAL(state.count, uint64_t(6));

    for (unsigned int bit : { 1u, 2u, 4u, 7u, 8u, 15u, 16u }) {
        const int before_errors = errors.errors;
        if (!hip_ok(hipMemcpy(state.errors, &bit, sizeof(bit),
                              hipMemcpyHostToDevice)))
            return;
        OIIO_CHECK_ASSERT(!store.check_errors());
        OIIO_CHECK_EQUAL(errors.errors, before_errors + 1);
        OIIO_CHECK_ASSERT(store.reset_errors());
        OIIO_CHECK_ASSERT(store.check_errors());
    }

    const int before_cleanup = errors.errors;
    OIIO_CHECK_ASSERT(store.clear());
    OIIO_CHECK_ASSERT(!store.device_state());
    OIIO_CHECK_ASSERT(store.clear());
    for (int repeat = 0; repeat < 3; ++repeat) {
        OIIO_CHECK_EQUAL(store.load(ustring("hart-texture-gray.exr")),
                         uint64_t(1));
        OIIO_CHECK_ASSERT(store.prepare());
        OIIO_CHECK_ASSERT(inspect(store, 1, gray));
        OIIO_CHECK_ASSERT(store.clear());
    }
    {
        HartTextureStore lifetime(errors);
        OIIO_CHECK_EQUAL(lifetime.load(ustring("hart-texture-two.tif")),
                         uint64_t(1));
        OIIO_CHECK_ASSERT(lifetime.prepare());
    }
    OIIO_CHECK_EQUAL(errors.errors, before_cleanup);
}

}  // namespace



int
main()
{
    if (!hip_ok(hipSetDevice(0)) || !hip_ok(hipFree(nullptr)))
        return unit_test_failures;
    test_resources();
    return unit_test_failures;
}
