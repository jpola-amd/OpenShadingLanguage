// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "hartclosureparams.h"
#include "harttexture.h"

OSL_NAMESPACE_BEGIN

class SimpleRenderer;
class ShadingSystem;
class ShaderGroup;

struct HartOptions {
    std::string module;
    std::string callable_module;
    std::string entry = "__raygen__testshade";
    int device        = 0;
    bool no_cache     = false;
    bool fused        = false;
    std::string local_groupdata = "0";
    bool has_local_groupdata    = false;
    bool runstats               = false;
    bool pixelcenters           = false;
    bool has_module = false, has_callables = false, has_entry = false;
    std::vector<std::string> shader_entries;
    std::vector<std::string> entry_outputs;
    std::function<bool()> update_parameters;
    std::vector<HartUserdataBinding> userdata_bindings;
};

struct HartOutputRequest {
    std::string name;
    std::string filename;
};

int
testshade_hart(int argc, const char* argv[]);

bool
testshade_hart_validate_generated(int argc, const char* argv[],
                                  const HartOptions& options, int width,
                                  int height, int iterations);

std::unique_ptr<SimpleRenderer>
testshade_hart_renderer(int device, std::string& arch, bool closures = false);

// GPU-only numerical inspection, separate from ordinary RGB testshade output.
bool
testshade_hart_closure_test(SimpleRenderer& renderer, ShadingSystem& shadingsys,
                            ShaderGroup* group, const HartOptions& options,
                            string_view arch, int width, int height,
                            span<testshade::HartClosureSummary> summaries,
                            size_t capacity = testshade::HartClosureCapacity);

// With no explicit outputs, retain the RGB Cout default when present. If Cout
// is absent and output_file is "null", execute without an output arena.
bool
testshade_hart_generated(SimpleRenderer& renderer, ShadingSystem& shadingsys,
                         ShaderGroup& group, const HartOptions& options,
                         string_view arch, int width, int height,
                         int iterations, bool warmup, bool verbose, int raytype,
                         bool print_pixels, string_view output_file,
                         string_view dataformat, const Matrix44& object2common,
                         const Matrix44& shader2common,
                         cspan<HartOutputRequest> outputs = { });

OSL_NAMESPACE_END
