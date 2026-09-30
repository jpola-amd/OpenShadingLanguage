<!--
Copyright Contributors to the Open Shading Language project.
SPDX-License-Identifier: BSD-3-Clause
-->

# Arnold-compatible OSL package summary

**Date:** 2026-09-30  
**Status:** Rebuilt and validated OSL-side integration candidate.

The package provides static OSL with Arnold's customized OpenImageIO 2.6,
the `/MD` runtime, and both HART and OptiX enabled. Arnold and its installed
dependencies were not modified. The separate Current/shared profile was
preserved. Source changes remain uncommitted.

This is not binary compatibility with the old Autodesk OSL package or an
exhaustive certification of Arnold. The consuming renderer must be rebuilt.

## Deliverables

| Deliverable | Location |
| --- | --- |
| Installation | [install/hart-arnold](../../install/hart-arnold/) |
| Staged package | [D:\OSL\artifacts\hart-arnold](../../../artifacts/hart-arnold/) |
| Compatibility manifest | [compatibility-manifest.md](../../../artifacts/hart-arnold/share/OSL/compatibility-manifest.md) |
| Complete dependency/link inputs | [arnold-dependencies.json](../../../artifacts/hart-arnold/share/OSL/arnold-dependencies.json) |
| Validation evidence | [validation logs](../../../artifacts/hart-arnold/share/OSL/validation/) |
| Package checksums | [package-sha256.json](../../../artifacts/hart-arnold/share/OSL/package-sha256.json) |

## Restored interfaces and behavior

- **Custom closure allocator:** Four-argument registration executes the
  renderer's allocator on scalar CPU with the correct globals, ID and weight.
  Constructed payloads are preserved, registered parameters are written at
  their declared offsets, and alignment and closure arithmetic are supported.
  The upstream prepare/setup overload remains operational.
- **Loaded-shader lookup:** `ShaderLoaded` performs synchronized, exact-key
  cache lookup, including cached failed parses, matching the observed
  Autodesk behavior.
- **GPU operation metadata:** `num_shade_ops_needed` and `shade_ops_needed`
  expose actual retained operations, with group-owned storage valid until
  group destruction.
- **Shader globals:** The Arnold profile enables `OSL_ARNOLD_MODIFIED_API`
  while retaining the full upstream layout. Generated initialization covers
  fields Arnold previously omitted. Host runtime, generated code and HIP
  layout checks agree.
- **HART device allocation:** A documented device service supports
  renderer-defined IDs, constructed payloads, weights and alignment above
  16 bytes. It never invokes a CPU callback on the GPU or silently substitutes
  the sample renderer's allocator.
- **OptiX callable ABI:** Exported init, entry and fused callables use Arnold's
  exact five-argument signature. CPU and HART retain six arguments.
  OptiX groups requiring an interactive-parameter arena are explicitly
  rejected. HART and OptiX require separate shading-system instances.

## Build and packaging

| Component | Selection |
| --- | --- |
| Host | Windows x64 Release, MSVC 19.44.35228.0, `/MD` |
| OIIO / Imath | Arnold `autodesk-arnold-2.6.3.2-1-3` / `3.1.10-9` |
| LLVM | 23.0.0git; exact compiler identities recorded in the dependency manifest |
| HART | 0.1.0; embedded architecture **gfx1201 only** |
| ROCm / HIP | TheRock 7.14.0rc3; HIP 7.14.60850 |
| OptiX / CUDA | OptiX 8.0.0; CUDA SDK 12.9.1; PTX target **sm_60** |
| JPEG | Official libjpeg-turbo 3.1.0, ABI62, isolated static `/MD` rebuild |

The JPEG CRT conflict was resolved without suppressing LNK4098 or replacing
Arnold's installed archive. CUDA uses dynamic cudart rather than its `/MT`
static archive. Audited tools and consumers have no core OSL/OIIO DLL imports.
All 120 files match between the installation and staged package.

Static CMake exports were verified with fresh consumers. For SCons, use C++17,
`/MD`, `OSL_STATIC_DEFINE=1`, `OIIO_STATIC_DEFINE=1`, and the recorded link
inputs. Replace the complete LLVM 20 input set with LLVM 23; do not mix them.
External SDK/dependency roots remain required: this is not a self-contained
SDK. Standalone tools require Arnold's real `ai.dll` for its OIIO allocators;
Arnold itself supplies those symbols when rebuilt.

## Validation results

| Check | Result |
| --- | --- |
| Final combined acceptance | **14/14 passed** |
| Fresh staged public-API and string-identity consumers | **2/2 compiled, linked and executed successfully** |
| CPU closures, loading and metadata | Passed across OSL/LLVM O0/O2 combinations |
| Shader-globals contract | All 38 host/HIP member offsets, generated initialization and poisoned-state checks passed |
| HART custom closures | Real RX 9070 XT/gfx1201 execution at O0/O2; payloads, 32/64-byte alignment, weights, ADD/MUL and failure paths passed |
| Staged CPU/HART smoke | O0/O2 passed; six pixels per case, maximum numerical error zero |
| Full generated-runtime matrix | Passed in about 1,248 seconds |
| Current/shared controls | CPU, string identity, package export and real HART controls passed |
| JPEG replacement | **211 upstream tests passed** |
| OptiX | PTX generation and five-argument signatures passed; **NVIDIA execution not tested** |

CPU and HART ran on an AMD-only machine without a NVIDIA device. The first
generated-runtime attempt exceeded its 600-second overall timeout; the full
rerun passed. The CTest budget is now 1,800 seconds, with unchanged numerical
tolerances and individual subprocess limits.

## Remaining Arnold integration work

1. Connect Arnold's HART device allocator and callables to the new contract.
2. Rebuild its CPU integration and GPU renderer-service bitcode against the
   matching full-layout headers and LLVM 23 inputs.
3. Run NVIDIA hardware acceptance; successful PTX generation is not a
   substitute for execution.
4. Run Arnold end-to-end regressions, including `test_1580`, `test_1581`
   and `test_1461`, before replacing the production dependency.

Additional AMD architectures require new bitcode and validation.
Renderer-owned allocation in batched or experimental C++ execution, and
interactive-arena OptiX groups, are unsupported and explicitly rejected.

Autodesk's source server was unreachable. Semantics were established from
installed headers, the original static-library implementation and read-only
Arnold callers, not directly forward-ported from the unavailable source patch.

Further details: [API and device contract](ArnoldCompatibility.md) and
[build, runtime and SCons instructions](Arnold_Profile.md).
