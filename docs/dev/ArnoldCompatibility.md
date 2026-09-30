# Rebuilt Arnold consumer contract

This is a source-integration contract, not binary compatibility with Autodesk
OSL 1.14.2. Rebuild the renderer and its device programs against the headers
from the same installation as the libraries and embedded bitcode. Do not
combine this package's LLVM 23 libraries with Arnold's previous LLVM 20 set.

## Renderer-owned closures

The scalar CPU API is:

```cpp
typedef ClosureComponent* (*AllocClosureFunc)(
    ShaderGlobals*, int id, const Color3* weight);

void ShadingSystem::register_closure(
    string_view name, int id, const ClosureParam* params, AllocClosureFunc alloc);
```

The existing five-argument prepare/setup registration remains available.
The four-argument overload requires a non-null allocator. Registration and
renderer state must remain valid for the lifetime of compiled groups.

The CPU allocator receives the current shader globals and registered ID.
A null weight means unit weight; otherwise the pointed-to color is the
component weight, including a dynamic zero weight. Constant folding may
eliminate a zero-weight closure before execution.
The renderer constructs the component, including its ID and weight, and
constructs the payload at `ClosureComponent::data()`. OSL does **not** clear
or reconstruct that payload. Only supplied formal and keyword parameters
are written; omitted keywords retain the renderer's constructed defaults.
The callback's weight pointer is borrowed for the duration of the call.

`CLOSURE_FINISH_PARAM` describes payload size and alignment, not the combined
allocation. Renderer-owned payloads may require alignment greater than 16.
The renderer must align the payload **and** the component: for example, put
the 16-byte component immediately before a 64-byte-aligned payload. The
component and payload must remain alive while the closure tree is consumed.
Addition/multiplication nodes retain OSL's ordinary closure-tree ABI.
Batched execution and the experimental C++ backend explicitly reject
renderer-owned allocation; they do not substitute a generic allocator.

### HART device construction

A CPU function pointer is never a device allocator. A HART renderer using
the custom registration must advertise:

```cpp
supports("HARTClosureAllocator") == 1
```

This also enables `HARTClosures` and `HARTClosureParameters`. The renderer
must link the following device service, declared in `OSL/rs_free_function.h`,
into its HART pipeline or supply it in compatible renderer-library bitcode:

```cpp
extern "C" __device__ void* rs_hart_allocate_closure_component(
    OSL::OpaqueExecContextPtr globals, int id, int payload_size,
    int payload_alignment, const OSL::Color3* weight);
```

The return value is a constructed `ClosureComponent*`, not a payload pointer.
Its `data()` points to `payload_size` writable bytes satisfying
`payload_alignment`. IDs are renderer-defined nonnegative IDs; neither OSL's
compiler nor this service assumes testshade's IDs or pool. Parameter offsets
are relative to `data()`. CPU and GPU registrations must describe their
respective actual payloads; CPU pointers and CPU object layouts cannot simply
be copied to device memory.

Unweighted calls pass null; weighted calls pass the actual color. OSL skips
all-zero weights and never overwrites constructed headers or unspecified
payload members. Provide `rs_allocate_closure` too, for OSL ADD/MUL nodes and
any closures registered through the upstream overload. That service's size
and alignment arguments describe the whole allocation.

On allocation failure, return null **and record a renderer-visible device
error**. The renderer must reject the resulting launch rather than publish
a silently incomplete material. OSL guards all parameter writes against null.
Unknown IDs, impossible layouts, and unsupported alignment must likewise be
reported, not mapped to a sample closure. Missing capability and host
prepare/setup callbacks are rejected before device compilation. Missing
device service definitions are link errors, not a host-allocation fallback.

Connecting this service to Arnold's HART callables is separate renderer
integration work. The OptiX path continues to use its existing device
allocation entry points; it does not invoke CPU allocators either.

## Loaded shaders and operation metadata

`ShaderLoaded(name)` performs a synchronized lookup of the exact cache key.
Like Autodesk's implementation, it tests cache presence, including a cached
failed parse whose master is null. It does not search disk, compile, load, or
replace a shader, and is not a validity check for a previous failed load.
The replacement policy of `LoadMemoryCompiledShader` is unchanged.

Group queries after optimization provide:

| Attribute | Type | Result |
| --- | --- | --- |
| `num_shade_ops_needed` | `int` | Number of distinct retained opcode names |
| `shade_ops_needed` | `PTR` | Borrowed `OSL::ustring*` array |

The array includes actual surviving operations from used layers, including
renderer-service operations such as `closure`, `getattribute`, `getmessage`,
`texture`, `trace`, `pointcloud_search`, and `pointcloud_get` when present.
All retained opcode names are included, even structural `nop`, `end`, and
`useparam` markers; this is not a renderer-service whitelist. Operations
actually removed from the optimized stream are absent. Querying can trigger
optimization, as with the existing group
resource attributes. Failed optimization is not reported as a successful
empty result. The group owns the immutable array; its pointer remains valid
until group destruction, including after JIT cleanup. A successful empty
group returns a zero count and null pointer.

## Shader globals initialization

`OSL_ARNOLD_COMPAT` is an opt-in source-compatibility profile, disabled by
default. Only this profile defines `OSL_ARNOLD_MODIFIED_API`. It preserves
the full upstream `ShaderGlobals` layout, field order, and construction
behavior; it does not recreate Autodesk's compact struct.

Arnold leaves fields out of its initialization because they were absent from
its former compact layout. Under this profile, generated group initialization
sets the following fields before shader code runs:

| Fields | Compatibility value |
| --- | --- |
| `dtime`, `dPdtime`, `Ps`, `dPsdx`, `dPsdy` | Zero |
| `object2common`, `shader2common` | Null |
| `flipHandedness` | Zero |

CPU execution retains the context, renderer, uniform state, thread/shade
indices, and `Ci` initialization provided by `ShadingSystem::execute()`.
Direct CPU JIT callers must call the group init function before entry layers
and still supply the ordinary execution state themselves.

GPU callers bypass `execute()`. Their generated group init additionally
clears `Ci`, `renderer`, `shadingStateUniform`, and `thread_index`, and sets
`shade_index` from the callable's shade-index argument. It preserves the
renderer-provided `context`, `renderstate`, `tracedata`, `objdata`, and all
other geometry inputs. In particular, Arnold's faux GPU context must not be
replaced with a CPU `ShadingContext`.

HART retains its split init/entry and fused init-plus-entry contracts. The
OptiX entry wrapper also invokes init under this profile because the Arnold
caller invokes only the entry, not the separate init callable. The
`optix_merge_layer_funcs` option controls inlining, not whether initialization
occurs. The OptiX entry continues using the caller's Groupdata storage, so its
outputs remain available to the renderer; redirecting it to the fused
callable could instead select callable-local storage.

These initialization changes are absent when the profile is disabled. The
profile deliberately resets the listed fields even if another caller supplies
them. Renderers needing ordinary upstream values should use a profile-disabled
build. Rebuild all renderer CPU code and GPU renderer-service bitcode against
the matching full-layout headers; existing compact-layout Autodesk bitcode is
not compatible.

### Profile-specific Arnold OptiX callable ABI

The inspected Arnold caller (`core/src/osl/osl_node.cpp`) invokes the entry as:

```cpp
optixDirectCall<void, OSL::ShaderGlobals*, void*, void*, void*, int>(
    program_id, &globals, params, nullptr, nullptr, 0);
```

The five shader arguments are shader globals, Groupdata, userdata base,
output base, and shade index, in that order. With `OSL_ARNOLD_COMPAT=ON`, all
OptiX init, entry, and fused exports have exactly this five-argument signature.
The wrappers forward null as argument six to current OSL's internal functions.
Groups requiring an interactive-parameter arena are explicitly rejected before
PTX cache lookup, code generation, or marking the group successfully JITted.
No device arena address is embedded in generated code. The profile also salts
PTX cache keys so incompatible six-argument cached exports cannot be reused.

CPU and HART callables, and profile-disabled OptiX exports, retain all six
arguments. Their sixth argument is the interactive-parameter arena pointer.
Although interactive data is semantically optional (null for a group that
does not use it), this remains a required function parameter in the
six-argument ABI: callers must pass it, not omit it.

This adapter keeps the current OSL version and requires no Arnold call-site
changes. It does not make compact-layout renderer bitcode compatible with
full-layout SG, nor does successful PTX generation establish NVIDIA runtime
acceptance. Connecting Arnold's HART callables remains separate integration
work and must honor the documented six-argument HART ABI.

## Backend selection

A combined library can provide CPU, HART, and OptiX support. Use separate
`ShadingSystem` instances for HART and OptiX; a renderer must not advertise
both for one instance. CPU and HART operation do not require a physical
NVIDIA device. Windows tools built with CUDA can still require the CUDA
runtime DLL to load, even when no NVIDIA execution is requested.

Consult the staged compatibility manifest for the exact build, dependencies,
architectures, executed tests, and production-readiness limitations. PTX
generation alone is not an NVIDIA execution test.

## Reference provenance

The recipe identifies Autodesk's
`autodesk-arnold-1.14.2.0-0` source revision. The Autodesk Git server was
unreachable during this work, and no local source archive was found.
Compatibility behavior was checked against the installed Autodesk headers,
read-only Arnold callers, and the original static library's implementation.
This is not a claim that the unavailable fork's source patch was directly
forward-ported, nor an exhaustive certification of Arnold's interfaces.
