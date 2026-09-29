// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <OpenImageIO/strutil.h>

#include "oslexec_pvt.h"
#include <OSL/hart_diagnostics.h>


OSL_NAMESPACE_BEGIN


namespace pvt {  // OSL::pvt

using OIIO::ParamValue;
using OIIO::ParamValueList;
using OIIO::spin_lock;

static int next_id = 0;  // We can statically init an int, not an atomic



ShaderInstance::ShaderInstance(ShaderMaster::ref master, string_view layername)
    : m_master(master)
    ,
    //DON'T COPY  m_instsymbols(m_master->m_symbols),
    //DON'T COPY  m_instops(m_master->m_ops), m_instargs(m_master->m_args),
    m_layername(layername)
    , m_writes_globals(false)
    , m_outgoing_connections(false)
    , m_renderer_outputs(false)
    , m_has_error_op(false)
    , m_has_trace_op(false)
    , m_merged_unused(false)
    , m_last_layer(false)
    , m_entry_layer(false)
    , m_firstparam(m_master->m_firstparam)
    , m_lastparam(m_master->m_lastparam)
    , m_maincodebegin(m_master->m_maincodebegin)
    , m_maincodeend(m_master->m_maincodeend)
{
    m_id = ++(*(atomic_int*)&next_id);
    shadingsys().m_stat_instances += 1;

    // We don't copy the symbol table yet, it stays with the master, but
    // we'll keep track of local override information in m_instoverrides.

    // Make it easy for quick lookups of common symbols
    m_Psym = findsymbol(Strings::P);
    m_Nsym = findsymbol(Strings::N);

    // Adjust statistics
    ShadingSystemImpl& ss(shadingsys());
    off_t parammem = vectorbytes(m_iparams) + vectorbytes(m_fparams)
                     + vectorbytes(m_sparams);
    off_t totalmem = (parammem + sizeof(ShaderInstance));
    {
        spin_lock lock(ss.m_stat_mutex);
        ss.m_stat_mem_inst_paramvals += parammem;
        ss.m_stat_mem_inst += totalmem;
        ss.m_stat_memory += totalmem;
    }
}



ShaderInstance::~ShaderInstance()
{
    shadingsys().m_stat_instances -= 1;

    OSL_DASSERT(m_instops.size() == 0 && m_instargs.size() == 0);
    ShadingSystemImpl& ss(shadingsys());
    off_t symmem   = vectorbytes(m_instsymbols) + vectorbytes(m_instoverrides);
    off_t parammem = vectorbytes(m_iparams) + vectorbytes(m_fparams)
                     + vectorbytes(m_sparams);
    off_t connectionmem = vectorbytes(m_connections);
    off_t totalmem      = (symmem + parammem + connectionmem
                           + sizeof(ShaderInstance));
    {
        spin_lock lock(ss.m_stat_mutex);
        ss.m_stat_mem_inst_syms -= symmem;
        ss.m_stat_mem_inst_paramvals -= parammem;
        ss.m_stat_mem_inst_connections -= connectionmem;
        ss.m_stat_mem_inst -= totalmem;
        ss.m_stat_memory -= totalmem;
    }
}



int
ShaderInstance::findsymbol(ustring name) const
{
    for (size_t i = 0, e = m_instsymbols.size(); i < e; ++i)
        if (m_instsymbols[i].name() == name)
            return (int)i;

    // If we haven't yet copied the syms from the master, get it from there
    if (m_instsymbols.empty())
        return m_master->findsymbol(name);

    return -1;
}



int
ShaderInstance::findparam(ustring name, bool search_master) const
{
    if (m_instsymbols.size())
        for (int i = m_firstparam, e = m_lastparam; i < e; ++i)
            if (m_instsymbols[i].name() == name)
                return i;

    // Not found? Try the master.
    if (search_master) {
        for (int i = m_firstparam, e = m_lastparam; i < e; ++i)
            if (master()->symbol(i)->name() == name)
                return i;
    }

    return -1;
}



void*
ShaderInstance::param_storage(int index)
{
    const Symbol* sym = m_instsymbols.size() ? symbol(index)
                                             : mastersymbol(index);

    // Get the data offset. If there are instance overrides for symbols,
    // check whether we are overriding the array size, otherwise just read
    // the offset from the symbol.  Overrides for arraylength -- which occur
    // when an indefinite-sized array parameter gets a value (with a concrete
    // length) -- are special, because in that case the new storage is
    // allocated at the end of the previous parameter list, and thus is not
    // where the master may have thought it was.
    int offset;
    if (m_instoverrides.size() && m_instoverrides[index].arraylen())
        offset = m_instoverrides[index].dataoffset();
    else
        offset = sym->dataoffset();

    TypeDesc t = sym->typespec().simpletype();
    if (t.basetype == TypeDesc::INT) {
        return &m_iparams[offset];
    } else if (t.basetype == TypeDesc::FLOAT) {
        return &m_fparams[offset];
    } else if (t.basetype == TypeDesc::STRING) {
        return &m_sparams[offset];
    } else {
        return NULL;
    }
}



const void*
ShaderInstance::param_storage(int index) const
{
    // Rather than repeating code here, just use const_cast and call the
    // non-const version of this method.
    return (const_cast<ShaderInstance*>(this))->param_storage(index);
}



// Can a parameter with type 'a' be bound to a value of type b?
// Requires matching types (and if arrays, matching lengths or for
// a's length to be undetermined), or it's also ok to bind a single float to
// a non-array triple. All triples are considered equivalent for this test.
inline bool
compatible_param(const TypeDesc& a, const TypeDesc& b)
{
    return equivalent(a, b) || (a.is_vec3() && b == TypeDesc::FLOAT);
}



void
ShaderInstance::parameters(const ParamValueList& params,
                           cspan<ParamHints> hints)
{
    // Seed the params with the master's defaults
    m_iparams = m_master->m_idefaults;
    m_fparams = m_master->m_fdefaults;
    m_sparams = m_master->m_sdefaults;

    m_instoverrides.resize(std::max(0, lastparam()));

    // Set the initial lockgeom and dataoffset on the instoverrides, based
    // on the master.
    for (int i = 0, e = (int)m_instoverrides.size(); i < e; ++i) {
        Symbol* sym = master()->symbol(i);
        m_instoverrides[i].interpolated(sym->interpolated());
        m_instoverrides[i].interactive(sym->interactive());
        m_instoverrides[i].dataoffset(sym->dataoffset());
    }

    for (size_t pi = 0; pi < params.size(); ++pi) {
        const ParamValue& p(params[pi]);
        if (p.name().size() == 0)
            continue;  // skip empty names
        int i = findparam(p.name());
        if (i >= 0) {
            // if (shadingsys().debug())
            //     shadingsys().info (" PARAMETER %s %s", p.name(), p.type());
            const Symbol* sm = master()->symbol(i);  // This sym in the master
            SymOverrideInfo* so
                = &m_instoverrides[i];  // Slot for sym's override info
            TypeSpec sm_typespec = sm->typespec();  // Type of the master's param
            if (sm_typespec.is_closure_based()) {
                // Can't assign a closure instance value.
                shadingsys().warningfmt("skipping assignment of closure: {}",
                                        sm->name());
                continue;
            }
            if (sm_typespec.is_structure_based())
                continue;  // structs are just placeholders; skip

            const void* data = p.data();
            float tmpdata[3];  // used for inline conversions to float/float[3]

            // Check type of parameter and matching symbol. Note that the
            // compatible accounts for indefinite-length arrays.
            TypeDesc paramtype
                = sm_typespec.simpletype();  // what the shader writer wants
            TypeDesc valuetype = p.type();  // what the data provided actually is

            if (master()->shadingsys().relaxed_param_typecheck()) {
                // first handle cases where we actually need to modify the data (like setting a float parameter with an int)
                if ((paramtype == TypeDesc::FLOAT || paramtype.is_vec3())
                    && valuetype.basetype == TypeDesc::INT
                    && valuetype.basevalues() == 1) {
                    int val    = *static_cast<const int*>(p.data());
                    float conv = float(val);
                    if (val != int(conv))
                        shadingsys().errorfmt(
                            "attempting to set parameter from wrong type would change the value: {} (set {:.9g} from {})",
                            sm->name(), conv, val);
                    tmpdata[0] = conv;
                    data       = tmpdata;
                    valuetype  = TypeDesc::FLOAT;
                }

                if (!relaxed_equivalent(sm_typespec, valuetype)) {
                    // We are being very relaxed in this mode, so if the user _still_ got it wrong
                    // something more serious is at play and we should treat it as an error.
                    shadingsys().errorfmt(
                        "attempting to set parameter from incompatible type: {} (expected '{}', received '{}')",
                        sm->name(), paramtype, valuetype);
                    continue;
                }
            } else if (!compatible_param(paramtype, valuetype)) {
                shadingsys().warningfmt(
                    "attempting to set parameter with wrong type: {} (expected '{}', received '{}')",
                    sm->name(), paramtype, valuetype);
                continue;
            }

            // Mark that the override as an instance value
            so->valuesource(Symbol::InstanceVal);

            // Pass on any interpolated or interactive hints.
            auto hint = hints[pi];
            so->interpolated(sm->interpolated()
                             || (hint & ParamHints::interpolated)
                                    == ParamHints::interpolated);
            so->interactive((hint & ParamHints::interactive)
                            == ParamHints::interactive);
            bool lockgeom = !so->interpolated() && !so->interactive();

            OSL_DASSERT(so->dataoffset() == sm->dataoffset());
            so->dataoffset(sm->dataoffset());

            if (paramtype.is_vec3() && valuetype == TypeDesc::FLOAT) {
                // Handle the special case of assigning a float for a triple
                // by replicating it into local memory.
                tmpdata[0] = *(const float*)data;
                tmpdata[1] = *(const float*)data;
                tmpdata[2] = *(const float*)data;
                data       = &tmpdata;
                valuetype  = paramtype;
            }

            if (paramtype.arraylen < 0) {
                // An array of definite size was supplied to a parameter
                // that was an array of indefinite size. Magic! The trick
                // here is that we need to allocate parameter space at the
                // END of the ordinary param storage, since when we assigned
                // data offsets to each parameter, we didn't know the length
                // needed to allocate this param in its proper spot.
                int nelements = valuetype.basevalues();
                // Store the actual length in the shader instance parameter
                // override info. Compute the length this way to account for relaxed
                // parameter checking (for example passing an array of floats to an array of colors)
                so->arraylen(nelements / paramtype.aggregate);
                // Allocate space for the new param size at the end of its
                // usual parameter area, and set the new dataoffset to that
                // position.
                if (paramtype.basetype == TypeDesc::FLOAT) {
                    so->dataoffset((int)m_fparams.size());
                    expand(m_fparams, nelements);
                } else if (paramtype.basetype == TypeDesc::INT) {
                    so->dataoffset((int)m_iparams.size());
                    expand(m_iparams, nelements);
                } else if (paramtype.basetype == TypeDesc::STRING) {
                    so->dataoffset((int)m_sparams.size());
                    expand(m_sparams, nelements);
                } else {
                    OSL_DASSERT(0 && "unexpected type");
                }
                // FIXME: There's a tricky case that we overlook here, where
                // an indefinite-length-array parameter is given DIFFERENT
                // definite length in subsequent rerenders. Don't do that.
            } else {
                // If the instance value is the same as the master's default,
                // just skip the parameter, let it "keep" the default by
                // marking the source as DefaultVal.
                //
                // N.B. Beware the situation where it has init ops, and so the
                // "default value" slot only coincidentally has the same value
                // as the instance value.  We can't mark it as DefaultVal in
                // that case, because the init ops need to be run.
                //
                // Note that this case also can't/shouldn't happen for the
                // indefinite- sized array case, which is why we have it in
                // the 'else' clause of that test.
                void* defaultdata = m_master->param_default_storage(i);
                if (lockgeom && !sm->has_init_ops()
                    && memcmp(defaultdata, data, valuetype.size()) == 0) {
                    // Must reset valuesource to default, in case the parameter
                    // was set already, and now is being changed back to default.
                    so->valuesource(Symbol::DefaultVal);
                }
            }

            // Copy the supplied data into place.
            memcpy(param_storage(i), data, valuetype.size());
        } else {
            shadingsys().warningfmt(
                "attempting to set nonexistent parameter: {}", p.name());
        }
    }

    {
        // Adjust the stats
        ShadingSystemImpl& ss(shadingsys());
        size_t symmem   = vectorbytes(m_instoverrides);
        size_t parammem = (vectorbytes(m_iparams) + vectorbytes(m_fparams)
                           + vectorbytes(m_sparams));
        spin_lock lock(ss.m_stat_mutex);
        ss.m_stat_mem_inst_syms += symmem;
        ss.m_stat_mem_inst_paramvals += parammem;
        ss.m_stat_mem_inst += (symmem + parammem);
        ss.m_stat_memory += (symmem + parammem);
    }
}



bool
hart_supports_noise(ustring name, bool periodic)
{
    return name == ustring("perlin") || name == ustring("uperlin")
           || name == ustring("noise") || name == ustring("snoise")
           || name == ustring("cell") || name == ustring("hash")
           || name == ustring("gabor")
           || (!periodic
               && (name == ustring("simplex") || name == ustring("usimplex")));
}



bool
ShaderInstance::hart_texture_filename(const Symbol& sym,
                                      ustring& filename) const
{
    if (!sym.typespec().is_string() || sym.typespec().is_array())
        return false;
    if (sym.is_constant()) {
        filename = sym.get_string();
        return !filename.empty();
    }
    // Spline selectors share parameter binding resolution with textures, but
    // can also use locals whose initialization proves a single static value.
    if (sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp) {
        const auto& code = m_instsymbols.empty() ? m_master->m_ops : m_instops;
        const auto& args = m_instsymbols.empty() ? m_master->m_args
                                                 : m_instargs;
        const auto& syms = m_instsymbols.empty() ? m_master->m_symbols
                                                 : m_instsymbols;
        const int first = sym.firstwrite(), last = sym.lastwrite();
        if (first < maincodebegin() || last < first || last >= int(code.size())
            || sym.firstread() <= first)
            return false;
        // The first initialization must dominate every read. Functioncall
        // marks an inlined body, not a conditional entry to that body.
        for (int i = maincodebegin(); i < first; ++i) {
            const Opcode& op = code[i];
            if (op.opname() == ustring("functioncall")) {
                if (op.farthest_jump() > i && op.farthest_jump() <= first)
                    i = op.farthest_jump() - 1;
                continue;
            }
            if (op.farthest_jump() > first || op.opname() == ustring("return")
                || op.opname() == ustring("exit"))
                return false;
        }
        bool initialized = false;
        for (int i = first; i <= last; ++i) {
            const Opcode& op = code[i];
            for (int a = 0; a < op.nargs(); ++a) {
                if (!op.argwrite(a) || &syms[args[op.firstarg() + a]] != &sym)
                    continue;
                if (op.opname() != ustring("assign") || op.nargs() != 2
                    || a != 0 || (!initialized && i != first))
                    return false;
                const Symbol& src = syms[args[op.firstarg() + 1]];
                // Strictly earlier definitions also bound alias recursion.
                if ((src.symtype() == SymTypeLocal
                     || src.symtype() == SymTypeTemp)
                    && src.firstwrite() >= first)
                    return false;
                ustring value;
                if (!hart_texture_filename(src, value)
                    || (initialized && filename != value))
                    return false;
                filename    = value;
                initialized = true;
            }
        }
        // Inlining repeated calls can initialize the same local more than
        // once. It is immutable only if every write resolves to one value.
        return initialized;
    }
    if (sym.symtype() != SymTypeParam || sym.everwritten()
        || sym.has_init_ops())
        return false;
    const int index = findparam(sym.name(), m_instsymbols.empty());
    if (index < 0)
        return false;
    // Validation precedes symbol copying; lowering also needs this at OSL O0.
    const auto source       = m_instoverrides.empty()
                                  ? sym.valuesource()
                                  : m_instoverrides[index].valuesource();
    const bool interpolated = m_instoverrides.empty()
                                  ? sym.interpolated()
                                  : m_instoverrides[index].interpolated();
    const bool interactive  = m_instoverrides.empty()
                                  ? sym.interactive()
                                  : m_instoverrides[index].interactive();
    if (interpolated || interactive
        || (source != Symbol::DefaultVal && source != Symbol::InstanceVal))
        return false;
    // LLVM layout repurposes dataoffset for Groupdata. The copied symbol's
    // data pointer still addresses its original default or instance value.
    filename = m_instsymbols.empty()
                   ? *static_cast<const ustring*>(param_storage(index))
                   : sym.get_string();
    return !filename.empty();
}



bool
ShaderInstance::validate_hart() const
{
    // Check the original code before constant folding can execute host-only
    // operations or hide unsupported paths in a particular specialization.
    const bool closures = shadingsys().renderer()->supports("HARTClosures");
    const bool bounds   = shadingsys().renderer()->supports("HARTArrayBounds");
    const bool geometry = shadingsys().renderer()->supports("HARTGeometry");
    auto resolved_array_length = [&](int index) {
        const Symbol& sym = m_master->m_symbols[index];
        const auto& type  = sym.typespec();
        int length        = type.is_unsized_array() ? sym.initializers()
                                                    : type.arraylength();
        if (index >= firstparam() && index < lastparam()
            && m_instoverrides[index].arraylen())
            length = m_instoverrides[index].arraylen();
        return length;
    };
    auto validate_type  = [&](const Symbol& sym) {
        const TypeSpec& type = sym.typespec();
        if (type.is_structure_array() && type.structspec()->numfields() == 0) {
            shadingsys().errorfmt(
                "HART: missing struct-array field metadata for '{}' in shader "
                "'{}'; recompile the shader",
                sym.name(), shadername());
            return false;
        }
        if (type.is_structure_based())
            return true;  // Placeholder; flattened members are checked separately.
        if (type.is_array() && !bounds) {
            shadingsys().errorfmt(
                "HART: renderer lacks HARTArrayBounds for '{}' in shader '{}'",
                sym.name(), shadername());
            return false;
        }
        if ((type.is_closure_based() && !closures)
            || (!type.is_float_based() && !type.is_int_based()
                && !type.is_string_based()
                && !(closures && type.is_closure_based()))) {
            shadingsys().errorfmt("HART: unsupported type '{}' for '{}' "
                                  "in shader '{}'",
                                  type.c_str(), sym.name(), shadername());
            return false;
        }
        return true;
    };
    auto validate_transform = [&](const Opcode& op) {
        auto symbol = [&](int arg) -> const Symbol& {
            return m_master->m_symbols[m_master->m_args[op.firstarg() + arg]];
        };
        auto type = [&](int arg) -> const TypeSpec& {
            return symbol(arg).typespec();
        };
        const ustring name = op.opname();
        const int nargs    = op.nargs();
        bool valid         = nargs >= 2 && !symbol(0).is_constant();
        int spaces = 0, first_float = nargs;
        if (valid && name == ustring("matrix")) {
            valid = type(0).is_matrix();
            if (nargs == 2 || nargs == 17)
                first_float = 1;
            else if (nargs == 3 || nargs == 18) {
                spaces      = nargs == 3 && type(2).is_string() ? 2 : 1;
                first_float = 1 + spaces;
            } else
                valid = false;
        } else if (valid && name == ustring("getmatrix")) {
            valid  = nargs == 4 && type(0).is_int() && type(3).is_matrix()
                     && !symbol(3).is_constant();
            spaces = 2;
        } else if (valid
                   && (name == ustring("point") || name == ustring("vector")
                       || name == ustring("normal"))) {
            valid       = (nargs == 4 || nargs == 5) && type(0).is_triple();
            spaces      = nargs == 5 ? 1 : 0;
            first_float = 1 + spaces;
        } else if (valid) {
            valid  = (nargs == 3 || nargs == 4) && type(0).is_triple()
                     && type(nargs - 1).is_triple();
            spaces = nargs == 4 ? 2 : (type(1).is_matrix() ? 0 : 1);
        }
        if (valid) {
            for (int a = 1; a <= spaces; ++a)
                valid &= type(a).is_string();
            for (int a = first_float; a < nargs; ++a)
                valid &= type(a).is_float();
        }
        if (!valid) {
            shadingsys().errorfmt(
                "HART: invalid coordinate transform operands for '{}' in "
                "shader '{}' ({}:{})",
                name, shadername(), op.sourcefile(), op.sourceline());
            return false;
        }
        if (spaces && !shadingsys().renderer()->supports("HARTTransforms")) {
            shadingsys().errorfmt(
                "HART: renderer lacks HARTTransforms in shader '{}' ({}:{})",
                shadername(), op.sourcefile(), op.sourceline());
            return false;
        }
        if (!shadingsys().renderer()->supports("HARTNamedTransforms"))
            for (int a = 1; a <= spaces; ++a) {
                const Symbol& space = symbol(a);
                if (!space.is_constant()) {
                    shadingsys().errorfmt(
                        "HART: coordinate spaces must be literal strings in "
                        "shader '{}' ({}:{})",
                        shadername(), op.sourcefile(), op.sourceline());
                    return false;
                }
                const ustring value = space.get_string();
                if (value != Strings::common && value != Strings::object
                    && value != Strings::shader) {
                    shadingsys().errorfmt(
                        "HART: unsupported coordinate space '{}' in shader "
                        "'{}' ({}:{})",
                        value, shadername(), op.sourcefile(), op.sourceline());
                    return false;
                }
            }
        return true;
    };
    auto validate_attribute = [&](const Opcode& op) {
        if (!shadingsys().renderer()->supports("HARTAttributes")) {
            shadingsys().errorfmt(
                "HART: renderer lacks HARTAttributes in shader '{}' ({}:{})",
                shadername(), op.sourcefile(), op.sourceline());
            return false;
        }
        auto symbol = [&](int arg) -> const Symbol& {
            return m_master->m_symbols[m_master->m_args[op.firstarg() + arg]];
        };
        auto type = [&](int arg) -> const TypeSpec& {
            return symbol(arg).typespec();
        };
        bool valid = op.nargs() >= 3 && op.nargs() <= 5;
        if (valid) {
            const bool object       = op.nargs() >= 4 && type(2).is_string();
            const int attribute     = object ? 2 : 1;
            const bool indexed      = op.nargs() == attribute + 3;
            const auto& destination = type(op.nargs() - 1);
            valid                   = !symbol(0).is_constant()
                    && !symbol(op.nargs() - 1).is_constant() && type(0).is_int()
                    && type(1).is_string() && type(attribute).is_string()
                    && (op.nargs() == attribute + 2 || indexed)
                    && (!indexed || type(attribute + 1).is_int())
                    && !destination.is_structure_based()
                    && !destination.is_closure_based()
                    && (destination.is_float_based()
                        || destination.is_int_based()
                        || destination.is_string_based());
        }
        if (!valid)
            shadingsys().errorfmt(
                "HART: invalid getattribute operands in shader '{}' ({}:{})",
                shadername(), op.sourcefile(), op.sourceline());
        return valid;
    };
    auto validate_string_operands = [&](const Opcode& op) {
        auto type = [&](int arg) -> const TypeSpec& {
            return m_master->m_symbols[m_master->m_args[op.firstarg() + arg]]
                .typespec();
        };
        const ustring name = op.opname();
        bool valid         = name == ustring("useparam");
        if (op.nargs() == 2
            && (name == ustring("assign") || name == ustring("arraycopy"))) {
            valid = type(0).is_string_based() && type(1).is_string_based()
                    && type(0).is_array() == type(1).is_array()
                    && (name != ustring("arraycopy") || type(0).is_array());
        } else if (op.nargs() == 2 && name == ustring("arraylength")) {
            valid = type(0).is_int() && type(1).is_string_based()
                    && type(1).is_array();
        } else if (op.nargs() == 2 && name == ustring("isconstant")) {
            valid = type(0).is_int();
        } else if (op.nargs() == 3 && name == ustring("aref")) {
            valid = type(0).is_string() && type(1).is_string_based()
                    && type(1).is_array() && type(2).is_int();
        } else if (op.nargs() == 3 && name == ustring("aassign")) {
            valid = type(0).is_string_based() && type(0).is_array()
                    && type(1).is_int() && type(2).is_string();
        } else if (op.nargs() == 3
                   && (name == ustring("eq") || name == ustring("neq"))) {
            valid = type(0).is_int() && type(1).is_string()
                    && type(2).is_string();
        } else if (op.nargs() == 2
                   && (name == ustring("hash") || name == ustring("raytype"))) {
            valid = type(0).is_int() && type(1).is_string();
        } else if (name == ustring("getattribute")) {
            valid = true;  // The complete operation is validated first.
        }
        if (!valid)
            shadingsys().errorfmt(
                "HART: unsupported string operands for '{}' in shader '{}' "
                "({}:{})",
                name, shadername(), op.sourcefile(), op.sourceline());
        return valid;
    };
    auto validate_hash = [&](const Opcode& op) {
        auto type = [&](int arg) -> const TypeSpec& {
            return m_master->m_symbols[m_master->m_args[op.firstarg() + arg]]
                .typespec();
        };
        const bool valid
            = (op.nargs() == 2 || op.nargs() == 3) && type(0).is_int()
              && !type(1).is_array()
              && (op.nargs() == 2
                      ? (type(1).is_int() || type(1).is_float()
                         || type(1).is_triple() || type(1).is_string())
                      : ((type(1).is_float() || type(1).is_triple())
                         && type(2).is_float()));
        if (!valid)
            shadingsys().errorfmt(
                "HART: invalid hash operands in shader '{}' ({}:{})",
                shadername(), op.sourcefile(), op.sourceline());
        return valid;
    };
    auto validate_diagnostic = [&](const Opcode& op) {
        auto symbol = [&](int arg) -> const Symbol& {
            return m_master->m_symbols[m_master->m_args[op.firstarg() + arg]];
        };
        auto fail = [&](string_view message) {
            shadingsys().errorfmt("HART: {} in shader '{}' ({}:{})", message,
                                  shadername(), op.sourcefile(),
                                  op.sourceline());
            return false;
        };
        if (!shadingsys().renderer()->supports("HARTDiagnostics"))
            return fail("renderer lacks HARTDiagnostics");
        if (op.nargs() < 1 || !symbol(0).typespec().is_string()
            || !symbol(0).is_constant())
            return fail("diagnostic formats must be literal strings");
        const string_view format(symbol(0).get_string());
        if (format.size() > HartDiagnosticMaxFormat)
            return fail("diagnostic format exceeds 4096 bytes");
        int fields = 0;
        for (size_t i = 0; i < format.size(); ++i) {
            if (format[i] != '%')
                continue;
            const size_t start = i++;
            if (i < format.size() && format[i] == '%')
                continue;
            while (i < format.size()
                   && string_view("-+ #0").find(format[i]) != string_view::npos)
                ++i;
            auto number = [&](unsigned limit) {
                unsigned value = 0;
                while (i < format.size() && format[i] >= '0'
                       && format[i] <= '9') {
                    const unsigned digit = unsigned(format[i++] - '0');
                    if (value > limit / 10 || value * 10 + digit > limit)
                        return false;
                    value = value * 10 + digit;
                }
                return true;
            };
            if (!number(HartDiagnosticMaxField))
                return fail("diagnostic width exceeds 1024");
            if (i < format.size() && format[i] == '.') {
                ++i;
                if (!number(128))
                    return fail("diagnostic precision exceeds 128");
            }
            if (i >= format.size() || i - start >= 120
                || string_view("cdefgimnopsvxX").find(format[i])
                       == string_view::npos)
                return fail("unsupported diagnostic format specification");
            ++fields;
        }
        if (fields != op.nargs() - 1)
            return fail("diagnostic format/argument count mismatch");
        for (int a = 1; a < op.nargs(); ++a) {
            const auto& type = symbol(a).typespec();
            if (type.is_closure_based() || type.is_structure_based()
                || (!type.is_float_based() && !type.is_int_based()
                    && !type.is_string_based()))
                return fail("unsupported diagnostic argument type");
        }
        return true;
    };
    auto validate_closure = [&](const Opcode& op) {
        auto symbol = [&](int arg) -> const Symbol& {
            return m_master->m_symbols[m_master->m_args[op.firstarg() + arg]];
        };
        auto fail = [&](string_view message) {
            shadingsys().errorfmt("HART: {} in shader '{}' ({}:{})", message,
                                  shadername(), op.sourcefile(),
                                  op.sourceline());
            return false;
        };
        if (!closures)
            return fail("unsupported operation 'closure' "
                        "(renderer lacks HARTClosures)");
        if (op.nargs() < 2 || symbol(0).is_constant()
            || !symbol(0).typespec().is_closure()
            || symbol(0).typespec().is_array())
            return fail("invalid closure argument list");
        const int weighted = symbol(1).typespec().is_string() ? 0 : 1;
        if (op.nargs() < 2 + weighted
            || (weighted && !symbol(1).typespec().is_color()))
            return fail("invalid closure weight");
        const Symbol& id = symbol(1 + weighted);
        if (!id.is_constant() || !id.typespec().is_string())
            return fail("closure names must be literal strings");
        const ustring name    = id.get_string();
        const bool diffuse    = name == ustring("diffuse");
        const bool parameters = shadingsys().renderer()->supports(
            "HARTClosureParameters");
        if (!parameters && !diffuse && name != ustring("emission"))
            return fail(fmtformat("unsupported closure '{}'", name));
        const auto* entry = shadingsys().find_closure(name);
        if (!entry)
            return fail(fmtformat("closure '{}' is not registered", name));
        if (entry->prepare || entry->setup)
            return fail(fmtformat(
                "closure '{}' prepare/setup callbacks are unsupported", name));
        if (parameters) {
            auto bad_layout = [&]() {
                return fail(
                    fmtformat("invalid closure '{}' parameter layout", name));
            };
            if (entry->id < 0 || entry->name != name || entry->nformal < 0
                || entry->nkeyword < 0 || entry->struct_size <= 0
                || entry->nformal
                       > std::numeric_limits<int>::max() - entry->nkeyword
                || size_t(entry->nformal) + size_t(entry->nkeyword) + 1
                       != entry->params.size())
                return bad_layout();
            const auto& finish  = entry->params.back();
            const int alignment = finish.field_size;
            if (finish.type != TypeDesc() || finish.key
                || finish.offset != entry->struct_size || alignment <= 0
                || alignment > 16 || (alignment & (alignment - 1))
                || entry->struct_size % alignment)
                return bad_layout();
            const int count = entry->nformal + entry->nkeyword;
            for (int i = 0; i < count; ++i) {
                const auto& p = entry->params[i];
                const TypeDesc element = p.type.elementtype();
                const bool numeric
                    = element == TypeInt || element == TypeFloat
                      || element == TypeColor || element == TypePoint
                      || element == TypeVector || element == TypeNormal
                      || element == TypeMatrix
                      || element == TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3);
                if (p.type.is_unsized_array()
                    || (!numeric && p.type != TypeString
                        && p.type != TypeDesc::PTR))
                    return fail(
                        fmtformat("unsupported closure '{}' parameter type",
                                  name));
                const int field_alignment
                    = p.type == TypeString || p.type == TypeDesc::PTR ? 8 : 4;
                if (p.offset < 0 || p.field_size <= 0
                    || p.offset > entry->struct_size
                    || p.field_size > entry->struct_size - p.offset
                    || size_t(p.field_size) != p.type.size()
                    || p.offset % field_alignment || alignment < field_alignment
                    || (i < entry->nformal ? p.key != nullptr
                                           : !p.key || !p.key[0]))
                    return bad_layout();
                for (int j = 0; j < i; ++j) {
                    const auto& previous = entry->params[j];
                    if ((p.offset < previous.offset + previous.field_size
                         && previous.offset < p.offset + p.field_size)
                        || (p.key && previous.key
                            && string_view(p.key) == previous.key))
                        return bad_layout();
                }
            }
            const int first = 2 + weighted;
            if (entry->nformal > op.nargs() - first
                || (op.nargs() - first - entry->nformal) % 2)
                return fail("invalid closure argument list");
            auto compatible = [&](int arg, TypeDesc type) {
                const Symbol& value    = symbol(arg);
                const TypeSpec& actual = value.typespec();
                if (actual.is_structure_based())
                    return false;
                if (type == TypeDesc::PTR)
                    return actual.is_closure() && !actual.is_array();
                TypeDesc actual_type = actual.simpletype();
                if (actual.is_array()) {
                    actual_type.arraylen = resolved_array_length(
                        m_master->m_args[op.firstarg() + arg]);
                    if (actual_type.arraylen <= 0)
                        return false;
                }
                return !actual.is_closure_based()
                       && equivalent(actual_type, type);
            };
            for (int i = 0; i < entry->nformal; ++i)
                if (!compatible(first + i, entry->params[i].type))
                    return fail(fmtformat(
                        "incompatible formal argument to closure '{}'", name));
            for (int i = first + entry->nformal; i < op.nargs(); i += 2) {
                const auto& key = symbol(i);
                if (!key.typespec().is_string() || !key.is_constant())
                    return fail(
                        "closure keyword names must be literal strings");
                bool found = false;
                for (int j = entry->nformal; j < count; ++j) {
                    const auto& p = entry->params[j];
                    if (key.get_string() == p.key
                        && compatible(i + 1, p.type)) {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    return fail(fmtformat(
                        "unsupported or incompatible keyword '{}' to closure '{}'",
                        key.get_string(), name));
            }
            return true;
        }
        const int nformal = diffuse ? 1 : 0;
        if (op.nargs() > 2 + weighted + nformal)
            return fail("closure keyword arguments are unsupported");
        if (op.nargs() != 2 + weighted + nformal)
            return fail("invalid closure argument list");
        // The initial device ABI is an empty emission or a diffuse normal at
        // offset zero, optionally followed by testshade's unused default label.
        bool layout = entry->nformal == nformal && entry->nkeyword == 0
                      && entry->struct_size == (diffuse ? 12 : 1);
        if (diffuse && entry->nformal == 1 && entry->nkeyword == 1
            && entry->params.size() >= 2) {
            const ClosureParam& label = entry->params[1];
            layout = label.key && string_view(label.key) == "label"
                     && label.type == TypeString && label.offset == 16
                     && label.field_size == 8 && entry->struct_size == 24;
        }
        if (diffuse && layout) {
            const ClosureParam& normal = entry->params[0];
            layout = !normal.key && normal.type == TypeVector
                     && normal.offset == 0 && normal.field_size == 12;
        }
        if (!layout)
            return fail(
                fmtformat("unsupported closure '{}' parameter layout", name));
        if (diffuse) {
            const TypeSpec& type = symbol(2 + weighted).typespec();
            if (type.is_array() || type.is_structure()
                || type.is_closure_based()
                || !equivalent(type.simpletype(), TypeVector))
                return fail(
                    fmtformat("incompatible formal argument to closure '{}'",
                              name));
        }
        return true;
    };
    auto validate_texture = [&](const Opcode& op) {
        auto symbol = [&](int arg) -> const Symbol& {
            return m_master->m_symbols[m_master->m_args[op.firstarg() + arg]];
        };
        auto fail = [&](string_view message) {
            shadingsys().errorfmt("HART: {} in shader '{}' ({}:{})", message,
                                  shadername(), op.sourcefile(),
                                  op.sourceline());
            return false;
        };
        if (!shadingsys().renderer()->supports("HARTTextures"))
            return fail("unsupported operation 'texture' "
                        "(renderer lacks HARTTextures)");
        ustring filename;
        if (op.nargs() < 4
            || (!symbol(1).is_constant() && symbol(1).symtype() != SymTypeParam)
            || !hart_texture_filename(symbol(1), filename))
            return fail("texture requires a literal filename or an immutable "
                        "nonempty input string parameter");
        const int first_option
            = op.nargs() > 4 && symbol(4).typespec().is_float() ? 8 : 4;
        if (op.nargs() < first_option || (op.nargs() - first_option) % 2)
            return fail("invalid texture argument list");
        const bool defaults = shadingsys().renderer()->supports(
            "HARTTextureDefaults");
        bool interp = defaults, swrap = defaults, twrap = defaults;
        for (int a = first_option; a < op.nargs(); a += 2) {
            const Symbol& token = symbol(a);
            const Symbol& value = symbol(a + 1);
            if (!token.is_constant() || !token.typespec().is_string())
                return fail("texture option names must be literal strings");
            const ustring name = token.get_string();
            if (name == Strings::alpha) {
                if (!value.typespec().is_float() || value.is_constant())
                    return fail("texture alpha requires a float output");
                continue;
            }
            if (name == Strings::firstchannel) {
                if (!value.is_constant() || !value.typespec().is_int()
                    || value.get_int() < 0)
                    return fail(
                        "texture firstchannel requires a literal nonnegative integer");
                continue;
            }
            if (name != ustring("interp") && name != ustring("wrap")
                && name != ustring("swrap") && name != ustring("twrap"))
                return fail(fmtformat("unsupported texture option '{}'", name));
            if (!value.is_constant() || !value.typespec().is_string())
                return fail("texture option values must be literal strings");
            const ustring mode = value.get_string();
            if (name == ustring("interp")) {
                if (mode != ustring("closest") && mode != ustring("linear"))
                    return fail(
                        fmtformat("unsupported texture interpolation '{}'",
                                  mode));
                interp = true;
            } else {
                if (mode != ustring("black") && mode != ustring("clamp")
                    && mode != ustring("periodic"))
                    return fail(
                        fmtformat("unsupported texture wrap mode '{}'", mode));
                swrap |= name != ustring("twrap");
                twrap |= name != ustring("swrap");
            }
        }
        if (!interp)
            return fail(
                "texture requires explicit closest or linear interpolation");
        if (!swrap || !twrap)
            return fail("texture requires explicit wrap modes");
        return true;
    };
    auto validate_noise = [&](const Opcode& op) {
        auto symbol = [&](int arg) -> const Symbol& {
            return m_master->m_symbols[m_master->m_args[op.firstarg() + arg]];
        };
        auto fail = [&](string_view message) {
            shadingsys().errorfmt("HART: {} in shader '{}' ({}:{})", message,
                                  shadername(), op.sourcefile(),
                                  op.sourceline());
            return false;
        };
        auto numeric = [](const Symbol& sym) {
            return !sym.typespec().is_array()
                   && (sym.typespec().is_float() || sym.typespec().is_triple());
        };
        if (op.nargs() < 2 || !numeric(symbol(0)))
            return fail("invalid noise result or argument list");
        const bool periodic = op.opname() == ustring("pnoise");
        int arg             = 1;
        ustring name        = op.opname();
        bool dynamic        = false;
        if (symbol(arg).typespec().is_string()) {
            dynamic = !symbol(arg).is_constant();
            name    = dynamic ? ustring() : symbol(arg).get_string();
            ++arg;
            if (!dynamic && !hart_supports_noise(name, periodic))
                return fail(fmtformat("unsupported noise type '{}'", name));
        }
        if (arg >= op.nargs() || !numeric(symbol(arg)))
            return fail("noise coordinates must be scalar or triple floats");
        const bool triple = symbol(arg++).typespec().is_triple();
        bool time         = false;
        if (periodic) {
            if (arg + 1 < op.nargs() && numeric(symbol(arg + 1)))
                time = true;
        } else if (arg < op.nargs() && symbol(arg).typespec().is_float()
                   && !symbol(arg).typespec().is_array())
            time = true;
        if (time) {
            if (arg >= op.nargs() || !symbol(arg).typespec().is_float()
                || symbol(arg).typespec().is_array())
                return fail("second noise coordinate must be a float");
            ++arg;
        }
        if (periodic) {
            if (arg >= op.nargs() || !numeric(symbol(arg))
                || symbol(arg).typespec().is_triple() != triple)
                return fail("noise period must match its coordinate type");
            ++arg;
            if (time) {
                if (arg >= op.nargs() || !symbol(arg).typespec().is_float()
                    || symbol(arg).typespec().is_array())
                    return fail("second noise period must be a float");
                ++arg;
            }
        }
        if (dynamic || name == ustring("gabor")) {
            if (!shadingsys().renderer()->supports("HARTNoiseErrors"))
                return fail("renderer lacks HARTNoiseErrors");
        } else if (arg != op.nargs())
            return fail("noise options require gabor");
        if ((op.nargs() - arg) % 2)
            return fail("invalid noise option list");
        for (; arg < op.nargs(); arg += 2) {
            const Symbol& token = symbol(arg);
            const Symbol& value = symbol(arg + 1);
            if (!token.typespec().is_string() || !token.is_constant())
                return fail("noise option names must be literal strings");
            const ustring option = token.get_string();
            const TypeSpec& type = value.typespec();
            const bool valid
                = !type.is_array()
                  && (((option == ustring("anisotropic")
                        || option == ustring("do_filter"))
                       && type.is_int())
                      || (option == ustring("direction") && type.is_triple())
                      || ((option == ustring("bandwidth")
                           || option == ustring("impulses"))
                          && (type.is_float() || type.is_int())));
            if (!valid)
                return fail(
                    fmtformat("unsupported noise option '{}' or type '{}'",
                              option, type.c_str()));
        }
        return true;
    };
    auto validate_spline = [&](const Opcode& op) {
        auto symbol = [&](int arg) -> const Symbol& {
            return m_master->m_symbols[m_master->m_args[op.firstarg() + arg]];
        };
        auto fail = [&](string_view message) {
            shadingsys().errorfmt("HART: {} in shader '{}' ({}:{})", message,
                                  shadername(), op.sourcefile(),
                                  op.sourceline());
            return false;
        };
        if (!shadingsys().renderer()->supports("HARTSplineErrors"))
            return fail("renderer lacks HARTSplineErrors");
        if (op.nargs() != 4 && op.nargs() != 5)
            return fail("invalid spline argument list");
        const Symbol& basis = symbol(1);
        ustring name;
        if (!hart_texture_filename(basis, name))
            return fail("spline basis must be a nonempty immutable string");
        if (name != ustring("catmull-rom") && name != ustring("bezier")
            && name != ustring("bspline") && name != ustring("hermite")
            && name != ustring("linear") && name != ustring("constant"))
            return fail(fmtformat("unsupported spline basis '{}'", name));
        const int step         = name == ustring("bezier")    ? 3
                                 : name == ustring("hermite") ? 2
                                                              : 1;
        const Symbol& knots    = symbol(op.nargs() - 1);
        const TypeSpec& type   = knots.typespec();
        const TypeDesc element = type.simpletype().elementtype();
        const TypeSpec& result = symbol(0).typespec();
        const bool inverse     = op.opname() == ustring("splineinverse");
        if (!type.is_array() || element.basetype != TypeDesc::FLOAT
            || (element.aggregate != TypeDesc::SCALAR
                && element.aggregate != TypeDesc::VEC3)
            || result.is_array() || (!result.is_float() && !result.is_triple())
            || (result.is_float() != (element.aggregate == TypeDesc::SCALAR))
            || (inverse && !result.is_float())
            || !symbol(2).typespec().is_float()
            || (op.nargs() == 5 && !symbol(3).typespec().is_int()))
            return fail("invalid spline knot/value types");
        const int length = resolved_array_length(
            m_master->m_args[op.firstarg() + op.nargs() - 1]);
        if (length < 4)
            return fail("at least four resolved spline knots are required");
        if (op.nargs() == 4 || symbol(3).is_constant()) {
            const int count = op.nargs() == 4 ? length : symbol(3).get_int();
            if (count < 4 || count > length || (count - 4) % step)
                return fail("invalid spline knot count for array/basis");
        }
        return true;
    };
    auto validate_color = [&](const Opcode& op) {
        const bool constructor = op.opname() == ustring("color");
        if (constructor && op.nargs() == 4)
            return true;
        auto symbol = [&](int arg) -> const Symbol& {
            return m_master->m_symbols[m_master->m_args[op.firstarg() + arg]];
        };
        auto fail = [&](string_view message) {
            shadingsys().errorfmt("HART: {} in shader '{}' ({}:{})", message,
                                  shadername(), op.sourcefile(),
                                  op.sourceline());
            return false;
        };
        if (!shadingsys().renderer()->supports("HARTColorSystem"))
            return fail("renderer lacks HARTColorSystem");
        const bool transform = op.opname() == ustring("transformc");
        const bool luminance = op.opname() == ustring("luminance");
        if (op.nargs() != (constructor ? 5 : (transform ? 4 : 2)))
            return fail("invalid color operation arguments");
        const TypeSpec& result = symbol(0).typespec();
        if (result.is_array()
            || (luminance ? !result.is_float() : !result.is_triple()))
            return fail("invalid color operation result type");
        const int first_value = constructor ? 2 : (transform ? 3 : 1);
        for (int a = first_value; a < op.nargs(); ++a) {
            const TypeSpec& type = symbol(a).typespec();
            if (type.is_array()
                || ((luminance || transform) ? !type.is_triple()
                                             : !type.is_float()))
                return fail("invalid color operation value type");
        }
        if (!constructor && !transform)
            return true;
        for (int a = 1; a < first_value; ++a) {
            const Symbol& space = symbol(a);
            if (!space.is_constant() || !space.typespec().is_string())
                return fail("color spaces must be literal strings");
            const ustring name = space.get_string();
            // Constructors use to_rgb, whose built-ins intentionally differ
            // from transformc. Other RGB system names are only current aliases.
            const bool builtin
                = name == ustring("RGB") || name == ustring("rgb")
                  || name == ustring("hsv") || name == ustring("hsl")
                  || name == ustring("YIQ") || name == ustring("XYZ")
                  || name == ustring("xyY")
                  || (transform
                      && (name == ustring("linear") || name == ustring("sRGB")));
            if (!builtin
                && ustringhash(name) != shadingsys().colorsystem().colorspace())
                return fail(fmtformat("unsupported color space '{}'", name));
        }
        return true;
    };
    for (int i = firstparam(); i < lastparam(); ++i) {
        const Symbol& sym = *mastersymbol(i);
        if (!validate_type(sym))
            return false;
        const auto& hints = m_instoverrides[i];
        if (hints.interpolated() && sym.typespec().is_closure_based()) {
            shadingsys().errorfmt(
                "HART: interpolated parameter '{}' cannot be closure-based",
                sym.name());
            return false;
        }
        if (hints.interactive() && sym.typespec().is_closure_based()) {
            shadingsys().errorfmt("HART: interactive closure parameter '{}' "
                                  "is unsupported",
                                  sym.name());
            return false;
        }
        if (hints.interpolated() && hints.interactive()) {
            if (sym.symtype() != SymTypeParam
                || (!sym.typespec().is_float_based()
                    && !sym.typespec().is_int_based())) {
                shadingsys().errorfmt(
                    "HART: interpolated interactive parameter '{}' must be "
                    "a numeric input",
                    sym.name());
                return false;
            }
            if (sym.has_init_ops()
                && hints.valuesource() == Symbol::DefaultVal) {
                shadingsys().errorfmt(
                    "HART: interpolated interactive parameter '{}' requires "
                    "a constant default or an instance value",
                    sym.name());
                return false;
            }
        }
        const bool missing_userdata    = hints.interpolated()
                                         && !shadingsys().renderer()->supports(
                                             "HARTUserdata");
        const bool missing_interactive = hints.interactive()
                                         && !shadingsys().renderer()->supports(
                                             "HARTInteractive");
        if (missing_userdata || missing_interactive) {
            shadingsys().errorfmt(
                "HART: {} parameter '{}' is unsupported by the renderer in "
                "shader '{}'",
                missing_userdata ? "interpolated" : "interactive", sym.name(),
                shadername());
            return false;
        }
    }
    // clang-format off
    static const ustring supported[] = {
        ustring("nop"),
        ustring("end"),
        ustring("useparam"),
        ustring("isconstant"),
        ustring("assign"),
        ustring("add"),
        ustring("sub"),
        ustring("mul"),
        ustring("div"),
        ustring("neg"),
        ustring("color"),
        ustring("sin"),
        ustring("compref"),
        ustring("compassign"),
        ustring("if"),
        ustring("lt"),
        ustring("le"),
        ustring("eq"),
        ustring("ge"),
        ustring("gt"),
        ustring("neq"),
        ustring("for"),
        ustring("while"),
        ustring("dowhile"),
        ustring("break"),
        ustring("continue"),
        ustring("return"),
        ustring("exit"),
        ustring("and"),
        ustring("or"),
        ustring("bitand"),
        ustring("bitor"),
        ustring("xor"),
        ustring("compl"),
        ustring("shl"),
        ustring("shr"),
        ustring("mod"),
        ustring("arraycopy"),
        ustring("arraylength"),
        ustring("aref"),
        ustring("aassign"),
        ustring("Dx"),
        ustring("Dy"),
        ustring("point"),
        ustring("vector"),
        ustring("normal"),
        ustring("dot"),
        ustring("length"),
        ustring("normalize"),
        ustring("filterwidth"),
        ustring("noise"),
        ustring("snoise"),
        ustring("abs"),
        ustring("min"),
        ustring("max"),
        ustring("clamp"),
        ustring("mix"),
        ustring("step"),
        ustring("smoothstep"),
        ustring("floor"),
        ustring("ceil"),
        ustring("fmod"),
        ustring("cos"),
        ustring("sqrt"),
        ustring("pow"),
        ustring("functioncall"),
        ustring("pnoise"),
        ustring("psnoise"),
        ustring("cellnoise"),
        ustring("hashnoise"),
        ustring("hash"),
        ustring("printf"),
        ustring("warning"),
        ustring("error"),
        ustring("texture"),
        ustring("matrix"),
        ustring("mxcompref"),
        ustring("mxcompassign"),
        ustring("transpose"),
        ustring("determinant"),
        ustring("transform"),
        ustring("transformv"),
        ustring("transformn"),
        ustring("getmatrix"),
        ustring("closure"),
        ustring("tan"),
        ustring("asin"),
        ustring("acos"),
        ustring("atan"),
        ustring("atan2"),
        ustring("sinh"),
        ustring("cosh"),
        ustring("tanh"),
        ustring("sincos"),
        ustring("log"),
        ustring("log2"),
        ustring("log10"),
        ustring("logb"),
        ustring("exp"),
        ustring("exp2"),
        ustring("expm1"),
        ustring("erf"),
        ustring("erfc"),
        ustring("cbrt"),
        ustring("inversesqrt"),
        ustring("round"),
        ustring("trunc"),
        ustring("sign"),
        ustring("isnan"),
        ustring("isinf"),
        ustring("isfinite"),
        ustring("fabs"),
        ustring("cross"),
        ustring("distance"),
        ustring("area"),
        ustring("calculatenormal"),
        ustring("spline"),
        ustring("splineinverse"),
        ustring("blackbody"),
        ustring("wavelength_color"),
        ustring("luminance"),
        ustring("transformc"),
        ustring("raytype"),
        ustring("backfacing"),
        ustring("surfacearea"),
        ustring("getattribute"),
    };
    // clang-format on
    static const ustring readable_globals[] = {
        ustring("u"),    ustring("v"),  ustring("P"),
        ustring("N"),    ustring("Ng"), ustring("dPdu"),
        ustring("dPdv"), ustring("I"),  ustring("time"),
    };
    for (const Opcode& op : m_master->m_ops) {
        if (std::find(std::begin(supported), std::end(supported), op.opname())
            == std::end(supported)) {
            shadingsys().errorfmt("HART: unsupported operation '{}' in shader "
                                  "'{}' ({}:{})",
                                  op.opname(), shadername(), op.sourcefile(),
                                  op.sourceline());
            return false;
        }
        if (!geometry
            && (op.opname() == ustring("raytype")
                || op.opname() == ustring("backfacing")
                || op.opname() == ustring("surfacearea"))) {
            shadingsys().errorfmt(
                "HART: renderer lacks HARTGeometry for '{}' in shader '{}'",
                op.opname(), shadername());
            return false;
        }
        if (op.opname() == ustring("getattribute") && !validate_attribute(op))
            return false;
        const bool spatial = op.opname() == ustring("matrix")
                             || op.opname() == ustring("getmatrix")
                             || op.opname() == ustring("transform")
                             || op.opname() == ustring("transformv")
                             || op.opname() == ustring("transformn")
                             || op.opname() == ustring("point")
                             || op.opname() == ustring("vector")
                             || op.opname() == ustring("normal");
        if (spatial && !validate_transform(op))
            return false;
        if (op.opname() == ustring("texture") && !validate_texture(op))
            return false;
        if (op.opname() == ustring("closure") && !validate_closure(op))
            return false;
        if (op.opname() == ustring("hash") && !validate_hash(op))
            return false;
        const bool diagnostic = op.opname() == ustring("printf")
                                || op.opname() == ustring("warning")
                                || op.opname() == ustring("error");
        if (diagnostic && !validate_diagnostic(op))
            return false;
        if ((op.opname() == ustring("noise") || op.opname() == ustring("pnoise"))
            && !validate_noise(op))
            return false;
        if ((op.opname() == ustring("spline")
             || op.opname() == ustring("splineinverse"))
            && !validate_spline(op))
            return false;
        if ((op.opname() == ustring("color")
             || op.opname() == ustring("blackbody")
             || op.opname() == ustring("wavelength_color")
             || op.opname() == ustring("luminance")
             || op.opname() == ustring("transformc"))
            && !validate_color(op))
            return false;
        if (op.opname() == ustring("mxcompref")
            || op.opname() == ustring("mxcompassign")) {
            const int first = op.opname() == ustring("mxcompref") ? 2 : 1;
            for (int a = first; a < first + 2; ++a) {
                const Symbol& index
                    = m_master->m_symbols[m_master->m_args[op.firstarg() + a]];
                if (!index.typespec().is_int()
                    || (index.is_constant()
                        && (index.get_int() < 0 || index.get_int() >= 4))
                    || (!index.is_constant() && !bounds)) {
                    shadingsys().errorfmt(
                        "HART: matrix indices must be integers in [0,3]; "
                        "dynamic indices require HARTArrayBounds "
                        "in shader '{}' ({}:{})",
                        shadername(), op.sourcefile(), op.sourceline());
                    return false;
                }
            }
        }
        const bool array_ref        = op.opname() == ustring("aref");
        const bool array_assign     = op.opname() == ustring("aassign");
        const bool component_ref    = op.opname() == ustring("compref");
        const bool component_assign = op.opname() == ustring("compassign");
        const bool matrix_ref       = op.opname() == ustring("mxcompref");
        const bool matrix_assign    = op.opname() == ustring("mxcompassign");
        if (array_ref || array_assign || component_ref || component_assign
            || matrix_ref || matrix_assign) {
            const int first = array_ref || component_ref || matrix_ref ? 2 : 1;
            const int count = matrix_ref || matrix_assign ? 2 : 1;
            const Symbol& aggregate
                = m_master->m_symbols[m_master->m_args[op.firstarg()
                                                       + (first == 2 ? 1 : 0)]];
            const int length = array_ref || array_assign
                                   ? aggregate.typespec().arraylength()
                                   : (count == 2 ? 4 : 3);
            for (int a = first; a < first + count; ++a) {
                const Symbol& index
                    = m_master->m_symbols[m_master->m_args[op.firstarg() + a]];
                if (index.is_constant() && index.get_int() >= 0
                    && index.get_int() < length)
                    continue;
                if (!bounds || !m_master->range_checking()) {
                    shadingsys().errorfmt(
                        "HART: checked indexing requires HARTArrayBounds and "
                        "range_checking in shader '{}' ({}:{})",
                        shadername(), op.sourcefile(), op.sourceline());
                    return false;
                }
            }
        }
        for (int a = 0; a < op.nargs(); ++a) {
            const Symbol& sym
                = m_master->m_symbols[m_master->m_args[op.firstarg() + a]];
            if (op.opname() == ustring("closure") && sym.typespec().is_string())
                continue;  // Validated literal constructor name, not a string.
            if (op.opname() == ustring("texture") && sym.typespec().is_string())
                continue;
            if (a == 1
                && (op.opname() == ustring("spline")
                    || op.opname() == ustring("splineinverse")))
                continue;  // Validated immutable basis, not a device string.
            if ((op.opname() == ustring("color") && op.nargs() == 5 && a == 1)
                || (op.opname() == ustring("transformc") && (a == 1 || a == 2)))
                continue;  // Validated literal color spaces, not device strings.
            if (diagnostic && sym.typespec().is_string_based())
                continue;  // Literal format and typed argument payload.
            if (spatial && sym.typespec().is_string())
                continue;
            // Inlined function markers carry a name, not a device string.
            // The body remains subject to the same per-operation checks.
            if (op.opname() == ustring("functioncall") && op.nargs() == 1
                && sym.is_constant() && sym.typespec().is_string()
                && !op.argwrite(a))
                continue;
            // Selectors and option names were validated before optimization.
            if (sym.typespec().is_string()
                && (op.opname() == ustring("noise")
                    || op.opname() == ustring("pnoise"))) {
                continue;
            }
            if (!validate_type(sym))
                return false;
            if (sym.typespec().is_string_based()
                && !validate_string_operands(op))
                return false;
            if (sym.symtype() == SymTypeGlobal) {
                if (closures && sym.name() == ustring("Ci"))
                    continue;
                const bool readable = std::find(std::begin(readable_globals),
                                                std::end(readable_globals),
                                                sym.name())
                                      != std::end(readable_globals);
                if (geometry
                    && ((!op.argwrite(a)
                         && (sym.name() == ustring("dtime")
                             || sym.name() == ustring("dPdtime")))
                        || (readable && sym.name() != ustring("time"))))
                    continue;
                if (op.argwrite(a)) {
                    shadingsys().errorfmt("HART: writing shader global '{}' is "
                                          "unsupported in shader '{}'",
                                          sym.name(), shadername());
                    return false;
                }
                if (!readable) {
                    shadingsys().errorfmt("HART: unsupported shader global '{}' "
                                          "in shader '{}'",
                                          sym.name(), shadername());
                    return false;
                }
            }
        }
    }
    return true;
}



void
ShaderInstance::make_symbol_room(size_t moresyms)
{
    size_t oldsize = m_instsymbols.capacity();
    if (oldsize < m_instsymbols.size() + moresyms) {
        // Allocate a bit more than we need, so that most times we don't
        // need to reallocate.  But don't be wasteful by doubling or
        // anything like that, since we only expect a few to be added.
        const size_t extra_room = 10;
        size_t newsize          = m_instsymbols.size() + moresyms + extra_room;
        m_instsymbols.reserve(newsize);

        // adjust stats
        spin_lock lock(shadingsys().m_stat_mutex);
        size_t mem = (newsize - oldsize) * sizeof(Symbol);
        shadingsys().m_stat_mem_inst_syms += mem;
        shadingsys().m_stat_mem_inst += mem;
        shadingsys().m_stat_memory += mem;
    }
}



void
ShaderInstance::add_connection(int srclayer, const ConnectedParam& srccon,
                               const ConnectedParam& dstcon)
{
    // specialize symbol in case of dstcon is an unsized array
    if (dstcon.type.is_unsized_array()) {
        SymOverrideInfo* so = &m_instoverrides[dstcon.param];
        so->arraylen(srccon.type.arraylength());

        const TypeDesc& type = srccon.type.simpletype();
        // Skip structs for now, they're just placeholders
        /*if      (t.is_structure()) {
        }
        else*/
        if (type.basetype == TypeDesc::FLOAT) {
            so->dataoffset((int)m_fparams.size());
            expand(m_fparams, type.size());
        } else if (type.basetype == TypeDesc::INT) {
            so->dataoffset((int)m_iparams.size());
            expand(m_iparams, type.size());
        } else if (type.basetype == TypeDesc::STRING) {
            so->dataoffset((int)m_sparams.size());
            expand(m_sparams, type.size());
        } /* else if (t.is_closure()) {
            // Closures are pointers, so we allocate a string default taking
            // adventage of their default being NULL as well.
            so->dataoffset((int) m_sparams.size());
            expand (m_sparams, type.size());
        }*/
        else {
            OSL_DASSERT(0 && "unexpected type");
        }
    }

    off_t oldmem = vectorbytes(m_connections);
    m_connections.emplace_back(srclayer, srccon, dstcon);

    // adjust stats
    off_t mem = vectorbytes(m_connections) - oldmem;
    {
        spin_lock lock(shadingsys().m_stat_mutex);
        shadingsys().m_stat_mem_inst_connections += mem;
        shadingsys().m_stat_mem_inst += mem;
        shadingsys().m_stat_memory += mem;
    }
}



void
ShaderInstance::evaluate_writes_globals_and_userdata_params()
{
    writes_globals(false);
    userdata_params(false);
    for (auto&& s : symbols()) {
        if (s.symtype() == SymTypeGlobal && s.everwritten())
            writes_globals(true);
        if ((s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam)
            && !s.lockgeom() && !s.connected())
            userdata_params(true);
        if (s.symtype() == SymTypeTemp)  // Once we hit a temp, we'll never
            break;                       // see another global or param.
    }

    // In case this method is called before the Symbol vector is copied
    // (i.e. before copy_code_from_master is called), try to set
    // userdata_params as accurately as we can based on what we know from
    // the symbol overrides. This is very important to get instance merging
    // working correctly.
    for (auto&& s : m_instoverrides) {
        if (s.interpolated())
            userdata_params(true);
    }
}



void
ShaderInstance::copy_code_from_master(ShaderGroup& group)
{
    OSL_ASSERT(m_instops.empty() && m_instargs.empty());
    // reserve with enough room for a few insertions
    m_instops.reserve(master()->m_ops.size() + 10);
    m_instargs.reserve(master()->m_args.size() + 10);
    m_instops  = master()->m_ops;
    m_instargs = master()->m_args;

    // Copy the symbols from the master
    OSL_ASSERT(m_instsymbols.size() == 0
               && "should not have copied m_instsymbols yet");
    m_instsymbols = m_master->m_symbols;

    // Copy the instance override data
    // Also set the renderer_output flags where needed.
    OSL_ASSERT(m_instoverrides.size() == (size_t)std::max(0, lastparam()));
    OSL_ASSERT(m_instsymbols.size() >= (size_t)std::max(0, lastparam()));
    if (m_instoverrides.size()) {
        for (size_t i = 0, e = lastparam(); i < e; ++i) {
            Symbol* si = &m_instsymbols[i];
            if (m_instoverrides[i].valuesource() == Symbol::DefaultVal) {
                // Fix the length of any default-value variable length array
                // parameters.
                if (si->typespec().is_unsized_array())
                    si->arraylen(si->initializers());
            } else {
                if (m_instoverrides[i].arraylen())
                    si->arraylen(m_instoverrides[i].arraylen());
                si->valuesource(m_instoverrides[i].valuesource());
                si->connected_down(m_instoverrides[i].connected_down());
                si->interpolated(m_instoverrides[i].interpolated());
                si->interactive(m_instoverrides[i].interactive());
                si->dataoffset(m_instoverrides[i].dataoffset());
                si->set_dataptr(SymArena::Absolute, param_storage(i));
            }
            if (shadingsys().is_renderer_output(layername(), si->name(),
                                                &group)) {
                si->renderer_output(true);
                renderer_outputs(true);
            }
        }
    }
    evaluate_writes_globals_and_userdata_params();
    off_t symmem = vectorbytes(m_instsymbols) - vectorbytes(m_instoverrides);
    SymOverrideInfoVec().swap(m_instoverrides);  // free it

    // adjust stats
    {
        spin_lock lock(shadingsys().m_stat_mutex);
        shadingsys().m_stat_mem_inst_syms += symmem;
        shadingsys().m_stat_mem_inst += symmem;
        shadingsys().m_stat_memory += symmem;
    }
}



std::string
ConnectedParam::str(const ShaderInstance* inst, bool unmangle) const
{
    const Symbol* s = inst->symbol(param);
    return fmtformat("{}{}{} ({})",
                     unmangle ? s->unmangled() : string_view(s->name()),
                     arrayindex >= 0 ? fmtformat("[{}]", arrayindex)
                                     : std::string(),
                     channel >= 0 ? fmtformat("[{}]", channel) : std::string(),
                     type);
}



std::string
Connection::str(const ShaderGroup& group, const ShaderInstance* dstinst,
                bool unmangle) const
{
    return fmtformat("{} -> {}", src.str(group[srclayer], unmangle),
                     dst.str(dstinst, unmangle));
}



// Are the two vectors equivalent(a[i],b[i]) in each of their members?
template<class T>
inline bool
equivalent(const std::vector<T>& a, const std::vector<T>& b)
{
    if (a.size() != b.size())
        return false;
    typename std::vector<T>::const_iterator ai, ae, bi;
    for (ai = a.begin(), ae = a.end(), bi = b.begin(); ai != ae; ++ai, ++bi)
        if (!equivalent(*ai, *bi))
            return false;
    return true;
}



/// Are two symbols equivalent (from the point of view of merging
/// shader instances)?  Note that this is not a true ==, it ignores
/// the m_data, m_node, and m_alias pointers!
static bool
equivalent(const Symbol& a, const Symbol& b)
{
    // If they aren't used, don't consider them a mismatch
    if (!a.everused() && !b.everused())
        return true;

    // Different symbol types or data types are a mismatch
    if (a.symtype() != b.symtype() || a.typespec() != b.typespec())
        return false;

    // Don't consider different names to be a mismatch if the symbol
    // is a temp or constant.
    if (a.symtype() != SymTypeTemp && a.symtype() != SymTypeConst
        && a.name() != b.name())
        return false;
    // But constants had better match their values!
    if (a.symtype() == SymTypeConst
        && memcmp(a.data(), b.data(), a.typespec().simpletype().size()))
        return false;

    return a.has_derivs() == b.has_derivs() && a.lockgeom() == b.lockgeom()
           && a.valuesource() == b.valuesource() && a.fieldid() == b.fieldid()
           && a.initbegin() == b.initbegin() && a.initend() == b.initend();
}



bool
ShaderInstance::mergeable(const ShaderInstance& b,
                          const ShaderGroup& /*g*/) const
{
    // Must both be instances of the same master -- very fast early-out
    // for most potential pair comparisons.
    if (master() != b.master())
        return false;

    // If one or both instances are directly hooked up to renderer outputs,
    // don't merge them.
    if (renderer_outputs() || b.renderer_outputs())
        return false;

    // If the shaders haven't been optimized yet, they don't yet have
    // their own symbol tables and instructions (they just refer to
    // their unoptimized master), but they may have an "instance
    // override" vector that describes which parameters have
    // instance-specific values or connections.
    bool optimized = (m_instsymbols.size() != 0 || m_instops.size() != 0);

    // Same instance overrides
    if (m_instoverrides.size() || b.m_instoverrides.size()) {
        OSL_ASSERT(!optimized);  // should not be post-opt
        OSL_ASSERT(m_instoverrides.size() == b.m_instoverrides.size());
        for (size_t i = 0, e = m_instoverrides.size(); i < e; ++i) {
            if ((m_instoverrides[i].valuesource() == Symbol::DefaultVal
                 || m_instoverrides[i].valuesource() == Symbol::InstanceVal)
                && (b.m_instoverrides[i].valuesource() == Symbol::DefaultVal
                    || b.m_instoverrides[i].valuesource()
                           == Symbol::InstanceVal)) {
                // If both params are defaults or instances, let the
                // instance parameter value checking below handle
                // things. No need to reject default-vs-instance
                // mismatches if the actual values turn out to be the
                // same later.
                continue;
            }

            if (!(equivalent(m_instoverrides[i], b.m_instoverrides[i]))) {
                const Symbol* sym  = mastersymbol(i);  // remember, it's pre-opt
                const Symbol* bsym = b.mastersymbol(i);
                if (!sym->everused_in_group() && !bsym->everused_in_group())
                    continue;
                return false;
            }
            // But still, if they differ in whether they are interpolated or
            // interactive, we can't merge the instances.
            if (m_instoverrides[i].interpolated()
                    != b.m_instoverrides[i].interpolated()
                || m_instoverrides[i].interactive()
                       != b.m_instoverrides[i].interactive()) {
                return false;
            }
        }
    }

    // Make sure that the two nodes have the same parameter values.  If
    // the group has already been optimized, it's got an
    // instance-specific symbol table to check; but if it hasn't been
    // optimized, we check the symbol table in the master.
    for (int i = firstparam(); i < lastparam(); ++i) {
        const Symbol* sym = optimized ? symbol(i) : mastersymbol(i);
        if (!sym->everused_in_group())
            continue;
        if (sym->typespec().is_closure())
            continue;  // Closures can't have instance override values
        // Even if the symbols' values match now, they might not in the
        // future with 'interactive' parameters.
        const Symbol* b_sym = optimized ? b.symbol(i) : b.mastersymbol(i);
        if ((sym->valuesource() == Symbol::InstanceVal
             || sym->valuesource() == Symbol::DefaultVal)
            && (memcmp(param_storage(i), b.param_storage(i),
                       sym->typespec().simpletype().size())
                || b_sym->interactive())) {
            return false;
        }
    }

    if (run_lazily() != b.run_lazily()) {
        return false;
    }

    // The connection list need to be the same for the two shaders.
    if (m_connections.size() != b.m_connections.size()) {
        return false;
    }
    if (m_connections != b.m_connections) {
        return false;
    }

    // Make sure system didn't ask for instances that query userdata to be
    // immune from instance merging.
    if (!shadingsys().m_opt_merge_instances_with_userdata
        && (userdata_params() || b.userdata_params())) {
        return false;
    }

    // If there are no "local" ops or symbols, this instance hasn't been
    // optimized yet.  In that case, we've already done enough checking,
    // since the masters being the same and having the same instance
    // params and connections is all it takes.  The rest (below) only
    // comes into play after instances are more fully elaborated from
    // their masters in order to be optimized.
    if (!optimized) {
        return true;
    }

    // Same symbol table
    if (!equivalent(m_instsymbols, b.m_instsymbols)) {
        return false;
    }

    // Same opcodes to run
    if (!equivalent(m_instops, b.m_instops)) {
        return false;
    }
    // Same arguments to the ops
    if (m_instargs != b.m_instargs) {
        return false;
    }

    // Parameter and code ranges
    if (m_firstparam != b.m_firstparam || m_lastparam != b.m_lastparam
        || m_maincodebegin != b.m_maincodebegin
        || m_maincodeend != b.m_maincodeend || m_Psym != b.m_Psym
        || m_Nsym != b.m_Nsym) {
        return false;
    }

    // Nothing left to check, they must be identical!
    return true;
}


};  // namespace pvt



ShaderGroup::ShaderGroup(string_view name, ShadingSystemImpl& shadingsys)
    : m_shadingsys(shadingsys)
{
    m_id = ++(*(atomic_int*)&next_id);
    if (name.size()) {
        m_name = name;
    } else {
        // No name -- make one up using the unique
        m_name = ustring::fmtformat("unnamed_group_{}", m_id);
    }
}



ShaderGroup::~ShaderGroup()
{
#if 0
    if (m_layers.size()) {
        ustring name = m_layers.back()->layername();
        std::cerr << "Shader group " << this 
                  << " id #" << m_layers.back()->id() << " (" 
                  << (name.c_str() ? name.c_str() : "<unnamed>")
                  << ") executed on " << executions() << " points\n";
    } else {
        std::cerr << "Shader group " << this << " (no layers?) " 
                  << "executed on " << executions() << " points\n";
    }
#endif

    // Free any GPU memory associated with this group
    if (m_device_interactive_arena)
        shadingsys().renderer()->device_free(
            m_device_interactive_arena.d_get());

    // Unload the BackendCpp-compiled DSO, if one was loaded.
    if (m_cpp_dso_handle)
        OIIO::Plugin::close(m_cpp_dso_handle);
}



int
ShaderGroup::find_layer(ustring layername) const
{
    int i;
    for (i = nlayers() - 1; i >= 0 && layer(i)->layername() != layername; --i)
        ;
    return i;  // will be -1 if we never found a match
}



const Symbol*
ShaderGroup::find_symbol(ustring layername, ustring symbolname) const
{
    for (int layer = nlayers() - 1; layer >= 0; --layer) {
        const ShaderInstance* inst(m_layers[layer].get());
        if (layername.size() && layername != inst->layername())
            continue;  // They asked for a specific layer and this isn't it
        int symidx = inst->findsymbol(symbolname);
        if (symidx >= 0)
            return inst->symbol(symidx);
    }
    return NULL;
}



void
ShaderGroup::clear_entry_layers()
{
    for (int i = 0; i < nlayers(); ++i)
        m_layers[i]->entry_layer(false);
    m_num_entry_layers = 0;
}



void
ShaderGroup::mark_entry_layer(int layer)
{
    if (layer >= 0 && layer < nlayers() && !m_layers[layer]->entry_layer()) {
        m_layers[layer]->entry_layer(true);
        ++m_num_entry_layers;
    }
}



void
ShaderGroup::setup_interactive_arena(cspan<uint8_t> paramblock)
{
    if (paramblock.size()) {
        // CPU side
        m_interactive_arena_size = paramblock.size();
        m_interactive_arena.reset(new uint8_t[m_interactive_arena_size]);
        memcpy(m_interactive_arena.get(), paramblock.data(),
               m_interactive_arena_size);
        if (shadingsys().use_optix()) {
            // GPU side
            RendererServices* rs = shadingsys().renderer();
            m_device_interactive_arena.reset(reinterpret_cast<uint8_t*>(
                rs->device_alloc(m_interactive_arena_size)));
            rs->copy_to_device(m_device_interactive_arena.d_get(),
                               paramblock.data(), m_interactive_arena_size);
            // print("group {} has device interactive_params set to {:p}\n",
            //       name(), m_device_interactive_arena.d_get());
        }
        if (shadingsys().use_hart())
            upload_hart_interactive(0, paramblock);
    } else {
        m_interactive_arena_size = 0;
        m_interactive_arena.reset();
        m_device_interactive_arena.reset();
    }
}



bool
ShaderGroup::upload_hart_interactive(size_t offset, cspan<uint8_t> data)
{
    auto& ss = shadingsys();
    if (offset > m_interactive_arena_size
        || data.size() > m_interactive_arena_size - offset || !data.data()) {
        ss.errorfmt("HART: invalid interactive parameter arena range");
        return false;
    }
    auto* rs = ss.renderer();
    if (!m_device_interactive_arena) {
        m_device_interactive_arena_valid = false;
        m_device_interactive_arena.reset(
            static_cast<uint8_t*>(rs->device_alloc(m_interactive_arena_size)));
        if (!m_device_interactive_arena) {
            ss.errorfmt("HART: failed to allocate interactive parameters");
            return false;
        }
    }
    // A failed copy may have modified part of the device arena. Recover from
    // the last committed host mirror, not just the next parameter's bytes.
    std::vector<uint8_t> recovery;
    if (!m_device_interactive_arena_valid
        && (offset || data.size() != m_interactive_arena_size)) {
        recovery.assign(m_interactive_arena.get(),
                        m_interactive_arena.get() + m_interactive_arena_size);
        memcpy(recovery.data() + offset, data.data(), data.size());
        offset = 0;
        data   = recovery;
    }
    auto* destination = m_device_interactive_arena.d_get() + offset;
    m_device_interactive_arena_valid
        = rs->copy_to_device(destination, data.data(), data.size())
          == destination;
    if (!m_device_interactive_arena_valid)
        ss.errorfmt("HART: failed to upload interactive parameters");
    return m_device_interactive_arena_valid;
}



void
ShaderGroup::generate_optix_cache_key(string_view code)
{
    const uint64_t ir_key = Strutil::strhash(code);

    std::string safegroup;
    safegroup = Strutil::replace(name(), "/", "_", true);
    safegroup = Strutil::replace(safegroup, ":", "_", true);

    ShaderInstance* inst = layer(nlayers() - 1);
    ustring layername    = inst->layername();

    // Cache key includes group and entry layer names in addition to the serialized IR.
    // This is because the group and layer names make their way into the ptx's
    // direct callable name, but isn't included in the serialization.
    std::string cache_key = fmtformat("cache-osl-ptx-{}-{}-{}", safegroup,
                                      layername, ir_key);

    m_optix_cache_key = cache_key;
}



std::string
ShaderGroup::serialize() const
{
    std::ostringstream out;
    out.imbue(std::locale::classic());  // force C locale
    out.precision(9);
    lock_guard lock(m_mutex);
    for (int i = 0, nl = nlayers(); i < nl; ++i) {
        const ShaderInstance* inst = m_layers[i].get();

        bool dstsyms_exist = inst->symbols().size();
        for (int p = 0; p < inst->lastparam(); ++p) {
            const Symbol* s = dstsyms_exist ? inst->symbol(p)
                                            : inst->mastersymbol(p);
            OSL_ASSERT(s);
            if (!s
                || (s->symtype() != SymTypeParam
                    && s->symtype() != SymTypeOutputParam))
                continue;
            Symbol::ValueSource vs = dstsyms_exist
                                         ? s->valuesource()
                                         : inst->instoverride(p)->valuesource();
            if (vs == Symbol::InstanceVal) {
                TypeDesc type = s->typespec().simpletype();
                int offset    = s->dataoffset();
                if (type.is_unsized_array() && !dstsyms_exist) {
                    // If we're being asked to serialize a group that isn't
                    // yet optimized, any "unsized" arrays will have their
                    // concrete length and offset in the SymOverrideInfo,
                    // not in the Symbol belonging to the instance.
                    type.arraylen = inst->instoverride(p)->arraylen();
                    offset        = inst->instoverride(p)->dataoffset();
                }
                out << "param " << type << ' ' << s->name();
                int nvals = type.numelements() * type.aggregate;
                if (type.basetype == TypeDesc::INT) {
                    const int* vals = &inst->m_iparams[offset];
                    for (int i = 0; i < nvals; ++i)
                        out << ' ' << vals[i];
                } else if (type.basetype == TypeDesc::FLOAT) {
                    const float* vals = &inst->m_fparams[offset];
                    for (int i = 0; i < nvals; ++i)
                        out << ' ' << vals[i];
                } else if (type.basetype == TypeDesc::STRING) {
                    const ustring* vals = &inst->m_sparams[offset];
                    for (int i = 0; i < nvals; ++i)
                        out << ' ' << '\"' << Strutil::escape_chars(vals[i])
                            << '\"';
                } else {
                    OSL_ASSERT_MSG(0, "unknown type for serialization: %s (%s)",
                                   type.c_str(), s->typespec().c_str());
                }
                if (dstsyms_exist ? s->interpolated()
                                  : inst->instoverride(p)->interpolated())
                    print(out, " [[int interpolated=1]]");
                if (dstsyms_exist ? s->interactive()
                                  : inst->instoverride(p)->interactive())
                    print(out, " [[int interactive=1]]");
                out << " ;\n";
            }
        }
        out << "shader " << inst->shadername() << ' ' << inst->layername()
            << " ;\n";
        for (int c = 0, nc = inst->nconnections(); c < nc; ++c) {
            const Connection& con(inst->connection(c));
            OSL_ASSERT(con.srclayer >= 0);
            const ShaderInstance* srclayer = m_layers[con.srclayer].get();
            OSL_ASSERT(srclayer);
            ustring srclayername = srclayer->layername();
            OSL_ASSERT(con.src.param >= 0 && con.dst.param >= 0);
            bool srcsyms_exist = srclayer->symbols().size();
            ustring srcparam
                = srcsyms_exist ? srclayer->symbol(con.src.param)->name()
                                : srclayer->mastersymbol(con.src.param)->name();
            ustring dstparam = dstsyms_exist
                                   ? inst->symbol(con.dst.param)->name()
                                   : inst->mastersymbol(con.dst.param)->name();
            // FIXME: Assertions to be sure we don't yet support individual
            // channel or array element connections. Fix eventually.
            OSL_ASSERT(con.src.arrayindex == -1 && con.src.channel == -1);
            OSL_ASSERT(con.dst.arrayindex == -1 && con.dst.channel == -1);
            out << "connect " << srclayername << '.' << srcparam << ' '
                << inst->layername() << '.' << dstparam << " ;\n";
        }
    }
    return out.str();
}


OSL_NAMESPACE_END
