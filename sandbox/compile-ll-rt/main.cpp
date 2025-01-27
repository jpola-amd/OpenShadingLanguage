#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/MC/TargetRegistry.h>

// TargetLibraryInfoImpl
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/IR/Verifier.h>

#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/TargetPassConfig.h>

#include <llvm/Support/WithColor.h>

#include <llvm/PassRegistry.h>
#include <llvm/InitializePasses.h>

//lld
#include <lld/Common/Driver.h>
#include <lld/Common/Filesystem.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>


bool validateGfxArch(llvm::StringRef gfxArch)
{
    if (gfxArch.size() != 6 && gfxArch.size() != 7) {
        llvm::errs() << "Error: invalid gfx arch: " << gfxArch << "\n";
        return false;
    }
    if (!gfxArch.starts_with("gfx")) {
        llvm::errs() << "Error: invalid gfx arch: " << gfxArch << "\n";
        return false;
    }

    int gfxArchNum;
    gfxArch.substr(3).getAsInteger(10, gfxArchNum);

    constexpr int gfxArchMin = 900;
    constexpr int gfxArchMax = 1201;

    if (gfxArchNum < gfxArchMin || gfxArchNum > gfxArchMax) {
        llvm::errs() << "Error: invalid gfx arch: " << gfxArch << "\n";
        return false;
    }

    return true;
}


LLD_HAS_DRIVER(elf)


int main(int argc, char* argv[])
{
    // i want to have the <gfx arch> to be defaulted to gfx1036
    if (argc < 4) {
        llvm::errs() << "Usage: " << argv[0] << " <input.llf> <output.asm|o> <gfx arch>\n";
        return EXIT_FAILURE;
    }

    llvm::StringRef gfxArch = (argc > 3) ? argv[3] : "gfx1036";

    if (!validateGfxArch(gfxArch)) {
        return EXIT_FAILURE;
    }

    llvm::StringRef inputFilename = argv[1];
    // check if the input file exists
    if (!llvm::sys::fs::exists(inputFilename)) {
        llvm::errs() << "Error: file not found: " << inputFilename << "\n";
        return EXIT_FAILURE;
    }


    llvm::StringRef outputFileName = argv[2];

    const llvm::StringRef outputObjExt = ".o";
    const llvm::StringRef outputAsmExt = ".s";

    if (!outputFileName.ends_with(outputObjExt) && !outputFileName.ends_with(outputAsmExt)) {
        llvm::errs() << "Error: invalid output file extension: " << outputFileName << "\n";
        return EXIT_FAILURE;
    }
    llvm::CodeGenFileType fileType {llvm::CodeGenFileType::Null};
    bool Binary = true;
    if (outputFileName.ends_with(outputObjExt)) {
        fileType = llvm::CodeGenFileType::ObjectFile;
    }
    else if (outputFileName.ends_with(outputAsmExt)) {
        fileType = llvm::CodeGenFileType::AssemblyFile;
        Binary = false;
    }


    // print input arguments
    llvm::outs() << "Begin Test to compile to amdgcn: " << argc << "\n";
    llvm::outs() << "Input file: " << argv[1] << "\n";
    llvm::outs() << "Output files: " << argv[2] << "\n";
    llvm::outs() << "offload arch: " << gfxArch << "\n";
    llvm::outs() << "file type: " << (fileType == llvm::CodeGenFileType::ObjectFile ? "Obj" : "Asm") << "\n";

    // Initialize LLVM
    {
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmPrinters();
        llvm::InitializeAllAsmParsers();

        llvm::PassRegistry *Registry = llvm::PassRegistry::getPassRegistry();
        llvm::initializeCore(*Registry);
        llvm::initializeCodeGen(*Registry);
        llvm::initializeLoopStrengthReducePass(*Registry);
        llvm::initializeLowerIntrinsicsPass(*Registry);
        llvm::initializePostInlineEntryExitInstrumenterPass(*Registry);
        llvm::initializeUnreachableBlockElimLegacyPassPass(*Registry);
        llvm::initializeConstantHoistingLegacyPassPass(*Registry);
        llvm::initializeScalarOpts(*Registry);
        llvm::initializeVectorization(*Registry);
        llvm::initializeScalarizeMaskedMemIntrinLegacyPassPass(*Registry);
        llvm::initializeExpandReductionsPass(*Registry);
        llvm::initializeExpandVectorPredicationPass(*Registry);
        llvm::initializeHardwareLoopsLegacyPass(*Registry);
        llvm::initializeTransformUtils(*Registry);
        llvm::initializeReplaceWithVeclibLegacyPass(*Registry);
        llvm::initializeTLSVariableHoistLegacyPassPass(*Registry);
        // Initialize debugging passes.
        llvm::initializeScavengerTestPass(*Registry);
    }
    
    llvm::LLVMContext Context;
    llvm::SMDiagnostic Err;

    // Set the target options
    llvm::TargetOptions TargetOptions;
    {
        TargetOptions.AllowFPOpFusion = llvm::FPOpFusion::Standard;
        TargetOptions.UnsafeFPMath                           = 1;
        TargetOptions.NoInfsFPMath                           = 1;
        TargetOptions.NoNaNsFPMath                           = 1;
        TargetOptions.HonorSignDependentRoundingFPMathOption = 0;
        TargetOptions.FloatABIType          = llvm::FloatABI::Default;
        TargetOptions.AllowFPOpFusion       = llvm::FPOpFusion::Fast;
        TargetOptions.NoZerosInBSS          = 0;
        TargetOptions.GuaranteedTailCallOpt = 0;
        TargetOptions.UseInitArray = 0;

    }

    // Setup the Target
    llvm::StringRef TargetTripleStr = "amdgcn-unknown-linux-gnu";
    llvm::Triple TheTriple(TargetTripleStr);

    std::string ErrStr;
    const llvm::Target* TargetPtr = llvm::TargetRegistry::lookupTarget(TargetTripleStr, ErrStr);
    if (!TargetPtr) {
        Err.print(argv[0], llvm::errs());
        return EXIT_FAILURE;
    }
    else {
        llvm::outs() << "Target found: " << TargetPtr->getName() << "\n";
    }

    std::string FeaturesStr = "";
    std::unique_ptr<llvm::TargetMachine> TargetMachine = std::unique_ptr<llvm::TargetMachine>(TargetPtr->createTargetMachine(
            TargetTripleStr, gfxArch, FeaturesStr,
            TargetOptions,
            llvm::Reloc::PIC_,
            llvm::CodeModel::Small,
            llvm::CodeGenOptLevel::Default
    ));
    if (!TargetMachine) {
        llvm::errs() << "Error: failed to create target machine\n";
        return EXIT_FAILURE;
    }

    llvm::errs() << "Data layout: " << TargetMachine->createDataLayout().getStringRepresentation()  << "\n";

    // Print out the settings of the TargetMachine.
    llvm::errs() << "Target Machine Settings:\n";
    llvm::errs() << "\t*Target Triple: " << TargetMachine->getTargetTriple().str() << "\n";
    llvm::errs() << "\t*CPU: " << TargetMachine->getTargetCPU() << "\n";
    llvm::errs() << "\t*Feature String: " << TargetMachine->getTargetFeatureString() << "\n";
    llvm::errs() << "\t*Relocation Model: " << TargetMachine->getRelocationModel() << "\n";
    llvm::errs() << "\t*Code Model: " << TargetMachine->getCodeModel() << "\n";
    llvm::errs() << "\t*Opt Level: " << static_cast<int>(TargetMachine->getOptLevel()) << "\n";


    auto SetDataLayout = [&](llvm::StringRef DataLayoutTargetTriple, llvm::StringRef OldString) -> std::optional<std::string> {
        llvm::errs() << "Setting Data Layout\n";
        llvm::errs() << "Switching from " << OldString << " to " << DataLayoutTargetTriple << "\n";
        return TargetMachine->createDataLayout().getStringRepresentation();
    };

    std::unique_ptr<llvm::Module> M = llvm::parseIRFile(inputFilename, Err, Context, llvm::ParserCallbacks(SetDataLayout));

    if (!M) {
        Err.print(argv[0], llvm::errs());
        return EXIT_FAILURE;
    }

    M->setTargetTriple(TargetMachine->getTargetTriple().str());

    llvm::errs() << "Module Data layout: " << M->getDataLayoutStr() << "\n";
    llvm::errs() << "Module Target Triple: " << M->getTargetTriple() << "\n";

    // Figure out where we are going to send the output.
    std::error_code EC;
    llvm::sys::fs::OpenFlags openFlags = llvm::sys::fs::OF_None;
    if (!Binary)
        openFlags |= llvm::sys::fs::OF_TextWithCRLF;

    auto outputFileDescriptor = std::make_unique<llvm::ToolOutputFile>(outputFileName, EC, openFlags);
    if (EC) {
        llvm::errs() << "Error: failed to open " << outputFileName << "\n";
        return EXIT_FAILURE;
    }

    auto debugOut = std::make_unique<llvm::ToolOutputFile>("example.dbg.log", EC, llvm::sys::fs::OF_None);
    if (EC)
    {
        llvm::errs() << "Error: failed to open example.dbg.log\n";
        return EXIT_FAILURE;
    }


    if (llvm::verifyModule(*M, &llvm::errs()))
    {
        llvm::errs() << "Error: failed to verify module\n";
        return EXIT_FAILURE;
    }


    // Ensure the filename is passed down to CodeViewDebug.
    TargetMachine->Options.ObjectFilenameForDebug = debugOut->outputFilename();


    llvm::TargetLibraryInfoImpl TLII(TargetMachine->getTargetTriple());
    llvm::LLVMTargetMachine &LLVMTM = static_cast<llvm::LLVMTargetMachine &>(*TargetMachine);
    llvm::MachineModuleInfoWrapperPass *MMIWP = new llvm::MachineModuleInfoWrapperPass(&LLVMTM);

    llvm::legacy::PassManager PassManager;

    PassManager.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

    bool NoVerify = false;

    if (TargetMachine->addPassesToEmitFile(
            PassManager,
            outputFileDescriptor->os(),
            nullptr,
            fileType,
            NoVerify,
            MMIWP))
    {
        llvm::errs() << "Error: target does not support generation of this file type(asm)\n";
        return EXIT_FAILURE;
    }

    PassManager.run(*M);

    if (Context.getDiagHandlerPtr()->HasErrors)
    {
        // print the errors.
        llvm::errs() << "Error: failed to run pass manager\n";
        return EXIT_FAILURE;
    }


    // Declare success.
    outputFileDescriptor->keep();
    //lld part
    {
         std::vector<const char *> Args = {
            "ld.lld",
            "-shared",
            "-o", "output.hsaco",
            outputFileDescriptor->outputFilename().c_str()
        };

        lld::Result res = lld::lldMain(Args, llvm::outs(), llvm::errs(), {lld::DriverDef{lld::Gnu, &lld::elf::link}});
        if (res.retCode != 0 && res.canRunAgain)
        {
            llvm::errs() << "Error: failed to link\n";
            return EXIT_FAILURE;
        }
    }

    llvm::raw_fd_ostream ofs("output.hsaco", EC, llvm::sys::fs::OF_None);

    llvm::outs() << "Program finished successfully\n";
    return EXIT_SUCCESS;;
}