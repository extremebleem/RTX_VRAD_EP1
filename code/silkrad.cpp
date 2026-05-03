#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <memory>

#include "cuda.h"

#include "bsp.h"
#include "cudabsp.h"
#include "cudarad.h"
#include "fxaa.h"

#include "cudautils.h"

struct CommandLineOptions {
    std::string inputFilename;
    std::string outputFilename;
    std::string gameRoot;
};

static std::string infer_asset_root_from_bsp(const std::string& filename) {
    std::string normalized = filename;
    for (char& c : normalized) {
        if (c == '/') {
            c = '\\';
        }
    }

    const std::string mapsToken = "\\maps\\";
    size_t mapsPos = normalized.rfind(mapsToken);
    if (mapsPos != std::string::npos) {
        return normalized.substr(0, mapsPos);
    }

    size_t slashPos = normalized.find_last_of('\\');
    if (slashPos != std::string::npos) {
        return normalized.substr(0, slashPos);
    }

    return ".";
}

static void print_usage(void) {
    std::cerr
        << "Usage: SilkRAD.exe <input.bsp> [output.bsp] -game <mod_root_or_gameinfo.gi>"
        << std::endl
        << "Example: SilkRAD.exe D:\\games\\CSS_LOVE\\cstrike\\maps\\de_brigia_hvh.bsp "
        << "D:\\games\\CSS_LOVE\\cstrike\\maps\\out.bsp -game D:\\games\\CSS_LOVE\\cstrike"
        << std::endl
        << "Example: SilkRAD.exe D:\\games\\CSS_LOVE\\cstrike\\maps\\de_brigia_hvh.bsp "
        << "D:\\games\\CSS_LOVE\\cstrike\\maps\\out.bsp -game D:\\games\\CSS_LOVE\\cstrike\\gameinfo.gi"
        << std::endl;
}

static bool parse_command_line(
    int argc,
    char** argv,
    CommandLineOptions& options
) {
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        if (arg == "-game" || arg == "--game") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << arg << "." << std::endl;
                return false;
            }

            options.gameRoot = argv[++i];
            continue;
        }

        positional.push_back(arg);
    }

    if (positional.empty()) {
        return false;
    }

    options.inputFilename = positional[0];
    options.outputFilename =
        positional.size() >= 2
            ? positional[1]
            : std::string("D:\\games\\CSS_LOVE\\cstrike\\maps\\out.bsp");

    if (options.gameRoot.empty() && positional.size() >= 3) {
        options.gameRoot = positional[2];
    }

    if (positional.size() > 3) {
        std::cerr << "Too many positional arguments." << std::endl;
        return false;
    }

    return true;
}


void print_cudainfo(void) {
    int device;
    CUDA_CHECK_ERROR(cudaGetDevice(&device));

    cudaDeviceProp deviceProps;
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&deviceProps, device));

    std::cout << "CUDA Device: " << deviceProps.name << std::endl;
    std::cout << "    Device Memory: "
        << deviceProps.totalGlobalMem << std::endl;
    std::cout << "    Max Threads/Block: "
        << deviceProps.maxThreadsPerBlock << std::endl;
    std::cout << "    Max Block Dim X: "
        << deviceProps.maxThreadsDim[0] << std::endl;
    std::cout << "    Max Block Dim Y: "
        << deviceProps.maxThreadsDim[1] << std::endl;
    std::cout << "    Max Block Dim Z: "
        << deviceProps.maxThreadsDim[2] << std::endl;
    std::cout << "    Max Grid Size X: "
        << deviceProps.maxGridSize[0] << std::endl;
    std::cout << "    Max Grid Size Y: "
        << deviceProps.maxGridSize[1] << std::endl;
    std::cout << "    Max Grid Size Z: "
        << deviceProps.maxGridSize[2] << std::endl;
    std::cout << std::endl;
}


int main(int argc, char** argv) {
    CommandLineOptions options;
    if (!parse_command_line(argc, argv, options)) {
        std::cerr << "Invalid arguments." << std::endl;
        print_usage();
        return 1;
    }

    std::cout << "SilkRAD -- GPU-Accelerated Radiosity Simulator" << std::endl;

    const std::string filename(options.inputFilename);
    const std::string outputFilename(options.outputFilename);
    std::ifstream f(filename, std::ios::binary);

    std::unique_ptr<BSP::BSP> pBSP;

    try {
        pBSP = std::unique_ptr<BSP::BSP>(new BSP::BSP(filename));
    }
    catch (BSP::InvalidBSP e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    //pBSP->dump_lumps();

    if (!pBSP->has_visibility_data()) {
        std::cerr
            << "ERROR: BSP file " << filename << " has no visibility matrix!"
                << std::endl
            << "SilkRAD does not support BSPs without visibility data."
                << std::endl
            << "Please run VIS on the map before continuing. "
            << "If you are positive that VIS was performed on this map, "
            << "check for leaks." << std::endl;
        return 1;
    }

    /*
     * HACK!
     * Disable normal maps throughout the entire BSP, because I didn't
     * implement them and we don't have time.
     */
    for (const BSP::TexInfo& texInfo : pBSP->get_texinfos()) {
        BSP::TexInfo& ti = const_cast<BSP::TexInfo&>(texInfo);
        ti.flags &= ~BSP::SURF_BUMPLIGHT;
    }

    pBSP->build_worldlights();
    pBSP->clear_baked_lighting();

    if (options.gameRoot.empty()) {
        options.gameRoot = infer_asset_root_from_bsp(filename);
        std::cout
            << "WARNING: -game was not specified; falling back to inferred game root: "
            << options.gameRoot << std::endl;
    }
    else {
        std::cout << "Game root: " << options.gameRoot << std::endl;
    }
    CUDARAD::set_asset_root(options.gameRoot);

    print_cudainfo();

    CUDA_CHECK_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

    std::cout << "Copy BSP to device memory..." << std::endl;
    CUDABSP::CUDABSP* pCudaBSP = CUDABSP::make_cudabsp(*pBSP);
    CUDABSP::clear_lighting(pCudaBSP);

    std::cout << "Initialize radiosity subsystem..." << std::endl;
    CUDARAD::init(*pBSP);

    std::cout << "*** Start RAD! ***" << std::endl;

    std::cout << "Compute direct lighting..." << std::endl;
    CUDARAD::compute_direct_lighting(*pBSP, pCudaBSP);

    //std::cout << "Run lightmap FXAA passes..." << std::endl;
    //const size_t NUM_FXAA_PASSES = 5;
    //for (size_t i = 0; i<NUM_FXAA_PASSES; i++) {
    //    std::cout << "    Pass "
    //        << i + 1 << "/" << NUM_FXAA_PASSES << "..."
    //        << std::endl;
    //
    //    CUDAFXAA::antialias_lightsamples(pCudaBSP);
    //}
    //std::cout << "Done!" << std::endl;

    std::cout << "Run direct lighting antialiasing pass..." << std::endl;
    //CUDARAD::antialias_direct_lighting(*pBSP, pCudaBSP);

    std::cout << "Compute light bounces..." << std::endl;
    //CUDARAD::bounce_lighting(*pBSP, pCudaBSP);

    std::cout << "Compute ambient lighting..." << std::endl;
    //CUDARAD::compute_leaf_ambient(pCudaBSP);

    std::cout << "Convert light samples to RGBExp32..." << std::endl;
    CUDABSP::convert_lightsamples(pCudaBSP);

    std::cout << "Update host BSP data..." << std::endl;
    CUDABSP::update_bsp(*pBSP, pCudaBSP);

    CUDABSP::destroy_cudabsp(pCudaBSP);

    /*
     * Mark the BSP as non-fullbright.
     *
     * This tells the engine that there is actually lighting information
     * embedded in the map.
     */
    pBSP->set_fullbright(false);

    pBSP->write(outputFilename);

    std::cout << "Wrote to file \"" << outputFilename << "\"." << std::endl;

    /* Tear down the radiosity subsystem. */
    CUDARAD::cleanup();

    return 0;
}
