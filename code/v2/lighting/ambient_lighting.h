#pragma once

#include <vector>

#include "../../bsp.h"
#include "../../cudabsp.h"
#include "../../raytracer_optix.h"

namespace SilkRAD::V2::Bridge {
    struct BackendState;
}

namespace SilkRAD::V2::Lighting {
    void compute_leaf_ambient_from_cuda_runtime(
        const ::BSP::BSP& bsp,
        ::CUDABSP::CUDABSP* pCudaBSP,
        const Bridge::BackendState& state,
        OptixRT::OptixSunLosTracer& tracer
    );

    void compute_leaf_ambient_runtime(
        const ::BSP::BSP& bsp,
        const Bridge::BackendState& state,
        const std::vector<float3>& lightSamples,
        const std::vector<float3>& faceAverages,
        std::vector<::BSP::CompressedLightCube>& leafAmbient,
        OptixRT::OptixSunLosTracer& tracer
    );
}
