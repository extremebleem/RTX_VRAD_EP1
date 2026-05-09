#pragma once

#include <vector>

#include "../../bsp.h"
#include "../../cudabsp.h"
#include "../../raytracer_optix.h"

namespace SilkRAD::Core {
    struct RuntimeState;
}

namespace SilkRAD::Core::Lighting {
    void compute_leaf_ambient_from_cuda_runtime(
        const ::BSP::BSP& bsp,
        ::CUDABSP::CUDABSP* pCudaBSP,
        const RuntimeState& state,
        OptixRT::OptixSunLosTracer& tracer
    );

    void compute_leaf_ambient_runtime(
        const ::BSP::BSP& bsp,
        const RuntimeState& state,
        const std::vector<float3>& lightSamples,
        const std::vector<float3>& faceAverages,
        std::vector<::BSP::CompressedLightCube>& leafAmbient,
        OptixRT::OptixSunLosTracer& tracer
    );
}
