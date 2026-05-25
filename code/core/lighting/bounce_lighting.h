#pragma once

#include <vector>

#include "../../bsp.h"
#include "../../cudabsp.h"
#include "../../raytracer_optix.h"

namespace SilkRAD::Core {
    struct RuntimeState;
}

namespace SilkRAD::Core::Lighting {
    void compute_bounce_lighting_runtime(
        ::BSP::BSP& bsp,
        ::CUDABSP::CUDABSP* pCudaBSP,
        RuntimeState& state,
        const std::vector<OptixRT::Triangle>& triangles,
        OptixRT::OptixSunLosTracer& tracer
    );
}
