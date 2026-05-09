#pragma once

#include "../../bsp.h"
#include "../../raytracer_optix.h"

namespace SilkRAD::Core::Tracing {
    struct LineTraceResult {
        bool hit = false;
        int32_t contents = 0;
        uint32_t sourceId = 0xffffffffu;
        uint32_t sourceKind = 0;
        int32_t surfaceFlags = 0;
        uint32_t role = 0;
    };

    struct SurfaceTraceResult {
        bool hit = false;
        bool hitSky = false;
        uint32_t sourceId = 0xffffffffu;
        uint32_t sourceKind = 0;
        int32_t surfaceFlags = 0;
        uint32_t role = 0;
    };

    LineTraceResult test_line(
        const OptixRT::RayHit& hit,
        const std::vector<OptixRT::Triangle>& triangles
    );

    SurfaceTraceResult test_line_surface(
        const OptixRT::RayHit& hit,
        const std::vector<OptixRT::Triangle>& triangles
    );
}
