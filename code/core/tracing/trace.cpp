#include "trace.h"

namespace SilkRAD::Core::Tracing {
    LineTraceResult test_line(
        const OptixRT::RayHit& hit,
        const std::vector<OptixRT::Triangle>& triangles
    )
    {
        LineTraceResult out;
        if (!hit.hit || hit.primitiveIndex >= triangles.size()) {
            return out;
        }

        const OptixRT::Triangle& triangle = triangles[hit.primitiveIndex];
        out.hit = true;
        out.contents = triangle.contents;
        out.sourceId = triangle.sourceId;
        out.sourceKind = triangle.sourceKind;
        out.surfaceFlags = triangle.surfaceFlags;
        out.role = triangle.role;
        return out;
    }

    SurfaceTraceResult test_line_surface(
        const OptixRT::RayHit& hit,
        const std::vector<OptixRT::Triangle>& triangles
    )
    {
        SurfaceTraceResult out;
        if (!hit.hit || hit.primitiveIndex >= triangles.size()) {
            return out;
        }

        const OptixRT::Triangle& triangle = triangles[hit.primitiveIndex];
        out.hit = true;
        out.hitSky = (triangle.surfaceFlags & ::BSP::SURF_SKY) != 0;
        out.sourceId = triangle.sourceId;
        out.sourceKind = triangle.sourceKind;
        out.surfaceFlags = triangle.surfaceFlags;
        out.role = triangle.role;
        return out;
    }
}
