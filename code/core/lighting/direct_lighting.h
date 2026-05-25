#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "../../bsp.h"
#include "../../cudabsp.h"
#include "../../raytracer_optix.h"
#include "../geometry/disp_geometry.h"
#include "../geometry/face_geometry.h"
#include "../scene/scene_builder.h"

namespace SilkRAD::Core {
    struct RuntimeState;
}

namespace SilkRAD::Core::Lighting {
    struct DirectLightingResult {
        size_t ordinaryFaceCount = 0;
        size_t displacementFaceCount = 0;
        size_t worldTriangleCount = 0;
        size_t worldBrushTriangleCount = 0;
        size_t displacementTriangleCount = 0;
        size_t staticPropTriangleCount = 0;
        size_t staticPropCount = 0;
        size_t staticPropVertexCount = 0;
    };

    DirectLightingResult build_direct_lighting_inputs(
        const std::vector<Geometry::FaceGeometry>& faceGeometry,
        const std::vector<Geometry::DispGeometry>& dispGeometry,
        const Scene::SceneBuildResult& scene
    );

    void build_runtime_world(
        const RuntimeState& state,
        std::vector<OptixRT::Triangle>& outTriangles
    );

    void compute_direct_lighting_runtime(
        ::BSP::BSP& bsp,
        ::CUDABSP::CUDABSP* pCudaBSP,
        RuntimeState& state,
        const std::vector<OptixRT::Triangle>& triangles,
        OptixRT::OptixSunLosTracer& tracer
    );

    void write_static_prop_direct_lighting(
        ::BSP::BSP& bsp,
        const RuntimeState& state,
        const std::string& outputBspFilename
    );
}
