#pragma once

#include <string>
#include <vector>

#include "../bsp/source_map.h"
#include "../common/types.h"
#include "../geometry/disp_geometry.h"
#include "../geometry/face_geometry.h"

namespace SilkRAD::Core::Scene {
    struct SceneBuildOptions {
        std::vector<std::string> assetSearchRoots;
    };

    struct SceneBuildResult {
        std::vector<Common::OccluderTriangle> worldFaceTriangles;
        std::vector<Common::OccluderTriangle> worldBrushTriangles;
        std::vector<Common::OccluderTriangle> displacementTriangles;
        std::vector<Common::OccluderTriangle> staticPropTriangles;
    };

    SceneBuildResult build_scene(
        const BSP::SourceMap& sourceMap,
        const std::vector<Geometry::FaceGeometry>& faceGeometry,
        const std::vector<Geometry::DispGeometry>& dispGeometry,
        const SceneBuildOptions& options
    );
}
