#pragma once

#include <string>
#include <vector>

#include "bsp/source_map.h"
#include "geometry/disp_geometry.h"
#include "geometry/face_geometry.h"
#include "lighting/bounce_lighting.h"
#include "lighting/direct_lighting.h"
#include "scene/scene_builder.h"

namespace SilkRAD::Core {
    struct BuildOptions {
        std::vector<std::string> assetSearchRoots;
    };

    struct RuntimeState {
        std::vector<Geometry::FaceGeometry> faceGeometry;
        std::vector<Geometry::DispGeometry> dispGeometry;
        Scene::SceneBuildResult scene;
        Lighting::DirectLightingResult directLightingInputs;
    };

    RuntimeState build_runtime_state(
        const BSP::SourceMap& sourceMap,
        const BuildOptions& options
    );
}
