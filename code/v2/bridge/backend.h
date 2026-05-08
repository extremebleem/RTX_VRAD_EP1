#pragma once

#include <string>
#include <vector>

#include "../bsp/source_map.h"
#include "../geometry/disp_geometry.h"
#include "../geometry/face_geometry.h"
#include "../lighting/direct_lighting.h"
#include "../scene/scene_builder.h"

namespace SilkRAD::V2::Bridge {
    enum class BackendKind {
        Legacy,
        V2,
    };

    struct BuildOptions {
        std::vector<std::string> assetSearchRoots;
    };

    struct BackendState {
        std::vector<Geometry::FaceGeometry> faceGeometry;
        std::vector<Geometry::DispGeometry> dispGeometry;
        Scene::SceneBuildResult scene;
        Lighting::DirectLightingResult directLightingInputs;
    };

    BackendState build_backend_state(
        const BSP::SourceMap& sourceMap,
        const BuildOptions& options
    );
}
