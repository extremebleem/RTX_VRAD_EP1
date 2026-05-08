#pragma once

#include <vector>

#include "../bsp/source_map.h"
#include "../common/math.h"
#include "../common/types.h"

namespace SilkRAD::V2::Geometry {
    struct FaceGeometry {
        Common::FaceKey key;
        Common::LightmapDimensions lightmap;
        bool valid = false;
        bool isDisplacement = false;
        float worldAreaPerLuxel = 0.0f;
        Common::Vec3f faceNormal;
        float faceDist = 0.0f;
        Common::Vec3f modelOrigin;
        Common::Vec3f luxelOrigin;
        Common::Vec3f worldToLuxelSpace[2];
        Common::Vec3f luxelToWorldSpace[2];
        std::vector<Common::Vec2f> faceWindingLuxel;
        Common::SampleGrid samples;
        Common::LuxelGrid luxels;
    };

    Common::Vec3f face_model_origin(
        const BSP::SourceMap& sourceMap,
        size_t faceIndex
    );

    std::vector<Common::Vec3f> face_world_winding(
        const BSP::SourceMap& sourceMap,
        size_t faceIndex,
        Common::Vec3f modelOrigin
    );

    void world_to_luxel_space(
        const FaceGeometry& geometry,
        Common::Vec3f world,
        Common::Vec2f& coord,
        const ::BSP::DFace& face
    );

    Common::Vec3f luxel_space_to_world(
        const FaceGeometry& geometry,
        float s,
        float t,
        const ::BSP::DFace& face
    );

    FaceGeometry build_face_geometry(
        const BSP::SourceMap& sourceMap,
        size_t faceIndex
    );
}
