#pragma once

#include <vector>

#include "../bsp/source_map.h"
#include "../common/math.h"
#include "../common/types.h"

namespace SilkRAD::V2::Geometry {
    struct DispSurfaceVertex {
        Common::Vec3f pos;
        Common::Vec3f normal;
    };

    struct DispSampleMapping {
        bool valid = false;
        uint16_t triangleIndex = 0;
        Common::Vec3f barycentric;
    };

    struct DispGeometry {
        Common::DispKey key;
        Common::LightmapDimensions lightmap;
        bool valid = false;
        int power = 0;
        size_t gridSize = 0;
        int32_t lightmapSamplePositionStart = -1;
        Common::Vec3f faceNormal;
        std::vector<DispSurfaceVertex> surfaceVertices;
        std::vector<Common::ReceiverSample> samples;
        std::vector<Common::ReceiverLuxel> luxels;
        std::vector<DispSampleMapping> sampleMappings;
    };

    bool disp_uv_to_surf_point(
        const DispGeometry& geometry,
        float u,
        float v,
        float pushEps,
        Common::Vec3f& outPos
    );

    bool disp_uv_to_surf_normal(
        const DispGeometry& geometry,
        float u,
        float v,
        Common::Vec3f& outNormal
    );

    bool disp_world_to_uv(
        const DispGeometry& geometry,
        Common::Vec3f worldPoint,
        Common::Vec2f& outUv
    );

    DispGeometry build_disp_geometry(
        const BSP::SourceMap& sourceMap,
        size_t faceIndex
    );
}
