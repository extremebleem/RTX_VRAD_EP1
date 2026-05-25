#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace SilkRAD::Core::Common {
    struct Vec2f {
        float x = 0.0f;
        float y = 0.0f;
    };

    struct Vec3f {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    };

    struct FaceKey {
        size_t faceIndex = 0;
    };

    struct DispKey {
        size_t faceIndex = 0;
        size_t dispInfoIndex = 0;
    };

    struct LightmapDimensions {
        size_t width = 0;
        size_t height = 0;

        size_t sample_count() const
        {
            return width * height;
        }
    };

    struct ReceiverSample {
        size_t s = 0;
        size_t t = 0;
        Vec2f coord;
        Vec2f mins;
        Vec2f maxs;
        std::vector<Vec2f> polygon;
        std::vector<Vec3f> worldPolygon;
        Vec3f pos;
        Vec3f normal;
        float area = 0.0f;
    };

    struct ReceiverLuxel {
        size_t s = 0;
        size_t t = 0;
        Vec3f pos;
    };

    struct OccluderTriangle {
        enum class SourceKind : uint32_t {
            Unknown = 0,
            Face = 1,
            Displacement = 2,
            Brush = 3,
            StaticProp = 4,
        };

        Vec3f v0;
        Vec3f v1;
        Vec3f v2;
        uint32_t sourceId = 0;
        uint32_t role = 0;
        SourceKind sourceKind = SourceKind::Unknown;
        int32_t surfaceFlags = 0;
        int32_t contents = 0;
    };

    struct SampleGrid {
        std::vector<ReceiverSample> values;
    };

    struct LuxelGrid {
        std::vector<ReceiverLuxel> values;
    };

    struct StaticPropLightingVertex {
        Vec3f pos;
        Vec3f normal;
    };

    struct StaticPropLightingMesh {
        size_t lod = 0;
        std::vector<StaticPropLightingVertex> vertices;
        std::vector<Vec3f> colors;
    };

    struct StaticPropLightingProp {
        size_t propIndex = 0;
        uint32_t modelChecksum = 0;
        uint32_t flags = 0;
        bool hasLightingOrigin = false;
        Vec3f origin;
        Vec3f lightingOrigin;
        std::vector<StaticPropLightingMesh> meshes;
    };
}
