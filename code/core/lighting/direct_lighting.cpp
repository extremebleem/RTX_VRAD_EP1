#include "direct_lighting.h"
#include "ambient_lighting.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../../cudautils.h"
#include "../../geometry_rules.h"
#include "../state.h"
#include "../common/math.h"
#include "../tracing/trace.h"

namespace SilkRAD::Core::Lighting {
    namespace {
        static constexpr size_t NUM_BUMP_VECTS = 3;
        static constexpr uint32_t RTX_ROLE_SKY = 1u << 2;
        static constexpr uint32_t RTX_ROLE_TRANSLUCENT = 1u << 4;
        static constexpr size_t HOST_SKY_AMBIENT_SAMPLES = 16;
        static constexpr size_t HOST_GI_SAMPLES = 16;
        static constexpr float HOST_COORD_EXTENT = 32768.0f;
        static constexpr float HOST_INDIRECT_SCALE = 0.75f;
        static constexpr float HOST_SKY_GI_SCALE = 0.35f;
        static constexpr float HOST_AMBIENT_FLOOR_SCALE = 0.06f;
        static constexpr float HOST_RAY_BIAS = 0.03125f;
        static constexpr float HOST_RAY_TMIN = 0.001f;
        static constexpr float HOST_MIN_RAY_BIAS = 0.015625f;
        static constexpr float HOST_MAX_RAY_BIAS = 1.0f;
        static constexpr float HOST_DIRECT_FOOTPRINT_MIN_RADIUS = 2.0f;
        static constexpr float HOST_DIRECT_FOOTPRINT_MAX_RADIUS = 8.0f;
        static constexpr size_t HOST_DIRECT_SUBSAMPLES = 16;
        static constexpr size_t HOST_DIRECT_SUN_SHADOW_RAYS = 16;
        static constexpr float HOST_SUN_ANGULAR_RADIUS = 0.02f;
        static constexpr float HOST_LEAF_SURFACE_AMBIENT_SCALE = 0.08f;
        static constexpr float HOST_LEAF_SKY_AMBIENT_SCALE = 0.75f;
        static constexpr float HOST_LEAF_AMBIENT_MAX = 96.0f;
        static constexpr uint32_t LVLFLAGS_BAKED_STATIC_PROP_LIGHTING = 0x1u;
        static constexpr uint32_t VHV_VERSION = 2u;
        static constexpr uint32_t VERTEX_COLOR = 0x0004u;

#pragma pack(push, 1)
        struct VhvMeshHeader {
            uint32_t lod;
            uint32_t vertexCount;
            uint32_t offset;
            uint32_t unused[4];
        };

        struct VhvFileHeader {
            int32_t version;
            uint32_t checksum;
            uint32_t vertexFlags;
            uint32_t vertexSize;
            uint32_t vertexCount;
            int32_t meshCount;
            uint32_t unused[4];
        };
#pragma pack(pop)

        bool is_finite(float value)
        {
            return std::isfinite(value) != 0;
        }

        bool is_finite(float3 v)
        {
            return is_finite(v.x) && is_finite(v.y) && is_finite(v.z);
        }

        float3 to_float3(const Common::Vec3f& v)
        {
            return make_float3(v.x, v.y, v.z);
        }

        float3 safe_normalized(float3 v)
        {
            const float vlen = len(v);
            if (!std::isfinite(vlen) || vlen <= 1e-6f) {
                return make_float3();
            }

            return v / vlen;
        }

        bool is_valid_ray(const OptixRT::SunRay& ray)
        {
            return is_finite(ray.origin)
                && is_finite(ray.direction)
                && is_finite(ray.tmin)
                && is_finite(ray.tmax)
                && ray.tmax > ray.tmin
                && len(ray.direction) > 1e-6f;
        }

        bool cluster_in_pvs(
            const ::BSP::BSP& bsp,
            int16_t sampleCluster,
            int16_t lightCluster
        )
        {
            const ::BSP::BSP::VisMatrix& vis = bsp.get_visibility();

            if (sampleCluster < 0 || lightCluster < 0) {
                return true;
            }

            if (static_cast<size_t>(sampleCluster) >= vis.size()) {
                return true;
            }

            const std::vector<uint8_t>& pvs = vis[sampleCluster];
            const size_t byteIndex = static_cast<size_t>(lightCluster) / 8;
            const size_t bitIndex = static_cast<size_t>(lightCluster) % 8;

            if (byteIndex >= pvs.size()) {
                return true;
            }

            return ((pvs[byteIndex] >> bitIndex) & 0x1) != 0x0;
        }

        float3 find_sky_ambient(const ::BSP::BSP& bsp)
        {
            for (const ::BSP::DWorldLight& light : bsp.get_worldlights()) {
                if (light.type == ::BSP::EMIT_SKYAMBIENT && light.style == 0) {
                    return make_float3(light.intensity) * 255.0f;
                }
            }

            return make_float3(24.0f, 24.0f, 24.0f);
        }

        float3 make_tangent(float3 n)
        {
            const float3 up = std::fabs(n.z) < 0.999f
                ? make_float3(0.0f, 0.0f, 1.0f)
                : make_float3(1.0f, 0.0f, 0.0f);
            return safe_normalized(cross(up, n));
        }

        float3 make_bitangent(float3 n, float3 tangent)
        {
            return safe_normalized(cross(n, tangent));
        }

        float2 luxel_subsample_offset(size_t sampleIndex)
        {
            static const float2 kOffsets[16] = {
                { -0.375f, -0.375f },
                { -0.125f, -0.375f },
                { 0.125f, -0.375f },
                { 0.375f, -0.375f },
                { -0.375f, -0.125f },
                { -0.125f, -0.125f },
                { 0.125f, -0.125f },
                { 0.375f, -0.125f },
                { -0.375f, 0.125f },
                { -0.125f, 0.125f },
                { 0.125f, 0.125f },
                { 0.375f, 0.125f },
                { -0.375f, 0.375f },
                { -0.125f, 0.375f },
                { 0.125f, 0.375f },
                { 0.375f, 0.375f },
            };

            return kOffsets[sampleIndex % 16];
        }

        uint32_t hash_u32(uint32_t x)
        {
            x ^= x >> 16;
            x *= 0x7feb352du;
            x ^= x >> 15;
            x *= 0x846ca68bu;
            x ^= x >> 16;
            return x;
        }

        float unit_float_from_hash(uint32_t x)
        {
            return static_cast<float>(hash_u32(x) & 0x00ffffffu)
                * (1.0f / 16777216.0f);
        }

        float fract_host(float value)
        {
            return value - std::floor(value);
        }

        float radical_inverse_base2(uint32_t bits)
        {
            bits = (bits << 16) | (bits >> 16);
            bits = ((bits & 0x55555555u) << 1) | ((bits & 0xaaaaaaaau) >> 1);
            bits = ((bits & 0x33333333u) << 2) | ((bits & 0xccccccccu) >> 2);
            bits = ((bits & 0x0f0f0f0fu) << 4) | ((bits & 0xf0f0f0f0u) >> 4);
            bits = ((bits & 0x00ff00ffu) << 8) | ((bits & 0xff00ff00u) >> 8);
            return static_cast<float>(bits) * 2.3283064365386963e-10f;
        }

        float2 hammersley_2d(size_t sampleIndex, size_t sampleCount, uint32_t seed)
        {
            const float invCount = sampleCount > 0
                ? (1.0f / static_cast<float>(sampleCount))
                : 1.0f;
            const float jitterX = unit_float_from_hash(seed);
            const float jitterY = unit_float_from_hash(seed ^ 0x9e3779b9u);
            return make_float2(
                fract_host((static_cast<float>(sampleIndex) + 0.5f) * invCount + jitterX),
                fract_host(radical_inverse_base2(static_cast<uint32_t>(sampleIndex)) + jitterY)
            );
        }

        float2 concentric_sample_disk(float2 u)
        {
            const float sx = 2.0f * u.x - 1.0f;
            const float sy = 2.0f * u.y - 1.0f;
            if (sx == 0.0f && sy == 0.0f) {
                return make_float2(0.0f, 0.0f);
            }

            float r = 0.0f;
            float theta = 0.0f;
            if (std::fabs(sx) > std::fabs(sy)) {
                r = sx;
                theta = 0.7853981633974483f * (sy / sx);
            }
            else {
                r = sy;
                theta = 1.5707963267948966f
                    - 0.7853981633974483f * (sx / sy);
            }

            return make_float2(r * std::cos(theta), r * std::sin(theta));
        }

        float direct_ray_bias(float footprintRadius, float3 shadowNormal, float3 rayDir)
        {
            const float baseBias = std::clamp(
                HOST_RAY_BIAS + footprintRadius * 0.015625f,
                HOST_MIN_RAY_BIAS,
                HOST_MAX_RAY_BIAS
            );
            const float normalFacing = std::max(0.25f, std::fabs(dot(shadowNormal, rayDir)));
            return std::clamp(
                baseBias / normalFacing,
                HOST_MIN_RAY_BIAS,
                HOST_MAX_RAY_BIAS
            );
        }

        bool direct_hit_uses_two_sided_shadow(const Tracing::LineTraceResult& hit)
        {
            return hit.sourceKind
                == static_cast<uint32_t>(Common::OccluderTriangle::SourceKind::Face)
                || hit.sourceKind
                    == static_cast<uint32_t>(Common::OccluderTriangle::SourceKind::Displacement)
                || hit.sourceKind
                    == static_cast<uint32_t>(Common::OccluderTriangle::SourceKind::Brush)
                || hit.sourceKind
                    == static_cast<uint32_t>(Common::OccluderTriangle::SourceKind::StaticProp);
        }

        float direct_hit_shadow_visibility(const Tracing::LineTraceResult& hit)
        {
            if (!hit.hit) {
                return 1.0f;
            }

            if (!direct_hit_uses_two_sided_shadow(hit)) {
                return 1.0f;
            }

            if (hit.surfaceFlags & (::BSP::SURF_SKY | ::BSP::SURF_NOSHADOWS)) {
                return 1.0f;
            }

            if (hit.contents & ::BSP::CONTENTS_GRATE) {
                return 0.35f;
            }

            if (hit.role & RTX_ROLE_TRANSLUCENT) {
                return 0.65f;
            }

            if (hit.contents & (::BSP::CONTENTS_WINDOW | ::BSP::CONTENTS_TRANSLUCENT)) {
                return 0.65f;
            }

            if (hit.surfaceFlags & ::BSP::SURF_TRANS) {
                return 0.65f;
            }

            return 0.0f;
        }

        float direct_contents_shadow_visibility(int32_t contents, int32_t surfaceFlags)
        {
            if (surfaceFlags & (::BSP::SURF_SKY | ::BSP::SURF_NOSHADOWS)) {
                return 1.0f;
            }

            if (contents & ::BSP::CONTENTS_GRATE) {
                return 0.35f;
            }

            if (contents & (::BSP::CONTENTS_WINDOW | ::BSP::CONTENTS_TRANSLUCENT)) {
                return 0.65f;
            }

            if (surfaceFlags & ::BSP::SURF_TRANS) {
                return 0.65f;
            }

            return 0.0f;
        }

        int32_t leaf_index_for_pos(const ::BSP::BSP& bsp, float3 pos)
        {
            const std::vector<::BSP::DNode>& nodes = bsp.get_nodes();
            const std::vector<::BSP::DPlane>& planes = bsp.get_planes();

            if (nodes.empty()) {
                return -1;
            }

            int32_t nodeIndex = 0;
            if (!bsp.get_models().empty()) {
                nodeIndex = bsp.get_models()[0].headNode;
            }
            while (nodeIndex >= 0) {
                if (static_cast<size_t>(nodeIndex) >= nodes.size()) {
                    return -1;
                }

                const ::BSP::DNode& node = nodes[static_cast<size_t>(nodeIndex)];
                if (node.planeNum < 0
                    || static_cast<size_t>(node.planeNum) >= planes.size()) {
                    return -1;
                }

                const ::BSP::DPlane& plane = planes[static_cast<size_t>(node.planeNum)];
                const float dist =
                    pos.x * plane.normal.x
                    + pos.y * plane.normal.y
                    + pos.z * plane.normal.z
                    - plane.dist;
                nodeIndex = node.children[dist >= 0.0f ? 0 : 1];
            }

            return -nodeIndex - 1;
        }

        bool brush_contains_point(
            const ::BSP::BSP& bsp,
            const ::BSP::DBrush& brush,
            float3 pos,
            float epsilon
        )
        {
            const std::vector<::BSP::DBrushSide>& sides = bsp.get_brushsides();
            const std::vector<::BSP::DPlane>& planes = bsp.get_planes();

            if (brush.firstSide < 0 || brush.numSides <= 0) {
                return false;
            }

            const size_t firstSide = static_cast<size_t>(brush.firstSide);
            const size_t numSides = static_cast<size_t>(brush.numSides);
            if (firstSide + numSides > sides.size()) {
                return false;
            }

            for (size_t localSide = 0; localSide < numSides; ++localSide) {
                const ::BSP::DBrushSide& side = sides[firstSide + localSide];
                if (side.bevel) {
                    continue;
                }
                if (side.planeNum >= planes.size()) {
                    return false;
                }

                const ::BSP::DPlane& plane = planes[side.planeNum];
                const float dist =
                    pos.x * plane.normal.x
                    + pos.y * plane.normal.y
                    + pos.z * plane.normal.z
                    - plane.dist;
                if (dist > epsilon) {
                    return false;
                }
            }

            return true;
        }

        bool direct_brush_blocks_light(
            const ::BSP::BSP& bsp,
            const ::BSP::DBrush& brush,
            int32_t& outSurfaceFlags
        )
        {
            outSurfaceFlags = 0;

            if (GeometryRules::brush_contents_block_light(brush.contents)) {
                return true;
            }

            const std::vector<::BSP::DBrushSide>& sides = bsp.get_brushsides();
            if (brush.firstSide < 0 || brush.numSides <= 0) {
                return false;
            }

            const size_t firstSide = static_cast<size_t>(brush.firstSide);
            const size_t numSides = static_cast<size_t>(brush.numSides);
            if (firstSide + numSides > sides.size()) {
                return false;
            }

            for (size_t localSide = 0; localSide < numSides; ++localSide) {
                const ::BSP::DBrushSide& side = sides[firstSide + localSide];
                if (side.texInfo < 0
                    || static_cast<size_t>(side.texInfo) >= bsp.get_texinfos().size()) {
                    continue;
                }

                const ::BSP::TexInfo& texInfo = bsp.get_texinfos()[side.texInfo];
                const std::string materialName = bsp.get_texture_name(texInfo.texData);
                if (GeometryRules::world_brush_side_blocks_light(
                        brush.contents,
                        texInfo.flags,
                        materialName)) {
                    outSurfaceFlags = texInfo.flags;
                    return true;
                }
            }

            return false;
        }

        float direct_origin_shadow_visibility(const ::BSP::BSP& bsp, float3 origin)
        {
            const int32_t leafIndex = leaf_index_for_pos(bsp, origin);
            if (leafIndex < 0
                || static_cast<size_t>(leafIndex) >= bsp.get_leaves().size()) {
                return 1.0f;
            }

            const ::BSP::DLeaf& leaf = bsp.get_leaves()[static_cast<size_t>(leafIndex)];
            if (leaf.contents & ::BSP::CONTENTS_SOLID) {
                return 0.0f;
            }

            const std::vector<uint16_t>& leafBrushes = bsp.get_leafbrushes();
            const std::vector<::BSP::DBrush>& brushes = bsp.get_brushes();
            const size_t firstLeafBrush = static_cast<size_t>(leaf.firstLeafBrush);
            const size_t numLeafBrushes = static_cast<size_t>(leaf.numLeafBrushes);
            if (firstLeafBrush + numLeafBrushes > leafBrushes.size()) {
                return 1.0f;
            }

            float visibility = 1.0f;
            const float insideEpsilon = std::max(HOST_RAY_BIAS, HOST_MIN_RAY_BIAS);
            for (size_t i = 0; i < numLeafBrushes; ++i) {
                const size_t brushIndex = static_cast<size_t>(leafBrushes[firstLeafBrush + i]);
                if (brushIndex >= brushes.size()) {
                    continue;
                }

                const ::BSP::DBrush& brush = brushes[brushIndex];
                int32_t surfaceFlags = 0;
                if (!direct_brush_blocks_light(bsp, brush, surfaceFlags)) {
                    continue;
                }

                if (!brush_contains_point(bsp, brush, origin, insideEpsilon)) {
                    continue;
                }

                visibility = std::min(
                    visibility,
                    direct_contents_shadow_visibility(brush.contents, surfaceFlags)
                );
            }

            return visibility;
        }

        bool point_in_luxel_polygon(
            const std::vector<Common::Vec2f>& polygon,
            Common::Vec2f point
        )
        {
            if (polygon.size() < 3) {
                return true;
            }

            bool hasPositive = false;
            bool hasNegative = false;
            for (size_t i = 0; i < polygon.size(); ++i) {
                const Common::Vec2f a = polygon[i];
                const Common::Vec2f b = polygon[(i + 1) % polygon.size()];
                const float cross =
                    (b.x - a.x) * (point.y - a.y)
                    - (b.y - a.y) * (point.x - a.x);
                if (cross > 1e-5f) {
                    hasPositive = true;
                }
                else if (cross < -1e-5f) {
                    hasNegative = true;
                }
                if (hasPositive && hasNegative) {
                    return false;
                }
            }

            return true;
        }

        Common::Vec2f clamp_luxel_subsample_to_polygon(
            const Common::ReceiverSample& sample,
            Common::Vec2f coord
        )
        {
            if (sample.polygon.size() < 3
                || point_in_luxel_polygon(sample.polygon, coord)) {
                return coord;
            }

            Common::Vec2f inside = sample.coord;
            if (!point_in_luxel_polygon(sample.polygon, inside)) {
                return inside;
            }

            Common::Vec2f outside = coord;
            for (size_t i = 0; i < 8; ++i) {
                const Common::Vec2f mid = Common::make_vec2(
                    (inside.x + outside.x) * 0.5f,
                    (inside.y + outside.y) * 0.5f
                );
                if (point_in_luxel_polygon(sample.polygon, mid)) {
                    inside = mid;
                }
                else {
                    outside = mid;
                }
            }

            return inside;
        }

        float3 ambient_direction(float3 n, size_t sampleIndex)
        {
            static const float2 kDisk[HOST_SKY_AMBIENT_SAMPLES] = {
                { 0.0000f, 0.0000f },
                { 0.2500f, 0.0000f },
                { -0.2500f, 0.0000f },
                { 0.0000f, 0.2500f },
                { 0.0000f, -0.2500f },
                { 0.3536f, 0.3536f },
                { -0.3536f, 0.3536f },
                { 0.3536f, -0.3536f },
                { -0.3536f, -0.3536f },
                { 0.6500f, 0.0000f },
                { -0.6500f, 0.0000f },
                { 0.0000f, 0.6500f },
                { 0.0000f, -0.6500f },
                { 0.4596f, 0.4596f },
                { -0.4596f, 0.4596f },
                { 0.4596f, -0.4596f },
            };

            const float3 tangent = make_tangent(n);
            const float3 bitangent = make_bitangent(n, tangent);
            const float x = kDisk[sampleIndex].x;
            const float y = kDisk[sampleIndex].y;
            const float z = std::sqrt(std::max(0.0f, 1.0f - x * x - y * y));
            return safe_normalized(tangent * x + bitangent * y + n * z);
        }

        float3 gi_direction(float3 n, size_t sampleIndex)
        {
            return ambient_direction(n, sampleIndex % HOST_SKY_AMBIENT_SAMPLES);
        }

        uint32_t hit_role(
            const std::vector<OptixRT::Triangle>& triangles,
            const OptixRT::RayHit& hit
        )
        {
            if (!hit.hit || hit.primitiveIndex >= triangles.size()) {
                return 0;
            }

            return triangles[hit.primitiveIndex].role;
        }

        float3 surface_reflectivity(
            const ::BSP::BSP& bsp,
            size_t faceIndex
        )
        {
            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.texInfo < 0
                || static_cast<size_t>(face.texInfo) >= bsp.get_texinfos().size()) {
                return make_float3(0.5f, 0.5f, 0.5f);
            }

            const ::BSP::TexInfo& texInfo = bsp.get_texinfos()[face.texInfo];
            if (texInfo.texData < 0
                || static_cast<size_t>(texInfo.texData) >= bsp.get_texdatas().size()) {
                return make_float3(0.5f, 0.5f, 0.5f);
            }

            return make_float3(bsp.get_texdatas()[texInfo.texData].reflectivity);
        }

        bool surf_has_bumped_lightmaps(const ::BSP::BSP& bsp, size_t faceIndex)
        {
            if (faceIndex >= bsp.get_faces().size()) {
                return false;
            }

            const int32_t flags = bsp.get_faces()[faceIndex].get_texinfo().flags;
            return ((flags & ::BSP::SURF_BUMPLIGHT) != 0)
                && ((flags & ::BSP::SURF_NOLIGHT) == 0);
        }

        size_t sample_plane_count(const ::BSP::BSP& bsp, size_t faceIndex)
        {
            return surf_has_bumped_lightmaps(bsp, faceIndex)
                ? (NUM_BUMP_VECTS + 1)
                : 1;
        }

        void get_bump_normals(
            const float* sVect,
            const float* tVect,
            float3 flatNormal,
            float3 phongNormal,
            float3 bumpNormals[NUM_BUMP_VECTS]
        )
        {
            static constexpr float kLocalBumpBasis[NUM_BUMP_VECTS][3] = {
                { 0.81649661064147949f, 0.0f, 0.57735025882720947f },
                { -0.40824821591377258f, 0.70710676908493042f, 0.57735025882720947f },
                { -0.40824821591377258f, -0.70710676908493042f, 0.57735025882720947f },
            };

            const float3 s = make_float3(sVect[0], sVect[1], sVect[2]);
            const float3 t = make_float3(tVect[0], tVect[1], tVect[2]);
            const bool leftHanded = dot(flatNormal, cross(s, t)) < 0.0f;

            float3 smoothBasis1 = safe_normalized(cross(phongNormal, s));
            float3 smoothBasis0 = safe_normalized(cross(smoothBasis1, phongNormal));
            float3 smoothBasis2 = safe_normalized(phongNormal);

            if (leftHanded) {
                smoothBasis1 *= -1.0f;
            }

            for (size_t i = 0; i < NUM_BUMP_VECTS; ++i) {
                const float3 local = make_float3(
                    kLocalBumpBasis[i][0],
                    kLocalBumpBasis[i][1],
                    kLocalBumpBasis[i][2]
                );
                bumpNormals[i] = safe_normalized(
                    smoothBasis0 * local.x
                    + smoothBasis1 * local.y
                    + smoothBasis2 * local.z
                );
            }
        }

        bool point_in_winding_world(
            const std::vector<Common::Vec3f>& polygon,
            float3 point,
            float3 normal
        )
        {
            if (polygon.size() < 3) {
                return true;
            }

            float sign = 0.0f;
            for (size_t i = 0; i < polygon.size(); ++i) {
                const Common::Vec3f& a = polygon[i];
                const Common::Vec3f& b = polygon[(i + 1) % polygon.size()];
                const float3 av = make_float3(a.x, a.y, a.z);
                const float3 bv = make_float3(b.x, b.y, b.z);
                const float3 edge = bv - av;
                const float3 toPoint = point - av;
                const float edgeSign = dot(cross(edge, toPoint), normal);

                if (std::fabs(edgeSign) <= 1e-5f) {
                    continue;
                }

                if (sign == 0.0f) {
                    sign = edgeSign;
                }
                else if ((sign > 0.0f && edgeSign < 0.0f) || (sign < 0.0f && edgeSign > 0.0f)) {
                    return false;
                }
            }

            return true;
        }

        bool leaf_receives_ambient(const ::BSP::DLeaf& leaf)
        {
            return (leaf.contents & ::BSP::CONTENTS_SOLID) == 0;
        }

        ::BSP::RGBExp32 rgbexp32_from_float3(float3 color)
        {
            color.x = std::max(color.x, 0.0f);
            color.y = std::max(color.y, 0.0f);
            color.z = std::max(color.z, 0.0f);

            float maxColor = std::max(color.x, std::max(color.y, color.z));
            int exponent = 0;

            if (maxColor > 0.0f) {
                float normalizedColor = maxColor;
                while (normalizedColor > 255.0f && exponent < 127) {
                    ++exponent;
                    normalizedColor *= 0.5f;
                }
                while (normalizedColor < 127.0f && exponent > -128) {
                    --exponent;
                    normalizedColor *= 2.0f;
                }
            }

            const float scalar = std::ldexp(1.0f, -exponent);
            return ::BSP::RGBExp32{
                static_cast<uint8_t>(std::min(color.x * scalar, 255.0f)),
                static_cast<uint8_t>(std::min(color.y * scalar, 255.0f)),
                static_cast<uint8_t>(std::min(color.z * scalar, 255.0f)),
                static_cast<int8_t>(exponent)
            };
        }

        float3 tonemap_leaf_ambient(float3 color)
        {
            color.x = std::max(color.x, 0.0f);
            color.y = std::max(color.y, 0.0f);
            color.z = std::max(color.z, 0.0f);

            const float maxColor = std::max(color.x, std::max(color.y, color.z));
            if (maxColor > HOST_LEAF_AMBIENT_MAX && maxColor > 1e-6f) {
                color = color * (HOST_LEAF_AMBIENT_MAX / maxColor);
            }

            return color;
        }

        size_t align_up(size_t value, size_t alignment)
        {
            return (value + alignment - 1) & ~(alignment - 1);
        }

        uint8_t clamp_u8(float value)
        {
            return static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
        }

        std::array<uint8_t, 4> vhv_color_from_float3(float3 color)
        {
            const ::BSP::RGBExp32 rgbexp = rgbexp32_from_float3(color);
            const float r = std::max(
                0.0f,
                static_cast<float>(rgbexp.r) * std::ldexp(1.0f, rgbexp.exp)
            );
            const float g = std::max(
                0.0f,
                static_cast<float>(rgbexp.g) * std::ldexp(1.0f, rgbexp.exp)
            );
            const float b = std::max(
                0.0f,
                static_cast<float>(rgbexp.b) * std::ldexp(1.0f, rgbexp.exp)
            );
            const float maxChannel = std::max(r, std::max(g, b));
            const float scale = maxChannel > 255.0f ? (255.0f / maxChannel) : 1.0f;

            return {
                clamp_u8(b * scale),
                clamp_u8(g * scale),
                clamp_u8(r * scale),
                255u
            };
        }

        uint32_t crc32_bytes(const uint8_t* data, size_t size)
        {
            static uint32_t table[256] = {};
            static bool tableReady = false;

            if (!tableReady) {
                for (uint32_t i = 0; i < 256; ++i) {
                    uint32_t c = i;
                    for (int bit = 0; bit < 8; ++bit) {
                        c = (c & 1u) != 0u ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
                    }
                    table[i] = c;
                }
                tableReady = true;
            }

            uint32_t crc = 0xffffffffu;
            for (size_t i = 0; i < size; ++i) {
                crc = table[(crc ^ data[i]) & 0xffu] ^ (crc >> 8);
            }
            return crc ^ 0xffffffffu;
        }

        uint16_t read_le16(const std::vector<uint8_t>& data, size_t offset)
        {
            if (offset + 2 > data.size()) {
                return 0;
            }

            return static_cast<uint16_t>(data[offset])
                | (static_cast<uint16_t>(data[offset + 1]) << 8);
        }

        uint32_t read_le32(const std::vector<uint8_t>& data, size_t offset)
        {
            if (offset + 4 > data.size()) {
                return 0;
            }

            return static_cast<uint32_t>(data[offset])
                | (static_cast<uint32_t>(data[offset + 1]) << 8)
                | (static_cast<uint32_t>(data[offset + 2]) << 16)
                | (static_cast<uint32_t>(data[offset + 3]) << 24);
        }

        void append_le16(std::vector<uint8_t>& data, uint16_t value)
        {
            data.push_back(static_cast<uint8_t>(value & 0xffu));
            data.push_back(static_cast<uint8_t>((value >> 8) & 0xffu));
        }

        void append_le32(std::vector<uint8_t>& data, uint32_t value)
        {
            data.push_back(static_cast<uint8_t>(value & 0xffu));
            data.push_back(static_cast<uint8_t>((value >> 8) & 0xffu));
            data.push_back(static_cast<uint8_t>((value >> 16) & 0xffu));
            data.push_back(static_cast<uint8_t>((value >> 24) & 0xffu));
        }

        struct ZipEntryMetadata {
            std::string name;
            uint16_t method = 0;
            uint32_t crc32 = 0;
            uint32_t compressedSize = 0;
            uint32_t uncompressedSize = 0;
            uint32_t localHeaderOffset = 0;
        };

        struct ZipParseResult {
            bool valid = false;
            size_t prefixSize = 0;
            std::vector<ZipEntryMetadata> entries;
        };

        bool is_static_prop_vhv_name(const std::string& name)
        {
            return name.size() >= 8
                && name.rfind("sp_", 0) == 0
                && name.substr(name.size() - 4) == ".vhv";
        }

        ZipParseResult parse_zip_metadata(const std::vector<uint8_t>& bytes)
        {
            static constexpr uint32_t EOCD_SIGNATURE = 0x06054b50u;
            static constexpr uint32_t CENTRAL_SIGNATURE = 0x02014b50u;
            ZipParseResult result;

            if (bytes.size() < 22) {
                result.valid = bytes.empty();
                result.prefixSize = 0;
                return result;
            }

            size_t eocdOffset = std::string::npos;
            const size_t searchStart = bytes.size() > 0xffff + 22
                ? bytes.size() - (0xffff + 22)
                : 0;
            for (size_t pos = bytes.size() - 22;; --pos) {
                if (read_le32(bytes, pos) == EOCD_SIGNATURE) {
                    eocdOffset = pos;
                    break;
                }
                if (pos == searchStart) {
                    break;
                }
            }

            if (eocdOffset == std::string::npos) {
                return result;
            }

            const uint16_t totalEntries = read_le16(bytes, eocdOffset + 10);
            const uint32_t centralSize = read_le32(bytes, eocdOffset + 12);
            const uint32_t centralOffset = read_le32(bytes, eocdOffset + 16);
            if (centralOffset + centralSize > bytes.size()) {
                return result;
            }

            size_t cursor = centralOffset;
            result.entries.reserve(totalEntries);
            for (uint16_t i = 0; i < totalEntries; ++i) {
                if (cursor + 46 > bytes.size() || read_le32(bytes, cursor) != CENTRAL_SIGNATURE) {
                    result.entries.clear();
                    return result;
                }

                const uint16_t nameLen = read_le16(bytes, cursor + 28);
                const uint16_t extraLen = read_le16(bytes, cursor + 30);
                const uint16_t commentLen = read_le16(bytes, cursor + 32);
                const size_t headerSize = 46u + nameLen + extraLen + commentLen;
                if (cursor + headerSize > bytes.size()) {
                    result.entries.clear();
                    return result;
                }

                ZipEntryMetadata entry;
                entry.method = read_le16(bytes, cursor + 10);
                entry.crc32 = read_le32(bytes, cursor + 16);
                entry.compressedSize = read_le32(bytes, cursor + 20);
                entry.uncompressedSize = read_le32(bytes, cursor + 24);
                entry.localHeaderOffset = read_le32(bytes, cursor + 42);
                entry.name.assign(
                    reinterpret_cast<const char*>(bytes.data() + cursor + 46),
                    static_cast<size_t>(nameLen)
                );
                result.entries.push_back(std::move(entry));
                cursor += headerSize;
            }

            result.valid = true;
            result.prefixSize = centralOffset;
            return result;
        }

        void append_zip_local_file(
            std::vector<uint8_t>& zipBytes,
            const std::string& name,
            const std::vector<uint8_t>& fileBytes,
            ZipEntryMetadata& outEntry
        )
        {
            static constexpr uint32_t LOCAL_SIGNATURE = 0x04034b50u;
            outEntry.name = name;
            outEntry.method = 0;
            outEntry.crc32 = crc32_bytes(fileBytes.data(), fileBytes.size());
            outEntry.compressedSize = static_cast<uint32_t>(fileBytes.size());
            outEntry.uncompressedSize = static_cast<uint32_t>(fileBytes.size());
            outEntry.localHeaderOffset = static_cast<uint32_t>(zipBytes.size());

            append_le32(zipBytes, LOCAL_SIGNATURE);
            append_le16(zipBytes, 20);
            append_le16(zipBytes, 0);
            append_le16(zipBytes, 0);
            append_le16(zipBytes, 0);
            append_le16(zipBytes, 0);
            append_le32(zipBytes, outEntry.crc32);
            append_le32(zipBytes, outEntry.compressedSize);
            append_le32(zipBytes, outEntry.uncompressedSize);
            append_le16(zipBytes, static_cast<uint16_t>(name.size()));
            append_le16(zipBytes, 0);
            zipBytes.insert(zipBytes.end(), name.begin(), name.end());
            zipBytes.insert(zipBytes.end(), fileBytes.begin(), fileBytes.end());
        }

        std::vector<uint8_t> build_zip_with_replaced_entries(
            const std::vector<uint8_t>& existingPak,
            const std::map<std::string, std::vector<uint8_t>>& replacementEntries
        )
        {
            static constexpr uint32_t CENTRAL_SIGNATURE = 0x02014b50u;
            static constexpr uint32_t EOCD_SIGNATURE = 0x06054b50u;

            const ZipParseResult parsed = parse_zip_metadata(existingPak);
            std::vector<uint8_t> zipBytes;
            std::vector<ZipEntryMetadata> centralEntries;
            std::unordered_map<std::string, bool> usedReplacementNames;

            if (parsed.valid) {
                zipBytes.insert(
                    zipBytes.end(),
                    existingPak.begin(),
                    existingPak.begin() + static_cast<std::ptrdiff_t>(parsed.prefixSize)
                );
                centralEntries.reserve(parsed.entries.size() + replacementEntries.size());

                for (const ZipEntryMetadata& entry : parsed.entries) {
                    if (replacementEntries.find(entry.name) != replacementEntries.end()) {
                        usedReplacementNames[entry.name] = true;
                        continue;
                    }
                    if (is_static_prop_vhv_name(entry.name)) {
                        continue;
                    }
                    centralEntries.push_back(entry);
                }
            }

            for (const auto& pair : replacementEntries) {
                ZipEntryMetadata entry;
                append_zip_local_file(zipBytes, pair.first, pair.second, entry);
                centralEntries.push_back(std::move(entry));
                usedReplacementNames[pair.first] = true;
            }

            const uint32_t centralOffset = static_cast<uint32_t>(zipBytes.size());
            for (const ZipEntryMetadata& entry : centralEntries) {
                append_le32(zipBytes, CENTRAL_SIGNATURE);
                append_le16(zipBytes, 20);
                append_le16(zipBytes, 20);
                append_le16(zipBytes, 0);
                append_le16(zipBytes, entry.method);
                append_le16(zipBytes, 0);
                append_le16(zipBytes, 0);
                append_le32(zipBytes, entry.crc32);
                append_le32(zipBytes, entry.compressedSize);
                append_le32(zipBytes, entry.uncompressedSize);
                append_le16(zipBytes, static_cast<uint16_t>(entry.name.size()));
                append_le16(zipBytes, 0);
                append_le16(zipBytes, 0);
                append_le16(zipBytes, 0);
                append_le16(zipBytes, 0);
                append_le32(zipBytes, 0);
                append_le32(zipBytes, entry.localHeaderOffset);
                zipBytes.insert(zipBytes.end(), entry.name.begin(), entry.name.end());
            }

            const uint32_t centralSize = static_cast<uint32_t>(zipBytes.size()) - centralOffset;
            append_le32(zipBytes, EOCD_SIGNATURE);
            append_le16(zipBytes, 0);
            append_le16(zipBytes, 0);
            append_le16(zipBytes, static_cast<uint16_t>(centralEntries.size()));
            append_le16(zipBytes, static_cast<uint16_t>(centralEntries.size()));
            append_le32(zipBytes, centralSize);
            append_le32(zipBytes, centralOffset);
            append_le16(zipBytes, 0);
            return zipBytes;
        }

        std::vector<uint8_t> build_vhv_bytes(const Common::StaticPropLightingProp& prop)
        {
            size_t totalVertices = 0;
            for (const Common::StaticPropLightingMesh& mesh : prop.meshes) {
                totalVertices += mesh.colors.size();
            }
            if (totalVertices == 0 || prop.meshes.empty()) {
                return {};
            }

            const size_t headerBytes =
                sizeof(VhvFileHeader)
                + prop.meshes.size() * sizeof(VhvMeshHeader);
            const size_t vertexDataOffset = align_up(headerBytes, 512);
            const size_t fileSize = align_up(vertexDataOffset + totalVertices * 4, 512);
            std::vector<uint8_t> bytes(fileSize, 0);

            VhvFileHeader header{};
            header.version = static_cast<int32_t>(VHV_VERSION);
            header.checksum = prop.modelChecksum;
            header.vertexFlags = VERTEX_COLOR;
            header.vertexSize = 4;
            header.vertexCount = static_cast<uint32_t>(totalVertices);
            header.meshCount = static_cast<int32_t>(prop.meshes.size());
            std::memcpy(bytes.data(), &header, sizeof(header));

            size_t currentVertexOffset = vertexDataOffset;
            VhvMeshHeader* meshHeaders = reinterpret_cast<VhvMeshHeader*>(
                bytes.data() + sizeof(VhvFileHeader)
            );

            for (size_t meshIndex = 0; meshIndex < prop.meshes.size(); ++meshIndex) {
                const Common::StaticPropLightingMesh& mesh = prop.meshes[meshIndex];
                VhvMeshHeader meshHeader{};
                meshHeader.lod = static_cast<uint32_t>(mesh.lod);
                meshHeader.vertexCount = static_cast<uint32_t>(mesh.colors.size());
                meshHeader.offset = static_cast<uint32_t>(currentVertexOffset);
                meshHeaders[meshIndex] = meshHeader;

                for (const Common::Vec3f& color : mesh.colors) {
                    const std::array<uint8_t, 4> packed =
                        vhv_color_from_float3(make_float3(color.x, color.y, color.z));
                    if (currentVertexOffset + packed.size() > bytes.size()) {
                        break;
                    }

                    std::memcpy(
                        bytes.data() + currentVertexOffset,
                        packed.data(),
                        packed.size()
                    );
                    currentVertexOffset += packed.size();
                }
            }

            return bytes;
        }

        bool core_receiver_sample(
            const RuntimeState& state,
            size_t faceIndex,
            size_t s,
            size_t t,
            float2 sampleOffset,
            float3& outPos,
            float3& outShadingNormal,
            float3& outShadowNormal,
            float& outDirectFootprintRadius
        )
        {
            if (faceIndex >= state.faceGeometry.size()) {
                return false;
            }

            const Geometry::DispGeometry& dispGeometry = state.dispGeometry[faceIndex];
            if (dispGeometry.valid) {
                const size_t sampleIndex = t * dispGeometry.lightmap.width + s;
                if (sampleIndex >= dispGeometry.samples.size()) {
                    return false;
                }

                const Common::ReceiverSample& sample = dispGeometry.samples[sampleIndex];
                Common::Vec3f pos;
                Common::Vec3f normal;
                if (!Geometry::disp_uv_to_surf_point(
                    dispGeometry,
                    sample.coord.x,
                    sample.coord.y,
                    1.0f,
                    pos
                )) {
                    return false;
                }
                if (!Geometry::disp_uv_to_surf_normal(
                    dispGeometry,
                    sample.coord.x,
                    sample.coord.y,
                    normal
                )) {
                    normal = sample.normal;
                }

                outShadingNormal = safe_normalized(to_float3(normal));
                outShadowNormal = outShadingNormal;
                outPos = to_float3(pos);
                outDirectFootprintRadius = HOST_DIRECT_FOOTPRINT_MIN_RADIUS;
                return is_finite(outPos)
                    && is_finite(outShadingNormal)
                    && is_finite(outShadowNormal);
            }

            const Geometry::FaceGeometry& faceGeometry = state.faceGeometry[faceIndex];
            if (!faceGeometry.valid || faceGeometry.isDisplacement) {
                return false;
            }

            const size_t sampleIndex = t * faceGeometry.lightmap.width + s;
            if (sampleIndex >= faceGeometry.samples.values.size()) {
                return false;
            }

            const Common::ReceiverSample& sample = faceGeometry.samples.values[sampleIndex];
            const Common::Vec2f sampleCoord = clamp_luxel_subsample_to_polygon(
                sample,
                Common::make_vec2(
                    sample.coord.x + sampleOffset.x,
                    sample.coord.y + sampleOffset.y
                )
            );
            const float ss = sampleCoord.x;
            const float tt = sampleCoord.y;

            outShadingNormal = safe_normalized(to_float3(sample.normal));
            outShadowNormal = safe_normalized(to_float3(faceGeometry.faceNormal));
            outDirectFootprintRadius = std::clamp(
                std::max(
                    len(to_float3(faceGeometry.luxelToWorldSpace[0])),
                    len(to_float3(faceGeometry.luxelToWorldSpace[1]))
                ) * 0.5f,
                HOST_DIRECT_FOOTPRINT_MIN_RADIUS,
                HOST_DIRECT_FOOTPRINT_MAX_RADIUS
            );
            const float ds = ss - sample.coord.x;
            const float dt = tt - sample.coord.y;
            const Common::Vec3f shiftedPos = Common::add(
                sample.pos,
                Common::add(
                    Common::scale(faceGeometry.luxelToWorldSpace[0], ds),
                    Common::scale(faceGeometry.luxelToWorldSpace[1], dt)
                )
            );
            outPos = to_float3(shiftedPos);
            return is_finite(outPos)
                && is_finite(outShadingNormal)
                && is_finite(outShadowNormal);
        }
    }

    DirectLightingResult build_direct_lighting_inputs(
        const std::vector<Geometry::FaceGeometry>& faceGeometry,
        const std::vector<Geometry::DispGeometry>& dispGeometry,
        const Scene::SceneBuildResult& scene
    )
    {
        DirectLightingResult result;

        for (const Geometry::FaceGeometry& geometry : faceGeometry) {
            if (!geometry.valid || geometry.isDisplacement) {
                continue;
            }
            ++result.ordinaryFaceCount;
        }

        for (const Geometry::DispGeometry& geometry : dispGeometry) {
            if (!geometry.valid) {
                continue;
            }
            ++result.displacementFaceCount;
        }

        result.worldTriangleCount = scene.worldFaceTriangles.size();
        result.worldBrushTriangleCount = scene.worldBrushTriangles.size();
        result.displacementTriangleCount = scene.displacementTriangles.size();
        result.staticPropTriangleCount = scene.staticPropTriangles.size();
        result.staticPropCount = scene.staticPropLightingProps.size();
        for (const Common::StaticPropLightingProp& prop : scene.staticPropLightingProps) {
            for (const Common::StaticPropLightingMesh& mesh : prop.meshes) {
                result.staticPropVertexCount += mesh.vertices.size();
            }
        }
        return result;
    }

    void build_runtime_world(
        const RuntimeState& state,
        std::vector<OptixRT::Triangle>& outTriangles
    )
    {
        outTriangles.clear();

        const Scene::SceneBuildResult& scene = state.scene;
        const size_t totalTriangles =
            scene.worldFaceTriangles.size()
            + scene.worldBrushTriangles.size()
            + scene.displacementTriangles.size()
            + scene.staticPropTriangles.size();
        outTriangles.reserve(totalTriangles);

        auto appendTriangles = [&outTriangles](const std::vector<Common::OccluderTriangle>& input) {
            for (const Common::OccluderTriangle& triIn : input) {
                OptixRT::Triangle tri{};
                tri.v0 = to_float3(triIn.v0);
                tri.v1 = to_float3(triIn.v1);
                tri.v2 = to_float3(triIn.v2);
                tri.sourceId = triIn.sourceId;
                tri.role = triIn.role;
                tri.sourceKind = static_cast<uint32_t>(triIn.sourceKind);
                tri.surfaceFlags = triIn.surfaceFlags;
                tri.contents = triIn.contents;
                tri.visibilityMask = 0xff;
                outTriangles.push_back(tri);
            }
        };

        appendTriangles(scene.worldFaceTriangles);
        appendTriangles(scene.worldBrushTriangles);
        appendTriangles(scene.displacementTriangles);
        appendTriangles(scene.staticPropTriangles);
    }

    void compute_direct_lighting_runtime(
        ::BSP::BSP& bsp,
        ::CUDABSP::CUDABSP* pCudaBSP,
        RuntimeState& state,
        const std::vector<OptixRT::Triangle>& triangles,
        OptixRT::OptixSunLosTracer& tracer
    )
    {
        using Clock = std::chrono::high_resolution_clock;

        enum class DirectVisibilityMode : uint8_t {
            PointOrSpot,
            DirectionalSun,
        };

        enum class DirectOutputKind : uint8_t {
            Lightmap,
            StaticPropVertex,
        };

        struct RayContribution {
            DirectOutputKind outputKind;
            size_t outputIndex;
            size_t outputStride;
            float3 color[NUM_BUMP_VECTS + 1];
            size_t planeCount;
            DirectVisibilityMode visibilityMode;
            uint32_t ignoreSourceId;
            uint32_t receiverFaceIndex;
            size_t firstRayIndex;
            size_t rayCount;
        };

        struct SampleNormalization {
            DirectOutputKind outputKind;
            size_t outputIndex;
            size_t outputStride;
            size_t planeCount;
            size_t validSubsamples;
        };

        auto clampf_host = [](float v, float lo, float hi) {
            return std::max(lo, std::min(v, hi));
        };

        auto attenuate_host = [](const ::BSP::DWorldLight& light, float dist) {
            return light.constantAtten
                + light.linearAtten * dist
                + light.quadraticAtten * dist * dist;
        };

        auto surface_faces_coplanar =
            [&](uint32_t receiverFaceIndex, uint32_t hitFaceIndex) -> bool {
            if (receiverFaceIndex == 0xffffffffu
                || hitFaceIndex == 0xffffffffu
                || receiverFaceIndex >= bsp.get_dfaces().size()
                || hitFaceIndex >= bsp.get_dfaces().size()) {
                return false;
            }

            const ::BSP::DFace& receiverFace = bsp.get_dfaces()[receiverFaceIndex];
            const ::BSP::DFace& hitFace = bsp.get_dfaces()[hitFaceIndex];
            if (receiverFace.planeNum == hitFace.planeNum
                && receiverFace.side == hitFace.side) {
                return true;
            }

            if (receiverFaceIndex >= state.faceGeometry.size()
                || hitFaceIndex >= state.faceGeometry.size()) {
                return false;
            }

            const Geometry::FaceGeometry& receiverGeometry = state.faceGeometry[receiverFaceIndex];
            const Geometry::FaceGeometry& hitGeometry = state.faceGeometry[hitFaceIndex];
            if (!receiverGeometry.valid || !hitGeometry.valid) {
                return false;
            }

            const float3 receiverNormal = safe_normalized(to_float3(receiverGeometry.faceNormal));
            const float3 hitNormal = safe_normalized(to_float3(hitGeometry.faceNormal));
            const float normalDot = dot(receiverNormal, hitNormal);
            return std::isfinite(normalDot)
                && normalDot > 0.9999f
                && std::fabs(receiverGeometry.faceDist - hitGeometry.faceDist) <= 0.03125f;
        };

        std::vector<float3> lightSamples(
            bsp.get_lightsamples().size(),
            make_float3()
        );
        size_t totalStaticPropVertices = 0;
        for (const Common::StaticPropLightingProp& prop : state.scene.staticPropLightingProps) {
            for (const Common::StaticPropLightingMesh& mesh : prop.meshes) {
                totalStaticPropVertices += mesh.vertices.size();
            }
        }
        std::vector<float3> staticPropVertexLighting(
            totalStaticPropVertices,
            make_float3()
        );
        std::vector<::BSP::DFace> dFaces = bsp.get_dfaces();
        std::vector<OptixRT::SunRay> rays;
        std::vector<float> rayOriginVisibility;
        std::vector<RayContribution> contributions;
        std::vector<SampleNormalization> sampleNormalizations;

        const size_t numFaces = bsp.get_faces().size();
        size_t processedFaces = 0;

        std::cout << "Building core OptiX direct-lighting ray batch for "
            << numFaces << " faces at " << HOST_DIRECT_SUBSAMPLES
            << " spp/luxel..." << std::endl;

        const auto startTime = Clock::now();

        auto append_direct_contributions =
            [&](float3 samplePos,
                float3 shadowNormal,
                const float3* samplePlaneNormals,
                size_t planeCount,
                int16_t sampleCluster,
                float directFootprintRadius,
                DirectOutputKind outputKind,
                size_t outputIndex,
                size_t outputStride,
                uint32_t ignoreSourceId,
                uint32_t receiverFaceIndex) {
            for (const ::BSP::DWorldLight& light : bsp.get_worldlights()) {
                if (light.type == ::BSP::EMIT_SKYAMBIENT) {
                    continue;
                }

                if (light.type == ::BSP::EMIT_SKYLIGHT) {
                    const float3 lightNormal = make_float3(light.normal);
                    const float3 sunDir = safe_normalized(lightNormal) * -1.0f;
                    if (!is_finite(sunDir)) {
                        continue;
                    }

                    bool anyPlaneVisible = false;
                    RayContribution contribution{};
                    contribution.outputKind = outputKind;
                    contribution.outputIndex = outputIndex;
                    contribution.outputStride = outputStride;
                    contribution.planeCount = planeCount;
                    contribution.visibilityMode = DirectVisibilityMode::DirectionalSun;
                    contribution.ignoreSourceId = ignoreSourceId;
                    contribution.receiverFaceIndex = receiverFaceIndex;
                    for (size_t planeIndex = 0; planeIndex < planeCount; ++planeIndex) {
                        contribution.color[planeIndex] = make_float3();
                        const float ndotl = dot(samplePlaneNormals[planeIndex], sunDir);
                        if (ndotl > 0.0f) {
                            contribution.color[planeIndex] =
                                make_float3(light.intensity) * ndotl * 255.0f;
                            anyPlaneVisible = true;
                        }
                    }
                    if (!anyPlaneVisible) {
                        continue;
                    }

                    const float3 tangent = make_tangent(samplePlaneNormals[0]);
                    const float3 bitangent = make_bitangent(samplePlaneNormals[0], tangent);
                    const float3 sunTangent = make_tangent(sunDir);
                    const float3 sunBitangent = make_bitangent(sunDir, sunTangent);
                    const uint32_t sampleSeed = hash_u32(
                        static_cast<uint32_t>(outputIndex)
                        ^ (static_cast<uint32_t>(sampleCluster) << 16)
                        ^ (static_cast<uint32_t>(receiverFaceIndex) * 0x9e3779b9u)
                    );

                    contribution.firstRayIndex = rays.size();
                    for (size_t rayIndex = 0;
                        rayIndex < HOST_DIRECT_SUN_SHADOW_RAYS;
                        ++rayIndex) {
                        const float2 footprintDisk = concentric_sample_disk(
                            hammersley_2d(
                                rayIndex,
                                HOST_DIRECT_SUN_SHADOW_RAYS,
                                sampleSeed
                            )
                        );
                        const float2 sunDisk = concentric_sample_disk(
                            hammersley_2d(
                                rayIndex,
                                HOST_DIRECT_SUN_SHADOW_RAYS,
                                sampleSeed ^ 0x85ebca6bu
                            )
                        );
                        const float3 receiverOffset =
                            tangent * (footprintDisk.x * directFootprintRadius)
                            + bitangent * (footprintDisk.y * directFootprintRadius);
                        const float3 rayDir = safe_normalized(
                            sunDir
                            + sunTangent * (sunDisk.x * HOST_SUN_ANGULAR_RADIUS)
                            + sunBitangent * (sunDisk.y * HOST_SUN_ANGULAR_RADIUS)
                        );
                        OptixRT::SunRay ray{};
                        const float bias = direct_ray_bias(
                            directFootprintRadius,
                            shadowNormal,
                            rayDir
                        );
                        ray.origin =
                            samplePos
                            + receiverOffset
                            + shadowNormal * bias
                            + rayDir * HOST_RAY_TMIN;
                        ray.direction = rayDir;
                        ray.tmin = HOST_RAY_TMIN;
                        ray.tmax = HOST_COORD_EXTENT;
                        ray.visibilityMask = 0xff;
                        if (is_valid_ray(ray)) {
                            rayOriginVisibility.push_back(
                                direct_origin_shadow_visibility(bsp, ray.origin)
                            );
                            rays.push_back(ray);
                        }
                    }

                    contribution.rayCount = rays.size() - contribution.firstRayIndex;
                    if (contribution.rayCount == 0) {
                        continue;
                    }
                    contributions.push_back(contribution);
                    continue;
                }

                if (!cluster_in_pvs(
                    bsp,
                    sampleCluster,
                    static_cast<int16_t>(light.cluster)
                )) {
                    continue;
                }

                const float3 lightPos = make_float3(light.origin);
                if (!is_finite(lightPos)) {
                    continue;
                }

                const float3 diff = samplePos - lightPos;
                if (dot(diff, shadowNormal) >= 0.0f) {
                    continue;
                }

                const float distToLight = len(diff);
                if (distToLight <= 1e-6f) {
                    continue;
                }

                const float3 dir = diff / distToLight;
                float penumbraScale = 1.0f;
                if (!is_finite(dir)) {
                    continue;
                }

                if (light.type == ::BSP::EMIT_SPOTLIGHT) {
                    const float3 lightNorm = make_float3(light.normal);
                    if (!is_finite(lightNorm)) {
                        continue;
                    }

                    const float lightDot = dot(dir, lightNorm);
                    if (lightDot < light.stopdot2) {
                        continue;
                    }
                    if (lightDot < light.stopdot) {
                        penumbraScale =
                            (lightDot - light.stopdot2)
                            / (light.stopdot - light.stopdot2);
                    }
                }

                const float3 lightDir = safe_normalized(lightPos - samplePos);
                if (!is_finite(lightDir)) {
                    continue;
                }
                const float3 shadowStart =
                    samplePos
                    + shadowNormal * direct_ray_bias(
                        directFootprintRadius,
                        shadowNormal,
                        lightDir
                    );
                const float3 shadowDelta = lightPos - shadowStart;
                const float shadowDist = len(shadowDelta);
                if (shadowDist <= 1e-6f) {
                    continue;
                }

                const float attenuation = attenuate_host(light, distToLight);
                if (!std::isfinite(attenuation) || attenuation <= 1e-6f) {
                    continue;
                }

                RayContribution contribution{};
                contribution.outputKind = outputKind;
                contribution.outputIndex = outputIndex;
                contribution.outputStride = outputStride;
                contribution.planeCount = planeCount;
                contribution.visibilityMode = DirectVisibilityMode::PointOrSpot;
                contribution.ignoreSourceId = ignoreSourceId;
                contribution.receiverFaceIndex = receiverFaceIndex;
                bool anyPlaneVisible = false;
                const float3 shadowLightDir = shadowDelta / shadowDist;
                for (size_t planeIndex = 0; planeIndex < planeCount; ++planeIndex) {
                    contribution.color[planeIndex] = make_float3();
                    const float ndotl = dot(samplePlaneNormals[planeIndex], shadowLightDir);
                    if (ndotl <= 0.0f) {
                        continue;
                    }

                    float3 lightContribution = make_float3(light.intensity);
                    lightContribution *=
                        penumbraScale * 255.0f * ndotl
                        / attenuation;
                    contribution.color[planeIndex] = lightContribution;
                    anyPlaneVisible = true;
                }
                if (!anyPlaneVisible) {
                    continue;
                }

                OptixRT::SunRay ray{};
                ray.origin = shadowStart;
                ray.direction = shadowDelta / shadowDist;
                ray.tmin = HOST_RAY_TMIN;
                ray.tmax = shadowDist - HOST_RAY_TMIN;
                ray.visibilityMask = 0xff;
                if (!is_valid_ray(ray)) {
                    continue;
                }

                contribution.firstRayIndex = rays.size();
                contribution.rayCount = 1;
                rayOriginVisibility.push_back(
                    direct_origin_shadow_visibility(bsp, ray.origin)
                );
                rays.push_back(ray);
                contributions.push_back(contribution);
            }
        };

        for (size_t faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.lightOffset < 0) {
                ++processedFaces;
                continue;
            }

            const Geometry::DispGeometry& dispGeometry = state.dispGeometry[faceIndex];
            const Geometry::FaceGeometry& faceGeometry = state.faceGeometry[faceIndex];
            const bool isDispReceiver = dispGeometry.valid;
            const bool isFaceReceiver = faceGeometry.valid && !faceGeometry.isDisplacement;
            const bool validReceiver = isDispReceiver || isFaceReceiver;
            if (!validReceiver) {
                ++processedFaces;
                continue;
            }

            const size_t lightmapWidth = static_cast<size_t>(face.lightmapTextureSizeInLuxels[0] + 1);
            const size_t lightmapHeight = static_cast<size_t>(face.lightmapTextureSizeInLuxels[1] + 1);
            const size_t lightmapSize = lightmapWidth * lightmapHeight;
            const size_t lightmapStartIndex = static_cast<size_t>(face.lightOffset / sizeof(::BSP::RGBExp32));

            for (size_t t = 0; t < lightmapHeight; ++t) {
                for (size_t s = 0; s < lightmapWidth; ++s) {
                    const size_t sampleIndex = t * lightmapWidth + s;
                    const size_t lightmapIndex = lightmapStartIndex + sampleIndex;
                    const size_t planeCount = sample_plane_count(bsp, faceIndex);
                    const size_t subSampleCount = isFaceReceiver
                        ? HOST_DIRECT_SUBSAMPLES
                        : 1;
                    size_t validSubsamples = 0;

                    for (size_t directSample = 0; directSample < subSampleCount; ++directSample) {
                        const float2 sampleOffset = luxel_subsample_offset(directSample);
                        float3 samplePos;
                        float3 sampleNormal;
                        float3 shadowNormal;
                        float directFootprintRadius = HOST_DIRECT_FOOTPRINT_MIN_RADIUS;
                        if (!core_receiver_sample(
                            state,
                            faceIndex,
                            s,
                            t,
                            sampleOffset,
                            samplePos,
                            sampleNormal,
                            shadowNormal,
                            directFootprintRadius
                        )) {
                            continue;
                        }
                        ++validSubsamples;

                        float3 samplePlaneNormals[NUM_BUMP_VECTS + 1];
                        samplePlaneNormals[0] = sampleNormal;
                        if (planeCount > 1) {
                            float3 flatNormal = sampleNormal;
                            const ::BSP::TexInfo& texInfo =
                                bsp.get_faces()[faceIndex].get_texinfo();
                            if (dispGeometry.valid) {
                                flatNormal = to_float3(dispGeometry.faceNormal);
                            }
                            else if (faceGeometry.valid) {
                                flatNormal = to_float3(faceGeometry.faceNormal);
                            }
                            get_bump_normals(
                                texInfo.textureVecs[0],
                                texInfo.textureVecs[1],
                                flatNormal,
                                sampleNormal,
                                &samplePlaneNormals[1]
                            );
                        }

                        const float3 clusterPos =
                            samplePos + shadowNormal * HOST_RAY_BIAS;
                        const ::BSP::Vec3<float> sampleVec{
                            clusterPos.x,
                            clusterPos.y,
                            clusterPos.z,
                        };
                        const int16_t sampleCluster = bsp.cluster_for_pos(sampleVec);

                        append_direct_contributions(
                            samplePos,
                            shadowNormal,
                            samplePlaneNormals,
                            planeCount,
                            sampleCluster,
                            directFootprintRadius,
                            DirectOutputKind::Lightmap,
                            lightmapIndex,
                            lightmapSize,
                            0xffffffffu,
                            static_cast<uint32_t>(faceIndex)
                        );
                    }

                    if (validSubsamples > 0) {
                        sampleNormalizations.push_back(SampleNormalization{
                            DirectOutputKind::Lightmap,
                            lightmapIndex,
                            lightmapSize,
                            planeCount,
                            validSubsamples,
                        });
                    }
                }
            }

            ++processedFaces;
            if (processedFaces % 32 == 0 || processedFaces == numFaces) {
                std::cout << "    " << processedFaces << "/"
                    << numFaces << " faces batched..." << std::endl;
            }
        }

        size_t staticPropVertexIndex = 0;
        for (const Common::StaticPropLightingProp& prop : state.scene.staticPropLightingProps) {
            for (const Common::StaticPropLightingMesh& mesh : prop.meshes) {
                for (size_t vertexIndex = 0; vertexIndex < mesh.vertices.size(); ++vertexIndex) {
                    const Common::StaticPropLightingVertex& vertex = mesh.vertices[vertexIndex];
                    const float3 samplePos = to_float3(vertex.pos);
                    const float3 sampleNormal = safe_normalized(to_float3(vertex.normal));
                    const float3 shadowNormal = sampleNormal;
                    const float3 clusterPos =
                        samplePos + shadowNormal * HOST_RAY_BIAS;
                    const ::BSP::Vec3<float> sampleVec{
                        clusterPos.x,
                        clusterPos.y,
                        clusterPos.z,
                    };
                    const int16_t sampleCluster = bsp.cluster_for_pos(sampleVec);
                    float3 samplePlaneNormals[NUM_BUMP_VECTS + 1];
                    samplePlaneNormals[0] = sampleNormal;

                    float3 receiverPos = samplePos;
                    uint32_t ignoreSourceId = 0xffffffffu;
                    if ((prop.flags & ::BSP::STATIC_PROP_NO_PER_VERTEX_LIGHTING) != 0) {
                        const Common::Vec3f origin =
                            prop.hasLightingOrigin ? prop.lightingOrigin : prop.origin;
                        receiverPos = to_float3(origin);
                        ignoreSourceId = static_cast<uint32_t>(prop.propIndex);
                    }
                    else if ((prop.flags & ::BSP::STATIC_PROP_NO_SELF_SHADOWING) != 0) {
                        ignoreSourceId = static_cast<uint32_t>(prop.propIndex);
                    }

                    append_direct_contributions(
                        receiverPos,
                        shadowNormal,
                        samplePlaneNormals,
                        1,
                        sampleCluster,
                        HOST_DIRECT_FOOTPRINT_MIN_RADIUS,
                        DirectOutputKind::StaticPropVertex,
                        staticPropVertexIndex,
                        0,
                        ignoreSourceId,
                        0xffffffffu
                    );
                    sampleNormalizations.push_back(SampleNormalization{
                        DirectOutputKind::StaticPropVertex,
                        staticPropVertexIndex,
                        0,
                        1,
                        1,
                    });
                    ++staticPropVertexIndex;
                }
            }
        }

        std::cout << "Tracing " << rays.size()
            << " core direct-lighting rays with OptiX..." << std::endl;

        std::vector<OptixRT::RayHit> directHits;
        tracer.trace_batch(rays, directHits);

        for (size_t i = 0; i < contributions.size(); ++i) {
            const RayContribution& contribution = contributions[i];
            if (contribution.firstRayIndex + contribution.rayCount > directHits.size()) {
                continue;
            }
            if (contribution.firstRayIndex + contribution.rayCount > rayOriginVisibility.size()) {
                continue;
            }

            float visibilityScale = 0.0f;
            Tracing::LineTraceResult lineTrace;
            if (contribution.visibilityMode == DirectVisibilityMode::DirectionalSun) {
                float visibleRays = 0.0f;
                for (size_t rayOffset = 0; rayOffset < contribution.rayCount; ++rayOffset) {
                    const size_t rayIndex = contribution.firstRayIndex + rayOffset;
                    lineTrace = Tracing::test_line(directHits[rayIndex], triangles);
                    visibleRays +=
                        rayOriginVisibility[rayIndex]
                        * direct_hit_shadow_visibility(lineTrace);
                }
                visibilityScale =
                    visibleRays / static_cast<float>(contribution.rayCount);
            }
            else {
                const size_t rayIndex = contribution.firstRayIndex;
                lineTrace = Tracing::test_line(directHits[rayIndex], triangles);
            }
            const bool ignoredSelfHit =
                contribution.ignoreSourceId != 0xffffffffu
                && lineTrace.hit
                && lineTrace.sourceKind
                    == static_cast<uint32_t>(Common::OccluderTriangle::SourceKind::StaticProp)
                && lineTrace.sourceId == contribution.ignoreSourceId;
            const bool ignoredCoplanarHit =
                contribution.receiverFaceIndex != 0xffffffffu
                && lineTrace.hit
                && (lineTrace.sourceKind
                        == static_cast<uint32_t>(Common::OccluderTriangle::SourceKind::Face)
                    || lineTrace.sourceKind
                        == static_cast<uint32_t>(Common::OccluderTriangle::SourceKind::Displacement))
                && surface_faces_coplanar(
                    contribution.receiverFaceIndex,
                    lineTrace.sourceId
                );
            if (contribution.visibilityMode == DirectVisibilityMode::PointOrSpot) {
                const size_t rayIndex = contribution.firstRayIndex;
                const float tracedVisibility =
                    (ignoredSelfHit || ignoredCoplanarHit)
                        ? 1.0f
                        : direct_hit_shadow_visibility(lineTrace);
                visibilityScale =
                    rayOriginVisibility[rayIndex]
                    * tracedVisibility;
            }

            if (visibilityScale > 0.0f) {
                for (size_t planeIndex = 0; planeIndex < contribution.planeCount; ++planeIndex) {
                    const size_t planeOutputIndex =
                        contribution.outputIndex + planeIndex * contribution.outputStride;
                    if (contribution.outputKind == DirectOutputKind::Lightmap) {
                        lightSamples[planeOutputIndex] +=
                            contribution.color[planeIndex] * visibilityScale;
                    }
                    else {
                        staticPropVertexLighting[planeOutputIndex] +=
                            contribution.color[planeIndex] * visibilityScale;
                    }
                }
            }
        }

        for (const SampleNormalization& normalization : sampleNormalizations) {
            if (normalization.validSubsamples == 0) {
                continue;
            }
            const float scale = 1.0f / static_cast<float>(normalization.validSubsamples);
            for (size_t planeIndex = 0; planeIndex < normalization.planeCount; ++planeIndex) {
                const size_t planeOutputIndex =
                    normalization.outputIndex + planeIndex * normalization.outputStride;
                if (normalization.outputKind == DirectOutputKind::Lightmap) {
                    lightSamples[planeOutputIndex] *= scale;
                }
                else {
                    staticPropVertexLighting[planeOutputIndex] *= scale;
                }
            }
        }

        std::vector<float3> faceAverages(numFaces, make_float3());
        for (size_t faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.lightOffset < 0) {
                continue;
            }

            const size_t lightmapWidth = static_cast<size_t>(face.lightmapTextureSizeInLuxels[0] + 1);
            const size_t lightmapHeight = static_cast<size_t>(face.lightmapTextureSizeInLuxels[1] + 1);
            const size_t lightmapSize = lightmapWidth * lightmapHeight;
            const size_t lightmapStartIndex = static_cast<size_t>(face.lightOffset / sizeof(::BSP::RGBExp32));
            if (lightmapSize == 0) {
                continue;
            }

            float3 totalLight = make_float3();
            for (size_t sampleIndex = 0; sampleIndex < lightmapSize; ++sampleIndex) {
                totalLight += lightSamples[lightmapStartIndex + sampleIndex];
            }
            faceAverages[faceIndex] = totalLight / static_cast<float>(lightmapSize);
        }

        for (size_t faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.lightOffset < 0) {
                continue;
            }

            const size_t lightmapWidth = static_cast<size_t>(face.lightmapTextureSizeInLuxels[0] + 1);
            const size_t lightmapHeight = static_cast<size_t>(face.lightmapTextureSizeInLuxels[1] + 1);
            const size_t lightmapSize = lightmapWidth * lightmapHeight;
            const size_t lightmapStartIndex = static_cast<size_t>(face.lightOffset / sizeof(::BSP::RGBExp32));

            float3 totalLight = make_float3();
            for (size_t sampleIndex = 0; sampleIndex < lightmapSize; ++sampleIndex) {
                const size_t lightmapIndex = lightmapStartIndex + sampleIndex;
                totalLight += lightSamples[lightmapIndex];
            }

            const float3 avgLight = totalLight / static_cast<float>(lightmapSize);
            faceAverages[faceIndex] = avgLight;
            if (lightmapStartIndex > 0) {
                lightSamples[lightmapStartIndex - 1] = avgLight;
            }

            dFaces[faceIndex].styles[0] = 0x00;
            dFaces[faceIndex].styles[1] = 0xFF;
            dFaces[faceIndex].styles[2] = 0xFF;
            dFaces[faceIndex].styles[3] = 0xFF;
        }

        ::CUDABSP::CUDABSP cudaBSP;
        CUDA_CHECK_ERROR(cudaMemcpy(
            &cudaBSP,
            pCudaBSP,
            sizeof(::CUDABSP::CUDABSP),
            cudaMemcpyDeviceToHost
        ));

        if (!lightSamples.empty()) {
            CUDA_CHECK_ERROR(cudaMemcpy(
                cudaBSP.lightSamples,
                lightSamples.data(),
                sizeof(float3) * lightSamples.size(),
                cudaMemcpyHostToDevice
            ));
        }

        if (!dFaces.empty()) {
            CUDA_CHECK_ERROR(cudaMemcpy(
                cudaBSP.faces,
                dFaces.data(),
                sizeof(::BSP::DFace) * dFaces.size(),
                cudaMemcpyHostToDevice
            ));
        }

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        staticPropVertexIndex = 0;
        for (Common::StaticPropLightingProp& prop : state.scene.staticPropLightingProps) {
            for (Common::StaticPropLightingMesh& mesh : prop.meshes) {
                for (size_t vertexIndex = 0; vertexIndex < mesh.vertices.size(); ++vertexIndex) {
                    if (staticPropVertexIndex >= staticPropVertexLighting.size()
                        || vertexIndex >= mesh.colors.size()) {
                        break;
                    }

                    mesh.colors[vertexIndex] = Common::make_vec3(
                        staticPropVertexLighting[staticPropVertexIndex].x,
                        staticPropVertexLighting[staticPropVertexIndex].y,
                        staticPropVertexLighting[staticPropVertexIndex].z
                    );
                    ++staticPropVertexIndex;
                }
            }
        }

        const auto endTime = Clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime
        );
        std::cout << "Done! (" << ms.count() << " ms)" << std::endl;
    }

    void write_static_prop_direct_lighting(
        ::BSP::BSP& bsp,
        const RuntimeState& state,
        const std::string& outputBspFilename
    )
    {
        const std::vector<Common::StaticPropLightingProp>& props =
            state.scene.staticPropLightingProps;
        const uint32_t oldFlags = bsp.get_level_flags();
        if (props.empty()) {
            bsp.set_level_flags(oldFlags & ~LVLFLAGS_BAKED_STATIC_PROP_LIGHTING);
            return;
        }

        (void)outputBspFilename;
        std::map<std::string, std::vector<uint8_t>> vhvEntries;

        for (const Common::StaticPropLightingProp& prop : props) {
            std::vector<uint8_t> vhvBytes = build_vhv_bytes(prop);
            if (vhvBytes.empty()) {
                continue;
            }

            vhvEntries.emplace(
                "sp_" + std::to_string(prop.propIndex) + ".vhv",
                std::move(vhvBytes)
            );
        }

        if (vhvEntries.empty()) {
            bsp.set_level_flags(oldFlags & ~LVLFLAGS_BAKED_STATIC_PROP_LIGHTING);
            return;
        }

        const auto& extras = bsp.get_extras();
        const auto pakIt = extras.find(::BSP::LUMP_PAKFILE);
        const std::vector<uint8_t> existingPak =
            pakIt != extras.end() ? pakIt->second : std::vector<uint8_t>{};
        if (!existingPak.empty() && !parse_zip_metadata(existingPak).valid) {
            std::cerr << "WARNING: existing pak lump is not a supported ZIP layout; "
                "replacing it with static prop lighting pack" << std::endl;
        }
        const std::vector<uint8_t> packedPak =
            build_zip_with_replaced_entries(existingPak, vhvEntries);
        bsp.set_extra_lump(::BSP::LUMP_PAKFILE, packedPak);
        bsp.set_level_flags(oldFlags | LVLFLAGS_BAKED_STATIC_PROP_LIGHTING);
    }
}
