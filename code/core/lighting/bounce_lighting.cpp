#include "bounce_lighting.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <iostream>
#include <tuple>
#include <vector>

#include "../../cudautils.h"
#include "../state.h"
#include "../common/math.h"
#include "../geometry/disp_geometry.h"
#include "../tracing/trace.h"

namespace SilkRAD::Core::Lighting {
    namespace {
        static constexpr size_t NUM_VERTEX_NORMALS = 162;
        static constexpr float HOST_RAY_BIAS = 0.03125f;
        static constexpr float HOST_RAY_TMIN = 0.001f;
        static constexpr float COORD_EXTENT = 16384.0f;
        static constexpr float BOUNCE_TRACE_DISTANCE = COORD_EXTENT * 1.74f;
        static constexpr uint32_t INVALID_SOURCE_FACE = 0xffffffffu;
        static constexpr size_t BOUNCE_DIRECTION_STRIDE = 4;
        static constexpr float WEIGHT_EPS = 1e-6f;
        static constexpr int BOUNCE_DENOISE_RADIUS = 1;
        static constexpr float BOUNCE_DENOISE_SPATIAL_SIGMA = 1.25f;
        static constexpr float BOUNCE_DENOISE_NORMAL_DOT_MIN = 0.9f;
        static constexpr float BOUNCE_DENOISE_WORLD_DISTANCE = 48.0f;

        struct LuxelRadial {
            size_t faceIndex = 0;
            size_t width = 0;
            size_t height = 0;
            std::vector<float3> light;
            std::vector<float> weight;
        };

        static const float3 ANORMS[NUM_VERTEX_NORMALS] = {
            { -0.525731f, 0.000000f, 0.850651f }, { -0.442863f, 0.238856f, 0.864188f },
            { -0.295242f, 0.000000f, 0.955423f }, { -0.309017f, 0.500000f, 0.809017f },
            { -0.162460f, 0.262866f, 0.951056f }, { 0.000000f, 0.000000f, 1.000000f },
            { 0.000000f, 0.850651f, 0.525731f }, { -0.147621f, 0.716567f, 0.681718f },
            { 0.147621f, 0.716567f, 0.681718f }, { 0.000000f, 0.525731f, 0.850651f },
            { 0.309017f, 0.500000f, 0.809017f }, { 0.525731f, 0.000000f, 0.850651f },
            { 0.295242f, 0.000000f, 0.955423f }, { 0.442863f, 0.238856f, 0.864188f },
            { 0.162460f, 0.262866f, 0.951056f }, { -0.681718f, 0.147621f, 0.716567f },
            { -0.809017f, 0.309017f, 0.500000f }, { -0.587785f, 0.425325f, 0.688191f },
            { -0.850651f, 0.525731f, 0.000000f }, { -0.864188f, 0.442863f, 0.238856f },
            { -0.716567f, 0.681718f, 0.147621f }, { -0.688191f, 0.587785f, 0.425325f },
            { -0.500000f, 0.809017f, 0.309017f }, { -0.238856f, 0.864188f, 0.442863f },
            { -0.425325f, 0.688191f, 0.587785f }, { -0.716567f, 0.681718f, -0.147621f },
            { -0.500000f, 0.809017f, -0.309017f }, { -0.525731f, 0.850651f, 0.000000f },
            { 0.000000f, 0.850651f, -0.525731f }, { -0.238856f, 0.864188f, -0.442863f },
            { 0.000000f, 0.955423f, -0.295242f }, { -0.262866f, 0.951056f, -0.162460f },
            { 0.000000f, 1.000000f, 0.000000f }, { 0.000000f, 0.955423f, 0.295242f },
            { -0.262866f, 0.951056f, 0.162460f }, { 0.238856f, 0.864188f, 0.442863f },
            { 0.262866f, 0.951056f, 0.162460f }, { 0.500000f, 0.809017f, 0.309017f },
            { 0.238856f, 0.864188f, -0.442863f }, { 0.262866f, 0.951056f, -0.162460f },
            { 0.500000f, 0.809017f, -0.309017f }, { 0.850651f, 0.525731f, 0.000000f },
            { 0.716567f, 0.681718f, 0.147621f }, { 0.716567f, 0.681718f, -0.147621f },
            { 0.525731f, 0.850651f, 0.000000f }, { 0.425325f, 0.688191f, 0.587785f },
            { 0.864188f, 0.442863f, 0.238856f }, { 0.688191f, 0.587785f, 0.425325f },
            { 0.809017f, 0.309017f, 0.500000f }, { 0.681718f, 0.147621f, 0.716567f },
            { 0.587785f, 0.425325f, 0.688191f }, { 0.955423f, 0.295242f, 0.000000f },
            { 1.000000f, 0.000000f, 0.000000f }, { 0.951056f, 0.162460f, 0.262866f },
            { 0.850651f, -0.525731f, 0.000000f }, { 0.955423f, -0.295242f, 0.000000f },
            { 0.864188f, -0.442863f, 0.238856f }, { 0.951056f, -0.162460f, 0.262866f },
            { 0.809017f, -0.309017f, 0.500000f }, { 0.681718f, -0.147621f, 0.716567f },
            { 0.850651f, 0.000000f, 0.525731f }, { 0.864188f, 0.442863f, -0.238856f },
            { 0.809017f, 0.309017f, -0.500000f }, { 0.951056f, 0.162460f, -0.262866f },
            { 0.525731f, 0.000000f, -0.850651f }, { 0.681718f, 0.147621f, -0.716567f },
            { 0.681718f, -0.147621f, -0.716567f }, { 0.850651f, 0.000000f, -0.525731f },
            { 0.809017f, -0.309017f, -0.500000f }, { 0.864188f, -0.442863f, -0.238856f },
            { 0.951056f, -0.162460f, -0.262866f }, { 0.147621f, 0.716567f, -0.681718f },
            { 0.309017f, 0.500000f, -0.809017f }, { 0.425325f, 0.688191f, -0.587785f },
            { 0.442863f, 0.238856f, -0.864188f }, { 0.587785f, 0.425325f, -0.688191f },
            { 0.688191f, 0.587785f, -0.425325f }, { -0.147621f, 0.716567f, -0.681718f },
            { -0.309017f, 0.500000f, -0.809017f }, { 0.000000f, 0.525731f, -0.850651f },
            { -0.525731f, 0.000000f, -0.850651f }, { -0.442863f, 0.238856f, -0.864188f },
            { -0.295242f, 0.000000f, -0.955423f }, { -0.162460f, 0.262866f, -0.951056f },
            { 0.000000f, 0.000000f, -1.000000f }, { 0.295242f, 0.000000f, -0.955423f },
            { 0.162460f, 0.262866f, -0.951056f }, { -0.442863f, -0.238856f, -0.864188f },
            { -0.309017f, -0.500000f, -0.809017f }, { -0.162460f, -0.262866f, -0.951056f },
            { 0.000000f, -0.850651f, -0.525731f }, { -0.147621f, -0.716567f, -0.681718f },
            { 0.147621f, -0.716567f, -0.681718f }, { 0.000000f, -0.525731f, -0.850651f },
            { 0.309017f, -0.500000f, -0.809017f }, { 0.442863f, -0.238856f, -0.864188f },
            { 0.162460f, -0.262866f, -0.951056f }, { 0.238856f, -0.864188f, -0.442863f },
            { 0.500000f, -0.809017f, -0.309017f }, { 0.425325f, -0.688191f, -0.587785f },
            { 0.716567f, -0.681718f, -0.147621f }, { 0.688191f, -0.587785f, -0.425325f },
            { 0.587785f, -0.425325f, -0.688191f }, { 0.000000f, -0.955423f, -0.295242f },
            { 0.000000f, -1.000000f, 0.000000f }, { 0.262866f, -0.951056f, -0.162460f },
            { 0.000000f, -0.850651f, 0.525731f }, { 0.000000f, -0.955423f, 0.295242f },
            { 0.238856f, -0.864188f, 0.442863f }, { 0.262866f, -0.951056f, 0.162460f },
            { 0.500000f, -0.809017f, 0.309017f }, { 0.716567f, -0.681718f, 0.147621f },
            { 0.525731f, -0.850651f, 0.000000f }, { -0.238856f, -0.864188f, -0.442863f },
            { -0.500000f, -0.809017f, -0.309017f }, { -0.262866f, -0.951056f, -0.162460f },
            { -0.850651f, -0.525731f, 0.000000f }, { -0.716567f, -0.681718f, -0.147621f },
            { -0.716567f, -0.681718f, 0.147621f }, { -0.525731f, -0.850651f, 0.000000f },
            { -0.500000f, -0.809017f, 0.309017f }, { -0.238856f, -0.864188f, 0.442863f },
            { -0.262866f, -0.951056f, 0.162460f }, { -0.864188f, -0.442863f, 0.238856f },
            { -0.809017f, -0.309017f, 0.500000f }, { -0.688191f, -0.587785f, 0.425325f },
            { -0.681718f, -0.147621f, 0.716567f }, { -0.442863f, -0.238856f, 0.864188f },
            { -0.587785f, -0.425325f, 0.688191f }, { -0.309017f, -0.500000f, 0.809017f },
            { -0.147621f, -0.716567f, 0.681718f }, { -0.425325f, -0.688191f, 0.587785f },
            { -0.162460f, -0.262866f, 0.951056f }, { 0.442863f, -0.238856f, 0.864188f },
            { 0.162460f, -0.262866f, 0.951056f }, { 0.309017f, -0.500000f, 0.809017f },
            { 0.147621f, -0.716567f, 0.681718f }, { 0.000000f, -0.525731f, 0.850651f },
            { 0.425325f, -0.688191f, 0.587785f }, { 0.587785f, -0.425325f, 0.688191f },
            { 0.688191f, -0.587785f, 0.425325f }, { -0.955423f, 0.295242f, 0.000000f },
            { -0.951056f, 0.162460f, 0.262866f }, { -1.000000f, 0.000000f, 0.000000f },
            { -0.850651f, 0.000000f, 0.525731f }, { -0.955423f, -0.295242f, 0.000000f },
            { -0.951056f, -0.162460f, 0.262866f }, { -0.864188f, 0.442863f, -0.238856f },
            { -0.951056f, 0.162460f, -0.262866f }, { -0.809017f, 0.309017f, -0.500000f },
            { -0.864188f, -0.442863f, -0.238856f }, { -0.951056f, -0.162460f, -0.262866f },
            { -0.809017f, -0.309017f, -0.500000f }, { -0.681718f, 0.147621f, -0.716567f },
            { -0.681718f, -0.147621f, -0.716567f }, { -0.850651f, 0.000000f, -0.525731f },
            { -0.688191f, 0.587785f, -0.425325f }, { -0.587785f, 0.425325f, -0.688191f },
            { -0.425325f, 0.688191f, -0.587785f }, { -0.425325f, -0.688191f, -0.587785f },
            { -0.587785f, -0.425325f, -0.688191f }, { -0.688191f, -0.587785f, -0.425325f },
        };

        struct BounceContribution {
            bool staticPropVertex = false;
            size_t outputIndex = 0;
            size_t outputStride = 0;
            size_t planeCount = 1;
            float weight = 0.0f;
        };

        float3 to_float3(const Common::Vec3f& v) { return make_float3(v.x, v.y, v.z); }
        Common::Vec3f from_float3(float3 v) { return Common::make_vec3(v.x, v.y, v.z); }
        bool is_finite(float value) { return std::isfinite(value) != 0; }
        bool is_finite(float3 v) { return is_finite(v.x) && is_finite(v.y) && is_finite(v.z); }

        float3 safe_normalized(float3 v)
        {
            const float vlen = len(v);
            if (!std::isfinite(vlen) || vlen <= 1e-6f) {
                return make_float3();
            }
            return v / vlen;
        }

        float3 component_multiply(float3 a, float3 b)
        {
            return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
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

        bool face_is_sky(const ::BSP::BSP& bsp, size_t faceIndex)
        {
            const ::BSP::Face& face = bsp.get_faces()[faceIndex];
            return (face.get_texinfo().flags & ::BSP::SURF_SKY) != 0;
        }

        bool vec3_nearly_equal(Common::Vec3f a, Common::Vec3f b, float eps = 1e-4f)
        {
            return std::fabs(a.x - b.x) <= eps
                && std::fabs(a.y - b.y) <= eps
                && std::fabs(a.z - b.z) <= eps;
        }

        using EdgeKey = std::pair<uint16_t, uint16_t>;

        std::vector<std::vector<size_t>> build_face_neighbors(const ::BSP::BSP& bsp)
        {
            std::vector<std::vector<size_t>> neighbors(bsp.get_faces().size());
            const std::vector<::BSP::DFace>& dfaces = bsp.get_dfaces();
            const std::vector<::BSP::DEdge>& edges = bsp.get_edges();
            const std::vector<int32_t>& surfedges = bsp.get_surfedges();
            std::map<EdgeKey, std::vector<size_t>> edgeFaces;

            for (size_t faceIndex = 0; faceIndex < dfaces.size(); ++faceIndex) {
                const ::BSP::DFace& face = dfaces[faceIndex];
                const int firstEdge = face.firstEdge;
                const int lastEdge = face.firstEdge + face.numEdges;
                if (firstEdge < 0 || lastEdge < firstEdge || static_cast<size_t>(lastEdge) > surfedges.size()) {
                    continue;
                }

                for (int i = firstEdge; i < lastEdge; ++i) {
                    int32_t surfEdge = surfedges[static_cast<size_t>(i)];
                    if (surfEdge < 0) {
                        surfEdge = -surfEdge;
                    }
                    if (surfEdge < 0 || static_cast<size_t>(surfEdge) >= edges.size()) {
                        continue;
                    }
                    const ::BSP::DEdge& edge = edges[static_cast<size_t>(surfEdge)];
                    const EdgeKey key = std::minmax(edge.vertex1, edge.vertex2);
                    edgeFaces[key].push_back(faceIndex);
                }
            }

            for (const auto& [key, faces] : edgeFaces) {
                (void)key;
                for (size_t i = 0; i < faces.size(); ++i) {
                    for (size_t j = 0; j < faces.size(); ++j) {
                        if (i == j) {
                            continue;
                        }
                        neighbors[faces[i]].push_back(faces[j]);
                    }
                }
            }

            for (std::vector<size_t>& faceNeighbors : neighbors) {
                std::sort(faceNeighbors.begin(), faceNeighbors.end());
                faceNeighbors.erase(
                    std::unique(faceNeighbors.begin(), faceNeighbors.end()),
                    faceNeighbors.end()
                );
            }

            return neighbors;
        }

        float3 surface_reflectivity(const ::BSP::BSP& bsp, size_t faceIndex)
        {
            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.texInfo < 0 || static_cast<size_t>(face.texInfo) >= bsp.get_texinfos().size()) {
                return make_float3(0.5f, 0.5f, 0.5f);
            }
            const ::BSP::TexInfo& texInfo = bsp.get_texinfos()[face.texInfo];
            if (texInfo.texData < 0 || static_cast<size_t>(texInfo.texData) >= bsp.get_texdatas().size()) {
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
            return ((flags & ::BSP::SURF_BUMPLIGHT) != 0) && ((flags & ::BSP::SURF_NOLIGHT) == 0);
        }

        size_t sample_plane_count(const ::BSP::BSP& bsp, size_t faceIndex)
        {
            return surf_has_bumped_lightmaps(bsp, faceIndex) ? 4u : 1u;
        }

        void add_direct_to_radial(
            LuxelRadial& radial,
            const Geometry::FaceGeometry& targetGeometry,
            const ::BSP::DFace& targetFace,
            Common::Vec3f samplePos,
            Common::Vec2f coordMins,
            Common::Vec2f coordMaxs,
            float3 light
        )
        {
            Common::Vec2f coord;
            Geometry::world_to_luxel_space(targetGeometry, samplePos, coord, targetFace);

            int sMin = static_cast<int>(coordMins.x);
            int tMin = static_cast<int>(coordMins.y);
            int sMax = static_cast<int>(coordMaxs.x + 0.9999f) + 1;
            int tMax = static_cast<int>(coordMaxs.y + 0.9999f) + 1;

            sMin = std::max(sMin, 0);
            tMin = std::max(tMin, 0);
            sMax = std::min(sMax, static_cast<int>(radial.width));
            tMax = std::min(tMax, static_cast<int>(radial.height));

            for (int s = sMin; s < sMax; ++s) {
                for (int t = tMin; t < tMax; ++t) {
                    const float s0 = std::max(coordMins.x - static_cast<float>(s), -1.0f);
                    const float t0 = std::max(coordMins.y - static_cast<float>(t), -1.0f);
                    const float s1 = std::min(coordMaxs.x - static_cast<float>(s), 1.0f);
                    const float t1 = std::min(coordMaxs.y - static_cast<float>(t), 1.0f);
                    const float area = (s1 - s0) * (t1 - t0);
                    if (area <= WEIGHT_EPS) {
                        continue;
                    }

                    const float ds = std::fabs(coord.x - static_cast<float>(s));
                    const float dt = std::fabs(coord.y - static_cast<float>(t));
                    float r = std::max(ds, dt);
                    if (r < 0.1f) {
                        r = area / 0.1f;
                    }
                    else {
                        r = area / r;
                    }

                    const size_t index = static_cast<size_t>(s) + static_cast<size_t>(t) * radial.width;
                    radial.light[index] += light * r;
                    radial.weight[index] += r;
                }
            }
        }

        Common::Vec2f face_sample_mins_world_to_target_luxel(
            const Geometry::FaceGeometry& sourceGeometry,
            const Geometry::FaceGeometry& targetGeometry,
            const ::BSP::DFace& sourceFace,
            const ::BSP::DFace& targetFace,
            Common::Vec2f sourceCoord
        )
        {
            Common::Vec3f world = Geometry::luxel_space_to_world(
                sourceGeometry,
                sourceCoord.x,
                sourceCoord.y,
                sourceFace
            );
            Common::Vec2f out;
            Geometry::world_to_luxel_space(targetGeometry, world, out, targetFace);
            return out;
        }

        LuxelRadial build_luxel_radial(
            const ::BSP::BSP& bsp,
            const RuntimeState& state,
            const std::vector<float3>& lightSamples,
            const std::vector<std::vector<size_t>>& faceNeighbors,
            size_t faceIndex
        )
        {
            LuxelRadial radial;
            radial.faceIndex = faceIndex;
            if (faceIndex >= state.faceGeometry.size() || faceIndex >= bsp.get_dfaces().size()) {
                return radial;
            }

            const Geometry::FaceGeometry& targetGeometry = state.faceGeometry[faceIndex];
            const ::BSP::DFace& targetFace = bsp.get_dfaces()[faceIndex];
            if (!targetGeometry.valid || targetGeometry.isDisplacement || targetFace.lightOffset < 0) {
                return radial;
            }

            radial.width = targetGeometry.lightmap.width;
            radial.height = targetGeometry.lightmap.height;
            radial.light.assign(radial.width * radial.height, make_float3());
            radial.weight.assign(radial.width * radial.height, 0.0f);

            auto add_face_samples = [&](size_t sourceFaceIndex) {
                if (sourceFaceIndex >= state.faceGeometry.size() || sourceFaceIndex >= bsp.get_dfaces().size()) {
                    return;
                }

                const Geometry::FaceGeometry& sourceGeometry = state.faceGeometry[sourceFaceIndex];
                const ::BSP::DFace& sourceFace = bsp.get_dfaces()[sourceFaceIndex];
                if (!sourceGeometry.valid || sourceGeometry.isDisplacement || sourceFace.lightOffset < 0) {
                    return;
                }

                if (sourceGeometry.samples.values.empty()) {
                    return;
                }

                const size_t sourceLightmapStartIndex =
                    static_cast<size_t>(sourceFace.lightOffset / sizeof(::BSP::RGBExp32));
                for (size_t sampleIndex = 0; sampleIndex < sourceGeometry.samples.values.size(); ++sampleIndex) {
                    if (sourceLightmapStartIndex + sampleIndex >= lightSamples.size()) {
                        break;
                    }

                    const Common::ReceiverSample& sample = sourceGeometry.samples.values[sampleIndex];
                    const float3 light = lightSamples[sourceLightmapStartIndex + sampleIndex];
                    Common::Vec2f mins = face_sample_mins_world_to_target_luxel(
                        sourceGeometry,
                        targetGeometry,
                        sourceFace,
                        targetFace,
                        sample.mins
                    );
                    Common::Vec2f maxs = face_sample_mins_world_to_target_luxel(
                        sourceGeometry,
                        targetGeometry,
                        sourceFace,
                        targetFace,
                        sample.maxs
                    );

                    if (mins.x > maxs.x) {
                        std::swap(mins.x, maxs.x);
                    }
                    if (mins.y > maxs.y) {
                        std::swap(mins.y, maxs.y);
                    }

                    add_direct_to_radial(
                        radial,
                        targetGeometry,
                        targetFace,
                        sample.pos,
                        mins,
                        maxs,
                        light
                    );
                }
            };

            add_face_samples(faceIndex);
            if (faceIndex < faceNeighbors.size()) {
                for (size_t neighborFaceIndex : faceNeighbors[faceIndex]) {
                    add_face_samples(neighborFaceIndex);
                }
            }

            return radial;
        }

        float3 sample_radial(
            const LuxelRadial& radial,
            const Geometry::FaceGeometry& geometry,
            const ::BSP::DFace& face,
            Common::Vec3f hitPoint
        )
        {
            if (radial.width == 0 || radial.height == 0) {
                return make_float3();
            }

            Common::Vec2f coord;
            Geometry::world_to_luxel_space(geometry, hitPoint, coord, face);
            const int u = static_cast<int>(coord.x + 0.5f);
            const int v = static_cast<int>(coord.y + 0.5f);
            if (u < 0 || v < 0 || static_cast<size_t>(u) >= radial.width || static_cast<size_t>(v) >= radial.height) {
                return make_float3();
            }

            const size_t index = static_cast<size_t>(u) + static_cast<size_t>(v) * radial.width;
            if (index >= radial.weight.size() || radial.weight[index] <= WEIGHT_EPS) {
                return make_float3();
            }

            return radial.light[index] / radial.weight[index];
        }

        bool bounce_receiver_sample(
            const RuntimeState& state,
            size_t faceIndex,
            size_t s,
            size_t t,
            float3& outPos,
            float3& outNormal
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
                if (!Geometry::disp_uv_to_surf_point(dispGeometry, sample.coord.x, sample.coord.y, 1.0f, pos)) {
                    return false;
                }
                if (!Geometry::disp_uv_to_surf_normal(dispGeometry, sample.coord.x, sample.coord.y, normal)) {
                    normal = sample.normal;
                }
                outNormal = safe_normalized(to_float3(normal));
                outPos = to_float3(pos) + outNormal * HOST_RAY_BIAS;
                return is_finite(outPos) && is_finite(outNormal);
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
            outNormal = safe_normalized(to_float3(sample.normal));
            outPos = to_float3(sample.pos) + outNormal * HOST_RAY_BIAS;
            return is_finite(outPos) && is_finite(outNormal);
        }

        bool bounce_receiver_surface_sample(
            const RuntimeState& state,
            size_t faceIndex,
            size_t s,
            size_t t,
            float3& outPos,
            float3& outNormal
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
                if (!Geometry::disp_uv_to_surf_point(dispGeometry, sample.coord.x, sample.coord.y, 1.0f, pos)) {
                    return false;
                }
                if (!Geometry::disp_uv_to_surf_normal(dispGeometry, sample.coord.x, sample.coord.y, normal)) {
                    normal = sample.normal;
                }
                outPos = to_float3(pos);
                outNormal = safe_normalized(to_float3(normal));
                return is_finite(outPos) && is_finite(outNormal);
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
            outPos = to_float3(sample.pos);
            outNormal = safe_normalized(to_float3(sample.normal));
            return is_finite(outPos) && is_finite(outNormal);
        }

        std::vector<float3> denoise_bounce_by_receivers(
            const ::BSP::BSP& bsp,
            const RuntimeState& state,
            const std::vector<float3>& rawBounceByLightmap,
            const std::vector<float>& sampleWeightsByLightmap
        )
        {
            std::vector<float3> filtered = rawBounceByLightmap;
            const size_t numFaces = bsp.get_dfaces().size();
            const float invTwoSigmaSq = 1.0f / (2.0f * BOUNCE_DENOISE_SPATIAL_SIGMA * BOUNCE_DENOISE_SPATIAL_SIGMA);

            for (size_t faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
                const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
                if (face.lightOffset < 0) {
                    continue;
                }

                const Geometry::DispGeometry& dispGeometry = state.dispGeometry[faceIndex];
                const Geometry::FaceGeometry& faceGeometry = state.faceGeometry[faceIndex];
                const bool isDispReceiver = dispGeometry.valid;
                const bool isFaceReceiver = faceGeometry.valid && !faceGeometry.isDisplacement;
                if (!isDispReceiver && !isFaceReceiver) {
                    continue;
                }

                const size_t width = static_cast<size_t>(face.lightmapTextureSizeInLuxels[0] + 1);
                const size_t height = static_cast<size_t>(face.lightmapTextureSizeInLuxels[1] + 1);
                const size_t lightmapStartIndex =
                    static_cast<size_t>(face.lightOffset / sizeof(::BSP::RGBExp32));

                for (size_t t = 0; t < height; ++t) {
                    for (size_t s = 0; s < width; ++s) {
                        const size_t sampleIndex = t * width + s;
                        const size_t lightmapIndex = lightmapStartIndex + sampleIndex;
                        if (lightmapIndex >= rawBounceByLightmap.size()
                            || sampleWeightsByLightmap[lightmapIndex] <= WEIGHT_EPS) {
                            continue;
                        }

                        float3 centerPos;
                        float3 centerNormal;
                        if (!bounce_receiver_surface_sample(state, faceIndex, s, t, centerPos, centerNormal)) {
                            continue;
                        }

                        float3 accum = make_float3();
                        float totalWeight = 0.0f;
                        const int sInt = static_cast<int>(s);
                        const int tInt = static_cast<int>(t);

                        for (int dt = -BOUNCE_DENOISE_RADIUS; dt <= BOUNCE_DENOISE_RADIUS; ++dt) {
                            for (int ds = -BOUNCE_DENOISE_RADIUS; ds <= BOUNCE_DENOISE_RADIUS; ++ds) {
                                const int ns = sInt + ds;
                                const int nt = tInt + dt;
                                if (ns < 0 || nt < 0
                                    || static_cast<size_t>(ns) >= width
                                    || static_cast<size_t>(nt) >= height) {
                                    continue;
                                }

                                const size_t neighborSampleIndex =
                                    static_cast<size_t>(nt) * width + static_cast<size_t>(ns);
                                const size_t neighborLightmapIndex = lightmapStartIndex + neighborSampleIndex;
                                if (neighborLightmapIndex >= rawBounceByLightmap.size()
                                    || sampleWeightsByLightmap[neighborLightmapIndex] <= WEIGHT_EPS) {
                                    continue;
                                }

                                float3 neighborPos;
                                float3 neighborNormal;
                                if (!bounce_receiver_surface_sample(
                                        state,
                                        faceIndex,
                                        static_cast<size_t>(ns),
                                        static_cast<size_t>(nt),
                                        neighborPos,
                                        neighborNormal
                                    )) {
                                    continue;
                                }

                                const float normalDot = dot(centerNormal, neighborNormal);
                                if (normalDot < BOUNCE_DENOISE_NORMAL_DOT_MIN) {
                                    continue;
                                }

                                const float3 delta = neighborPos - centerPos;
                                const float worldDistance = len(delta);
                                if (!std::isfinite(worldDistance) || worldDistance > BOUNCE_DENOISE_WORLD_DISTANCE) {
                                    continue;
                                }

                                const float spatialDistanceSq =
                                    static_cast<float>(ds * ds + dt * dt);
                                const float spatialWeight = std::exp(-spatialDistanceSq * invTwoSigmaSq);
                                const float normalWeight =
                                    std::max(0.0f, (normalDot - BOUNCE_DENOISE_NORMAL_DOT_MIN)
                                        / (1.0f - BOUNCE_DENOISE_NORMAL_DOT_MIN + 1e-6f));
                                const float weight = spatialWeight * std::max(normalWeight, 0.05f);
                                accum += rawBounceByLightmap[neighborLightmapIndex] * weight;
                                totalWeight += weight;
                            }
                        }

                        if (totalWeight > WEIGHT_EPS) {
                            filtered[lightmapIndex] = accum / totalWeight;
                        }
                    }
                }
            }

            return filtered;
        }

        float3 compute_lightmap_color_from_average(
            const ::BSP::BSP& bsp,
            const std::vector<float3>& lightSamples,
            size_t faceIndex
        )
        {
            if (faceIndex >= bsp.get_dfaces().size() || face_is_sky(bsp, faceIndex)) {
                return make_float3();
            }
            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.lightOffset < 0) {
                return make_float3();
            }
            const size_t lightmapStartIndex = static_cast<size_t>(face.lightOffset / sizeof(::BSP::RGBExp32));
            if (lightmapStartIndex == 0 || lightmapStartIndex > lightSamples.size()) {
                return make_float3();
            }
            return component_multiply(lightSamples[lightmapStartIndex - 1], surface_reflectivity(bsp, faceIndex));
        }

        float3 compute_lightmap_color_face_radial(
            const ::BSP::BSP& bsp,
            const RuntimeState& state,
            const std::vector<float3>& lightSamples,
            const std::vector<LuxelRadial>& radials,
            size_t faceIndex,
            Common::Vec3f hitPoint
        )
        {
            if (faceIndex >= state.faceGeometry.size() || face_is_sky(bsp, faceIndex)) {
                return make_float3();
            }

            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            const Geometry::FaceGeometry& geometry = state.faceGeometry[faceIndex];
            if (!geometry.valid || geometry.isDisplacement) {
                return compute_lightmap_color_from_average(bsp, lightSamples, faceIndex);
            }

            if (faceIndex >= radials.size()) {
                return compute_lightmap_color_from_average(bsp, lightSamples, faceIndex);
            }

            const float3 sampled = sample_radial(radials[faceIndex], geometry, face, hitPoint);
            if (!is_finite(sampled) || len(sampled) <= 1e-6f) {
                return compute_lightmap_color_from_average(bsp, lightSamples, faceIndex);
            }

            return component_multiply(sampled, surface_reflectivity(bsp, faceIndex));
        }

        float3 compute_lightmap_color_displacement(
            const ::BSP::BSP& bsp,
            const RuntimeState& state,
            const std::vector<float3>& lightSamples,
            size_t faceIndex,
            Common::Vec3f hitPoint
        )
        {
            if (face_is_sky(bsp, faceIndex) || faceIndex >= state.dispGeometry.size()) {
                return make_float3();
            }

            const Geometry::DispGeometry& geometry = state.dispGeometry[faceIndex];
            if (!geometry.valid) {
                return compute_lightmap_color_from_average(bsp, lightSamples, faceIndex);
            }

            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.lightOffset < 0) {
                return make_float3();
            }

            Common::Vec2f uv;
            if (!Geometry::disp_world_to_uv(geometry, hitPoint, uv)) {
                return compute_lightmap_color_from_average(bsp, lightSamples, faceIndex);
            }

            const int smax = face.lightmapTextureSizeInLuxels[0] + 1;
            const int tmax = face.lightmapTextureSizeInLuxels[1] + 1;
            const int ds = std::max(0, std::min(static_cast<int>(uv.x * static_cast<float>(smax - 1)), smax - 1));
            const int dt = std::max(0, std::min(static_cast<int>(uv.y * static_cast<float>(tmax - 1)), tmax - 1));
            const size_t base = static_cast<size_t>(face.lightOffset / sizeof(::BSP::RGBExp32));
            const size_t sampleIndex = base + static_cast<size_t>(dt * smax + ds);
            if (sampleIndex >= lightSamples.size()) {
                return make_float3();
            }

            return component_multiply(lightSamples[sampleIndex], surface_reflectivity(bsp, faceIndex));
        }

        float3 compute_bounce_color_from_hit(
            const ::BSP::BSP& bsp,
            const RuntimeState& state,
            const std::vector<float3>& lightSamples,
            const std::vector<LuxelRadial>& radials,
            const std::vector<OptixRT::Triangle>& triangles,
            const OptixRT::SunRay& ray,
            const OptixRT::RayHit& hit
        )
        {
            if (!hit.hit || hit.primitiveIndex >= triangles.size()) {
                return make_float3();
            }

            const Tracing::SurfaceTraceResult surfaceTrace = Tracing::test_line_surface(hit, triangles);
            if (surfaceTrace.hitSky
                || surfaceTrace.sourceId == INVALID_SOURCE_FACE
                || surfaceTrace.sourceId >= bsp.get_dfaces().size()) {
                return make_float3();
            }

            const size_t faceIndex = surfaceTrace.sourceId;
            const Common::Vec3f hitPoint = from_float3(ray.origin + ray.direction * hit.t);
            if (faceIndex < state.dispGeometry.size() && state.dispGeometry[faceIndex].valid) {
                return compute_lightmap_color_displacement(bsp, state, lightSamples, faceIndex, hitPoint);
            }
            return compute_lightmap_color_face_radial(
                bsp,
                state,
                lightSamples,
                radials,
                faceIndex,
                hitPoint
            );
        }
    }

    void compute_bounce_lighting_runtime(
        ::BSP::BSP& bsp,
        ::CUDABSP::CUDABSP* pCudaBSP,
        RuntimeState& state,
        const std::vector<OptixRT::Triangle>& triangles,
        OptixRT::OptixSunLosTracer& tracer
    )
    {
        ::CUDABSP::CUDABSP cudaBSP;
        CUDA_CHECK_ERROR(cudaMemcpy(&cudaBSP, pCudaBSP, sizeof(::CUDABSP::CUDABSP), cudaMemcpyDeviceToHost));

        std::vector<float3> sourceLightSamples(bsp.get_lightsamples().size(), make_float3());
        if (!sourceLightSamples.empty()) {
            CUDA_CHECK_ERROR(cudaMemcpy(
                sourceLightSamples.data(),
                cudaBSP.lightSamples,
                sizeof(float3) * sourceLightSamples.size(),
                cudaMemcpyDeviceToHost
            ));
        }

        std::vector<float3> bouncedLightSamples = sourceLightSamples;
        const std::vector<std::vector<size_t>> faceNeighbors = build_face_neighbors(bsp);
        std::vector<LuxelRadial> faceRadials(bsp.get_faces().size());
        for (size_t faceIndex = 0; faceIndex < faceRadials.size(); ++faceIndex) {
            faceRadials[faceIndex] = build_luxel_radial(
                bsp,
                state,
                sourceLightSamples,
                faceNeighbors,
                faceIndex
            );
        }

        size_t totalStaticPropVertices = 0;
        for (const Common::StaticPropLightingProp& prop : state.scene.staticPropLightingProps) {
            for (const Common::StaticPropLightingMesh& mesh : prop.meshes) {
                totalStaticPropVertices += mesh.vertices.size();
            }
        }
        std::vector<float3> staticPropBounce(totalStaticPropVertices, make_float3());
        std::vector<OptixRT::SunRay> rays;
        std::vector<BounceContribution> contributions;
        rays.reserve(bsp.get_lightsamples().size() * 8);
        contributions.reserve(bsp.get_lightsamples().size() * 8);

        const size_t numFaces = bsp.get_faces().size();
        for (size_t faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.lightOffset < 0) {
                continue;
            }

            const Geometry::DispGeometry& dispGeometry = state.dispGeometry[faceIndex];
            const Geometry::FaceGeometry& faceGeometry = state.faceGeometry[faceIndex];
            const bool isDispReceiver = dispGeometry.valid;
            const bool isFaceReceiver = faceGeometry.valid && !faceGeometry.isDisplacement;
            if (!isDispReceiver && !isFaceReceiver) {
                continue;
            }

            const size_t lightmapWidth = static_cast<size_t>(face.lightmapTextureSizeInLuxels[0] + 1);
            const size_t lightmapHeight = static_cast<size_t>(face.lightmapTextureSizeInLuxels[1] + 1);
            const size_t lightmapSize = lightmapWidth * lightmapHeight;
            const size_t lightmapStartIndex = static_cast<size_t>(face.lightOffset / sizeof(::BSP::RGBExp32));
            const size_t planeCount = sample_plane_count(bsp, faceIndex);

            for (size_t t = 0; t < lightmapHeight; ++t) {
                for (size_t s = 0; s < lightmapWidth; ++s) {
                    float3 samplePos;
                    float3 sampleNormal;
                    if (!bounce_receiver_sample(state, faceIndex, s, t, samplePos, sampleNormal)) {
                        continue;
                    }

                    const size_t sampleIndex = t * lightmapWidth + s;
                    const size_t lightmapIndex = lightmapStartIndex + sampleIndex;

                    for (size_t dirIndex = 0; dirIndex < NUM_VERTEX_NORMALS; dirIndex += BOUNCE_DIRECTION_STRIDE) {
                        const float3 direction = ANORMS[dirIndex];
                        const float weight = dot(sampleNormal, direction);
                        if (weight <= 1e-6f) {
                            continue;
                        }

                        OptixRT::SunRay ray{};
                        ray.origin = samplePos;
                        ray.direction = direction;
                        ray.tmin = HOST_RAY_TMIN;
                        ray.tmax = BOUNCE_TRACE_DISTANCE;
                        ray.visibilityMask = 0xff;
                        if (!is_valid_ray(ray)) {
                            continue;
                        }

                        rays.push_back(ray);
                        contributions.push_back(BounceContribution{
                            false,
                            lightmapIndex,
                            lightmapSize,
                            planeCount,
                            weight,
                        });
                    }
                }
            }
        }

        size_t staticPropVertexIndex = 0;
        for (const Common::StaticPropLightingProp& prop : state.scene.staticPropLightingProps) {
            for (const Common::StaticPropLightingMesh& mesh : prop.meshes) {
                for (const Common::StaticPropLightingVertex& vertex : mesh.vertices) {
                    const float3 sampleNormal = safe_normalized(to_float3(vertex.normal));
                    const float3 samplePos = to_float3(vertex.pos) + sampleNormal * HOST_RAY_BIAS;
                    if (!is_finite(samplePos) || !is_finite(sampleNormal)) {
                        ++staticPropVertexIndex;
                        continue;
                    }

                    for (size_t dirIndex = 0; dirIndex < NUM_VERTEX_NORMALS; dirIndex += BOUNCE_DIRECTION_STRIDE) {
                        const float3 direction = ANORMS[dirIndex];
                        const float weight = dot(sampleNormal, direction);
                        if (weight <= 1e-6f) {
                            continue;
                        }

                        OptixRT::SunRay ray{};
                        ray.origin = samplePos;
                        ray.direction = direction;
                        ray.tmin = HOST_RAY_TMIN;
                        ray.tmax = BOUNCE_TRACE_DISTANCE;
                        ray.visibilityMask = 0xff;
                        if (!is_valid_ray(ray)) {
                            continue;
                        }

                        rays.push_back(ray);
                        contributions.push_back(BounceContribution{
                            true,
                            staticPropVertexIndex,
                            0,
                            1,
                            weight,
                        });
                    }

                    ++staticPropVertexIndex;
                }
            }
        }

        std::cout << "Tracing " << rays.size() << " core bounce rays..." << std::endl;

        std::vector<OptixRT::RayHit> hits;
        tracer.trace_batch(rays, hits);

        std::vector<float3> sampleBounceByLightmap(bouncedLightSamples.size(), make_float3());
        std::vector<float> sampleWeightsByLightmap(bouncedLightSamples.size(), 0.0f);
        std::vector<float> staticPropBounceWeights(staticPropBounce.size(), 0.0f);

        for (size_t i = 0; i < hits.size(); ++i) {
            const BounceContribution& contribution = contributions[i];
            const float3 bounceColor = compute_bounce_color_from_hit(
                bsp,
                state,
                sourceLightSamples,
                faceRadials,
                triangles,
                rays[i],
                hits[i]
            );
            if (!is_finite(bounceColor)) {
                continue;
            }

            if (contribution.staticPropVertex) {
                staticPropBounce[contribution.outputIndex] += bounceColor * contribution.weight;
                staticPropBounceWeights[contribution.outputIndex] += contribution.weight;
            }
            else {
                sampleBounceByLightmap[contribution.outputIndex] += bounceColor * contribution.weight;
                sampleWeightsByLightmap[contribution.outputIndex] += contribution.weight;
            }
        }

        const std::vector<float3> denoisedSampleBounceByLightmap =
            denoise_bounce_by_receivers(
                bsp,
                state,
                sampleBounceByLightmap,
                sampleWeightsByLightmap
            );

        for (size_t faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.lightOffset < 0) {
                continue;
            }

            const Geometry::DispGeometry& dispGeometry = state.dispGeometry[faceIndex];
            const Geometry::FaceGeometry& faceGeometry = state.faceGeometry[faceIndex];
            const bool isDispReceiver = dispGeometry.valid;
            const bool isFaceReceiver = faceGeometry.valid && !faceGeometry.isDisplacement;
            if (!isDispReceiver && !isFaceReceiver) {
                continue;
            }

            const size_t lightmapWidth = static_cast<size_t>(face.lightmapTextureSizeInLuxels[0] + 1);
            const size_t lightmapHeight = static_cast<size_t>(face.lightmapTextureSizeInLuxels[1] + 1);
            const size_t lightmapSize = lightmapWidth * lightmapHeight;
            const size_t lightmapStartIndex = static_cast<size_t>(face.lightOffset / sizeof(::BSP::RGBExp32));
            const size_t planeCount = sample_plane_count(bsp, faceIndex);

            for (size_t sampleIndex = 0; sampleIndex < lightmapSize; ++sampleIndex) {
                const size_t lightmapIndex = lightmapStartIndex + sampleIndex;
                const float weight = sampleWeightsByLightmap[lightmapIndex];
                if (weight <= 1e-6f) {
                    continue;
                }

                const float3 bounceColor = denoisedSampleBounceByLightmap[lightmapIndex];
                for (size_t planeIndex = 0; planeIndex < planeCount; ++planeIndex) {
                    bouncedLightSamples[lightmapIndex + planeIndex * lightmapSize] += bounceColor;
                }
            }

            if (lightmapStartIndex > 0 && lightmapSize > 0) {
                float3 totalLight = make_float3();
                for (size_t sampleIndex = 0; sampleIndex < lightmapSize; ++sampleIndex) {
                    totalLight += bouncedLightSamples[lightmapStartIndex + sampleIndex];
                }
                bouncedLightSamples[lightmapStartIndex - 1] =
                    totalLight / static_cast<float>(lightmapSize);
            }
        }

        if (!bouncedLightSamples.empty()) {
            CUDA_CHECK_ERROR(cudaMemcpy(
                cudaBSP.lightSamples,
                bouncedLightSamples.data(),
                sizeof(float3) * bouncedLightSamples.size(),
                cudaMemcpyHostToDevice
            ));
        }

        staticPropVertexIndex = 0;
        for (Common::StaticPropLightingProp& prop : state.scene.staticPropLightingProps) {
            for (Common::StaticPropLightingMesh& mesh : prop.meshes) {
                for (size_t vertexIndex = 0; vertexIndex < mesh.vertices.size(); ++vertexIndex) {
                    if (staticPropVertexIndex >= staticPropBounce.size()
                        || vertexIndex >= mesh.colors.size()) {
                        break;
                    }

                    const float weight = staticPropBounceWeights[staticPropVertexIndex];
                    if (weight > 1e-6f) {
                        const float3 bounceColor = staticPropBounce[staticPropVertexIndex] / weight;
                        mesh.colors[vertexIndex] = Common::make_vec3(
                            mesh.colors[vertexIndex].x + bounceColor.x,
                            mesh.colors[vertexIndex].y + bounceColor.y,
                            mesh.colors[vertexIndex].z + bounceColor.z
                        );
                    }

                    ++staticPropVertexIndex;
                }
            }
        }

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }
}
