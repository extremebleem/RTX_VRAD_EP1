#include "ambient_lighting.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "../../cudautils.h"
#include "../state.h"
#include "../common/math.h"
#include "../tracing/trace.h"

namespace SilkRAD::Core::Lighting {
    namespace {
        static constexpr size_t NUM_VERTEX_NORMALS = 162;
        static constexpr float HOST_RAY_TMIN = 0.001f;
        static constexpr float COORD_EXTENT = 16384.0f;
        static constexpr float AMBIENT_TRACE_DISTANCE = COORD_EXTENT * 1.74f;
        static constexpr float WORLD_LIGHT_MIN_EMIT_SURFACE = 0.005f;
        static constexpr uint32_t RTX_ROLE_SKY = 1u << 2;
        static constexpr uint32_t INVALID_SOURCE_FACE = 0xffffffffu;

        static const float3 BOX_DIRECTIONS[6] = {
            { 1.0f, 0.0f, 0.0f },
            { -1.0f, 0.0f, 0.0f },
            { 0.0f, 1.0f, 0.0f },
            { 0.0f, -1.0f, 0.0f },
            { 0.0f, 0.0f, 1.0f },
            { 0.0f, 0.0f, -1.0f },
        };

        static const float3 ANORMS[NUM_VERTEX_NORMALS] = {
            { -0.525731f, 0.000000f, 0.850651f },
            { -0.442863f, 0.238856f, 0.864188f },
            { -0.295242f, 0.000000f, 0.955423f },
            { -0.309017f, 0.500000f, 0.809017f },
            { -0.162460f, 0.262866f, 0.951056f },
            { 0.000000f, 0.000000f, 1.000000f },
            { 0.000000f, 0.850651f, 0.525731f },
            { -0.147621f, 0.716567f, 0.681718f },
            { 0.147621f, 0.716567f, 0.681718f },
            { 0.000000f, 0.525731f, 0.850651f },
            { 0.309017f, 0.500000f, 0.809017f },
            { 0.525731f, 0.000000f, 0.850651f },
            { 0.295242f, 0.000000f, 0.955423f },
            { 0.442863f, 0.238856f, 0.864188f },
            { 0.162460f, 0.262866f, 0.951056f },
            { -0.681718f, 0.147621f, 0.716567f },
            { -0.809017f, 0.309017f, 0.500000f },
            { -0.587785f, 0.425325f, 0.688191f },
            { -0.850651f, 0.525731f, 0.000000f },
            { -0.864188f, 0.442863f, 0.238856f },
            { -0.716567f, 0.681718f, 0.147621f },
            { -0.688191f, 0.587785f, 0.425325f },
            { -0.500000f, 0.809017f, 0.309017f },
            { -0.238856f, 0.864188f, 0.442863f },
            { -0.425325f, 0.688191f, 0.587785f },
            { -0.716567f, 0.681718f, -0.147621f },
            { -0.500000f, 0.809017f, -0.309017f },
            { -0.525731f, 0.850651f, 0.000000f },
            { 0.000000f, 0.850651f, -0.525731f },
            { -0.238856f, 0.864188f, -0.442863f },
            { 0.000000f, 0.955423f, -0.295242f },
            { -0.262866f, 0.951056f, -0.162460f },
            { 0.000000f, 1.000000f, 0.000000f },
            { 0.000000f, 0.955423f, 0.295242f },
            { -0.262866f, 0.951056f, 0.162460f },
            { 0.238856f, 0.864188f, 0.442863f },
            { 0.262866f, 0.951056f, 0.162460f },
            { 0.500000f, 0.809017f, 0.309017f },
            { 0.238856f, 0.864188f, -0.442863f },
            { 0.262866f, 0.951056f, -0.162460f },
            { 0.500000f, 0.809017f, -0.309017f },
            { 0.850651f, 0.525731f, 0.000000f },
            { 0.716567f, 0.681718f, 0.147621f },
            { 0.716567f, 0.681718f, -0.147621f },
            { 0.525731f, 0.850651f, 0.000000f },
            { 0.425325f, 0.688191f, 0.587785f },
            { 0.864188f, 0.442863f, 0.238856f },
            { 0.688191f, 0.587785f, 0.425325f },
            { 0.809017f, 0.309017f, 0.500000f },
            { 0.681718f, 0.147621f, 0.716567f },
            { 0.587785f, 0.425325f, 0.688191f },
            { 0.955423f, 0.295242f, 0.000000f },
            { 1.000000f, 0.000000f, 0.000000f },
            { 0.951056f, 0.162460f, 0.262866f },
            { 0.850651f, -0.525731f, 0.000000f },
            { 0.955423f, -0.295242f, 0.000000f },
            { 0.864188f, -0.442863f, 0.238856f },
            { 0.951056f, -0.162460f, 0.262866f },
            { 0.809017f, -0.309017f, 0.500000f },
            { 0.681718f, -0.147621f, 0.716567f },
            { 0.850651f, 0.000000f, 0.525731f },
            { 0.864188f, 0.442863f, -0.238856f },
            { 0.809017f, 0.309017f, -0.500000f },
            { 0.951056f, 0.162460f, -0.262866f },
            { 0.525731f, 0.000000f, -0.850651f },
            { 0.681718f, 0.147621f, -0.716567f },
            { 0.681718f, -0.147621f, -0.716567f },
            { 0.850651f, 0.000000f, -0.525731f },
            { 0.809017f, -0.309017f, -0.500000f },
            { 0.864188f, -0.442863f, -0.238856f },
            { 0.951056f, -0.162460f, -0.262866f },
            { 0.147621f, 0.716567f, -0.681718f },
            { 0.309017f, 0.500000f, -0.809017f },
            { 0.425325f, 0.688191f, -0.587785f },
            { 0.442863f, 0.238856f, -0.864188f },
            { 0.587785f, 0.425325f, -0.688191f },
            { 0.688191f, 0.587785f, -0.425325f },
            { -0.147621f, 0.716567f, -0.681718f },
            { -0.309017f, 0.500000f, -0.809017f },
            { 0.000000f, 0.525731f, -0.850651f },
            { -0.525731f, 0.000000f, -0.850651f },
            { -0.442863f, 0.238856f, -0.864188f },
            { -0.295242f, 0.000000f, -0.955423f },
            { -0.162460f, 0.262866f, -0.951056f },
            { 0.000000f, 0.000000f, -1.000000f },
            { 0.295242f, 0.000000f, -0.955423f },
            { 0.162460f, 0.262866f, -0.951056f },
            { -0.442863f, -0.238856f, -0.864188f },
            { -0.309017f, -0.500000f, -0.809017f },
            { -0.162460f, -0.262866f, -0.951056f },
            { 0.000000f, -0.850651f, -0.525731f },
            { -0.147621f, -0.716567f, -0.681718f },
            { 0.147621f, -0.716567f, -0.681718f },
            { 0.000000f, -0.525731f, -0.850651f },
            { 0.309017f, -0.500000f, -0.809017f },
            { 0.442863f, -0.238856f, -0.864188f },
            { 0.162460f, -0.262866f, -0.951056f },
            { 0.238856f, -0.864188f, -0.442863f },
            { 0.500000f, -0.809017f, -0.309017f },
            { 0.425325f, -0.688191f, -0.587785f },
            { 0.716567f, -0.681718f, -0.147621f },
            { 0.688191f, -0.587785f, -0.425325f },
            { 0.587785f, -0.425325f, -0.688191f },
            { 0.000000f, -0.955423f, -0.295242f },
            { 0.000000f, -1.000000f, 0.000000f },
            { 0.262866f, -0.951056f, -0.162460f },
            { 0.000000f, -0.850651f, 0.525731f },
            { 0.000000f, -0.955423f, 0.295242f },
            { 0.238856f, -0.864188f, 0.442863f },
            { 0.262866f, -0.951056f, 0.162460f },
            { 0.500000f, -0.809017f, 0.309017f },
            { 0.716567f, -0.681718f, 0.147621f },
            { 0.525731f, -0.850651f, 0.000000f },
            { -0.238856f, -0.864188f, -0.442863f },
            { -0.500000f, -0.809017f, -0.309017f },
            { -0.262866f, -0.951056f, -0.162460f },
            { -0.850651f, -0.525731f, 0.000000f },
            { -0.716567f, -0.681718f, -0.147621f },
            { -0.716567f, -0.681718f, 0.147621f },
            { -0.525731f, -0.850651f, 0.000000f },
            { -0.500000f, -0.809017f, 0.309017f },
            { -0.238856f, -0.864188f, 0.442863f },
            { -0.262866f, -0.951056f, 0.162460f },
            { -0.864188f, -0.442863f, 0.238856f },
            { -0.809017f, -0.309017f, 0.500000f },
            { -0.688191f, -0.587785f, 0.425325f },
            { -0.681718f, -0.147621f, 0.716567f },
            { -0.442863f, -0.238856f, 0.864188f },
            { -0.587785f, -0.425325f, 0.688191f },
            { -0.309017f, -0.500000f, 0.809017f },
            { -0.147621f, -0.716567f, 0.681718f },
            { -0.425325f, -0.688191f, 0.587785f },
            { -0.162460f, -0.262866f, 0.951056f },
            { 0.442863f, -0.238856f, 0.864188f },
            { 0.162460f, -0.262866f, 0.951056f },
            { 0.309017f, -0.500000f, 0.809017f },
            { 0.147621f, -0.716567f, 0.681718f },
            { 0.000000f, -0.525731f, 0.850651f },
            { 0.425325f, -0.688191f, 0.587785f },
            { 0.587785f, -0.425325f, 0.688191f },
            { 0.688191f, -0.587785f, 0.425325f },
            { -0.955423f, 0.295242f, 0.000000f },
            { -0.951056f, 0.162460f, 0.262866f },
            { -1.000000f, 0.000000f, 0.000000f },
            { -0.850651f, 0.000000f, 0.525731f },
            { -0.955423f, -0.295242f, 0.000000f },
            { -0.951056f, -0.162460f, 0.262866f },
            { -0.864188f, 0.442863f, -0.238856f },
            { -0.951056f, 0.162460f, -0.262866f },
            { -0.809017f, 0.309017f, -0.500000f },
            { -0.864188f, -0.442863f, -0.238856f },
            { -0.951056f, -0.162460f, -0.262866f },
            { -0.809017f, -0.309017f, -0.500000f },
            { -0.681718f, 0.147621f, -0.716567f },
            { -0.681718f, -0.147621f, -0.716567f },
            { -0.850651f, 0.000000f, -0.525731f },
            { -0.688191f, 0.587785f, -0.425325f },
            { -0.587785f, 0.425325f, -0.688191f },
            { -0.425325f, 0.688191f, -0.587785f },
            { -0.425325f, -0.688191f, -0.587785f },
            { -0.587785f, -0.425325f, -0.688191f },
            { -0.688191f, -0.587785f, -0.425325f },
        };

        struct AmbientRayContribution {
            size_t leafIndex = 0;
            size_t dirIndex = 0;
        };

        struct EmitSurfaceContribution {
            size_t leafIndex = 0;
            size_t lightIndex = 0;
        };

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

        Common::Vec3f from_float3(float3 v)
        {
            return Common::make_vec3(v.x, v.y, v.z);
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

        float3 rgbexp32_to_linear_float3(const ::BSP::RGBExp32& sample)
        {
            const float scale = std::ldexp(1.0f, static_cast<int>(sample.exp));
            return make_float3(
                static_cast<float>(sample.r) * scale,
                static_cast<float>(sample.g) * scale,
                static_cast<float>(sample.b) * scale
            );
        }

        float3 find_sky_ambient(const ::BSP::BSP& bsp)
        {
            for (const ::BSP::DWorldLight& light : bsp.get_worldlights()) {
                if (light.type == ::BSP::EMIT_SKYAMBIENT && light.style == 0) {
                    return make_float3(light.intensity);
                }
            }

            return make_float3();
        }

        float3 surface_reflectivity(const ::BSP::BSP& bsp, size_t faceIndex)
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

        bool face_is_sky(const ::BSP::BSP& bsp, size_t faceIndex)
        {
            const ::BSP::Face& face = bsp.get_faces()[faceIndex];
            return (face.get_texinfo().flags & ::BSP::SURF_SKY) != 0;
        }

        bool surf_has_bumped_lightmaps(const ::BSP::BSP& bsp, size_t faceIndex)
        {
            const ::BSP::Face& face = bsp.get_faces()[faceIndex];
            const int32_t flags = face.get_texinfo().flags;
            return ((flags & ::BSP::SURF_BUMPLIGHT) != 0)
                && ((flags & ::BSP::SURF_NOLIGHT) == 0);
        }

        float3 compute_lightmap_color_from_average(
            const ::BSP::BSP& bsp,
            size_t faceIndex,
            float3 skyAmbient
        )
        {
            if (faceIndex >= bsp.get_dfaces().size()) {
                return make_float3();
            }

            if (face_is_sky(bsp, faceIndex)) {
                return skyAmbient;
            }

            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.lightOffset < 0) {
                return make_float3();
            }

            const size_t lightmapStartIndex =
                static_cast<size_t>(face.lightOffset / sizeof(::BSP::RGBExp32));
            if (lightmapStartIndex == 0 || lightmapStartIndex > bsp.get_lightsamples().size()) {
                return make_float3();
            }

            const float3 avgLight = rgbexp32_to_linear_float3(
                bsp.get_lightsamples()[lightmapStartIndex - 1]
            );
            return component_multiply(avgLight, surface_reflectivity(bsp, faceIndex));
        }

        float3 compute_lightmap_color_displacement(
            const ::BSP::BSP& bsp,
            const RuntimeState& state,
            const std::vector<float3>& lightSamples,
            size_t faceIndex,
            Common::Vec3f hitPoint,
            float3 skyAmbient
        )
        {
            if (face_is_sky(bsp, faceIndex)) {
                return skyAmbient;
            }

            if (faceIndex >= state.dispGeometry.size()) {
                return make_float3();
            }

            const Geometry::DispGeometry& geometry = state.dispGeometry[faceIndex];
            if (!geometry.valid) {
                return compute_lightmap_color_from_average(
                    bsp,
                    faceIndex,
                    skyAmbient
                );
            }

            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.lightOffset < 0) {
                return make_float3();
            }

            Common::Vec2f uv;
            if (!Geometry::disp_world_to_uv(geometry, hitPoint, uv)) {
                float bestDist2 = std::numeric_limits<float>::max();
                size_t bestIndex = 0;
                for (size_t i = 0; i < geometry.samples.size(); ++i) {
                    const Common::Vec3f delta = Common::sub(geometry.samples[i].pos, hitPoint);
                    const float dist2 = Common::dot(delta, delta);
                    if (dist2 < bestDist2) {
                        bestDist2 = dist2;
                        bestIndex = i;
                    }
                }

                if (bestIndex < geometry.samples.size()) {
                    uv = geometry.samples[bestIndex].coord;
                }
                else {
                    return compute_lightmap_color_from_average(
                        bsp,
                        faceIndex,
                        skyAmbient
                    );
                }
            }

            const int smax = face.lightmapTextureSizeInLuxels[0] + 1;
            const int tmax = face.lightmapTextureSizeInLuxels[1] + 1;
            const int ds = std::max(0, std::min(static_cast<int>(uv.x * static_cast<float>(smax - 1)), smax - 1));
            const int dt = std::max(0, std::min(static_cast<int>(uv.y * static_cast<float>(tmax - 1)), tmax - 1));

            int stylePlaneSize = smax * tmax;
            if (surf_has_bumped_lightmaps(bsp, faceIndex)) {
                stylePlaneSize *= 4;
            }

            const size_t base = static_cast<size_t>(face.lightOffset / sizeof(::BSP::RGBExp32));
            const size_t sampleIndex = base + static_cast<size_t>(dt * smax + ds);
            if (sampleIndex >= lightSamples.size()) {
                return make_float3();
            }

            return component_multiply(lightSamples[sampleIndex], surface_reflectivity(bsp, faceIndex));
        }

        float inv_r_squared(float3 delta)
        {
            const float distSquared = dot(delta, delta);
            if (distSquared <= 1e-20f) {
                return 0.0f;
            }

            return 1.0f / distSquared;
        }

        bool is_leaf_ambient_surface_light(const ::BSP::DWorldLight& light)
        {
            if (light.type != ::BSP::EMIT_SURFACE || light.style != 0) {
                return false;
            }

            const float intensity = std::max(light.intensity.x, std::max(light.intensity.y, light.intensity.z));
            return (intensity * inv_r_squared(make_float3(0.0f, 0.0f, 512.0f))) < WORLD_LIGHT_MIN_EMIT_SURFACE;
        }

        float engine_world_light_distance_falloff(
            const ::BSP::DWorldLight& light,
            float3 delta
        )
        {
            if (light.radius != 0.0f && dot(delta, delta) > light.radius * light.radius) {
                return 0.0f;
            }

            return inv_r_squared(delta);
        }

        float engine_world_light_angle(
            const ::BSP::DWorldLight& light,
            float3 deltaNormal
        )
        {
            const float dotSample = dot(deltaNormal, deltaNormal);
            if (dotSample < 0.0f) {
                return 0.0f;
            }

            const float dotLight = -dot(deltaNormal, make_float3(light.normal));
            if (dotLight <= 0.01f) {
                return 0.0f;
            }

            return dotSample * dotLight;
        }

        void build_ambient_world(
            const RuntimeState& state,
            std::vector<OptixRT::Triangle>& ambientTriangles
        )
        {
            ambientTriangles.clear();
            ambientTriangles.reserve(
                state.scene.worldFaceTriangles.size()
                + state.scene.worldBrushTriangles.size()
                + state.scene.displacementTriangles.size()
            );

            auto append = [&ambientTriangles](const std::vector<Common::OccluderTriangle>& input) {
                for (const Common::OccluderTriangle& triIn : input) {
                    OptixRT::Triangle tri{};
                    tri.v0 = to_float3(triIn.v0);
                    tri.v1 = to_float3(triIn.v1);
                    tri.v2 = to_float3(triIn.v2);
                    tri.sourceId = triIn.sourceId;
                    tri.role = triIn.role;
                    tri.visibilityMask = 0xff;
                    ambientTriangles.push_back(tri);
                }
            };

            append(state.scene.worldFaceTriangles);
            append(state.scene.worldBrushTriangles);
            append(state.scene.displacementTriangles);
        }

        float3 calc_ray_ambient_lighting(
            const ::BSP::BSP& bsp,
            const RuntimeState& state,
            const std::vector<float3>& lightSamples,
            const std::vector<OptixRT::Triangle>& triangles,
            const OptixRT::SunRay& ray,
            const OptixRT::RayHit& hit,
            float3 skyAmbient
        )
        {
            if (!hit.hit || hit.primitiveIndex >= triangles.size()) {
                return make_float3();
            }

            const Tracing::SurfaceTraceResult surfaceTrace =
                Tracing::test_line_surface(hit, triangles);
            if (surfaceTrace.hitSky) {
                return skyAmbient;
            }

            if (surfaceTrace.sourceId == INVALID_SOURCE_FACE
                || surfaceTrace.sourceId >= bsp.get_dfaces().size()) {
                return make_float3();
            }

            const size_t faceIndex = surfaceTrace.sourceId;
            const Common::Vec3f hitPoint = from_float3(ray.origin + ray.direction * hit.t);

            if (faceIndex < state.dispGeometry.size() && state.dispGeometry[faceIndex].valid) {
                return compute_lightmap_color_displacement(
                    bsp,
                    state,
                    lightSamples,
                    faceIndex,
                    hitPoint,
                    skyAmbient
                );
            }

            return compute_lightmap_color_from_average(
                bsp,
                faceIndex,
                skyAmbient
            );
        }

    }

    void compute_leaf_ambient_from_cuda_runtime(
        const ::BSP::BSP& bsp,
        ::CUDABSP::CUDABSP* pCudaBSP,
        const RuntimeState& state,
        OptixRT::OptixSunLosTracer& tracer
    )
    {
        ::CUDABSP::CUDABSP cudaBSP;
        CUDA_CHECK_ERROR(cudaMemcpy(
            &cudaBSP,
            pCudaBSP,
            sizeof(::CUDABSP::CUDABSP),
            cudaMemcpyDeviceToHost
        ));

        std::vector<float3> rawLightSamples(
            bsp.get_lightsamples().size(),
            make_float3()
        );
        if (!rawLightSamples.empty()) {
            CUDA_CHECK_ERROR(cudaMemcpy(
                rawLightSamples.data(),
                cudaBSP.lightSamples,
                sizeof(float3) * rawLightSamples.size(),
                cudaMemcpyDeviceToHost
            ));
        }

        std::vector<float3> lightSamples(
            rawLightSamples.size(),
            make_float3()
        );
        for (size_t i = 0; i < rawLightSamples.size(); ++i) {
            lightSamples[i] = rgbexp32_to_linear_float3(
                rgbexp32_from_float3(rawLightSamples[i])
            );
        }

        std::vector<::BSP::CompressedLightCube> leafAmbient;
        compute_leaf_ambient_runtime(
            bsp,
            state,
            lightSamples,
            leafAmbient,
            tracer
        );

        if (!leafAmbient.empty()) {
            CUDA_CHECK_ERROR(cudaMemcpy(
                cudaBSP.ambientLightSamples,
                leafAmbient.data(),
                sizeof(::BSP::CompressedLightCube) * leafAmbient.size(),
                cudaMemcpyHostToDevice
            ));
        }

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    void compute_leaf_ambient_runtime(
        const ::BSP::BSP& bsp,
        const RuntimeState& state,
        const std::vector<float3>& lightSamples,
        std::vector<::BSP::CompressedLightCube>& leafAmbient,
        OptixRT::OptixSunLosTracer& tracer
    )
    {
        leafAmbient.assign(bsp.get_leaves().size(), ::BSP::CompressedLightCube{});

        std::vector<uint8_t> activeLeaves(bsp.get_leaves().size(), 0);
        std::vector<float3> leafCenters(bsp.get_leaves().size(), make_float3());
        std::vector<std::array<float3, 6>> leafColors(bsp.get_leaves().size());
        std::vector<std::array<float, 6>> leafWeights(bsp.get_leaves().size());

        for (size_t leafIndex = 0; leafIndex < bsp.get_leaves().size(); ++leafIndex) {
            const ::BSP::DLeaf& leaf = bsp.get_leaves()[leafIndex];
            for (size_t side = 0; side < 6; ++side) {
                leafAmbient[leafIndex].color[side] = rgbexp32_from_float3(make_float3());
                leafColors[leafIndex][side] = make_float3();
                leafWeights[leafIndex][side] = 0.0f;
            }

            if ((leaf.contents & ::BSP::CONTENTS_SOLID) != 0) {
                continue;
            }

            activeLeaves[leafIndex] = 1;
            leafCenters[leafIndex] = make_float3(
                (static_cast<float>(leaf.mins[0]) + static_cast<float>(leaf.maxs[0])) * 0.5f,
                (static_cast<float>(leaf.mins[1]) + static_cast<float>(leaf.maxs[1])) * 0.5f,
                (static_cast<float>(leaf.mins[2]) + static_cast<float>(leaf.maxs[2])) * 0.5f
            );
        }

        std::vector<OptixRT::Triangle> ambientTriangles;
        build_ambient_world(state, ambientTriangles);
        tracer.build_world_gas(ambientTriangles);

        const float3 skyAmbient = find_sky_ambient(bsp);

        std::vector<OptixRT::SunRay> ambientRays;
        std::vector<AmbientRayContribution> ambientContributions;
        ambientRays.reserve(bsp.get_leaves().size() * NUM_VERTEX_NORMALS);
        ambientContributions.reserve(bsp.get_leaves().size() * NUM_VERTEX_NORMALS);

        for (size_t leafIndex = 0; leafIndex < bsp.get_leaves().size(); ++leafIndex) {
            if (!activeLeaves[leafIndex]) {
                continue;
            }

            for (size_t dirIndex = 0; dirIndex < NUM_VERTEX_NORMALS; ++dirIndex) {
                OptixRT::SunRay ray{};
                ray.origin = leafCenters[leafIndex];
                ray.direction = ANORMS[dirIndex];
                ray.tmin = HOST_RAY_TMIN;
                ray.tmax = AMBIENT_TRACE_DISTANCE;
                ray.visibilityMask = 0xff;
                if (!is_valid_ray(ray)) {
                    continue;
                }

                ambientRays.push_back(ray);
                ambientContributions.push_back(AmbientRayContribution{
                    leafIndex,
                    dirIndex,
                });
            }
        }

        std::cout << "Tracing " << ambientRays.size()
            << " core leaf ambient spherical rays..." << std::endl;

        std::vector<OptixRT::RayHit> ambientHits;
        tracer.trace_batch(ambientRays, ambientHits);

        for (size_t i = 0; i < ambientHits.size(); ++i) {
            const AmbientRayContribution& contribution = ambientContributions[i];
            const float3 radcolor = calc_ray_ambient_lighting(
                bsp,
                state,
                lightSamples,
                ambientTriangles,
                ambientRays[i],
                ambientHits[i],
                skyAmbient
            );

            for (size_t side = 0; side < 6; ++side) {
                const float c = dot(ANORMS[contribution.dirIndex], BOX_DIRECTIONS[side]);
                if (c <= 0.0f) {
                    continue;
                }

                leafWeights[contribution.leafIndex][side] += c;
                leafColors[contribution.leafIndex][side] += radcolor * c;
            }
        }

        for (size_t leafIndex = 0; leafIndex < bsp.get_leaves().size(); ++leafIndex) {
            if (!activeLeaves[leafIndex]) {
                continue;
            }

            for (size_t side = 0; side < 6; ++side) {
                const float weight = leafWeights[leafIndex][side];
                if (weight > 1e-6f) {
                    leafColors[leafIndex][side] /= weight;
                }
            }
        }

        std::vector<size_t> ambientSurfaceLights;
        ambientSurfaceLights.reserve(bsp.get_worldlights().size());
        for (size_t lightIndex = 0; lightIndex < bsp.get_worldlights().size(); ++lightIndex) {
            if (is_leaf_ambient_surface_light(bsp.get_worldlights()[lightIndex])) {
                ambientSurfaceLights.push_back(lightIndex);
            }
        }

        std::vector<OptixRT::SunRay> emitRays;
        std::vector<EmitSurfaceContribution> emitContributions;
        for (size_t leafIndex = 0; leafIndex < bsp.get_leaves().size(); ++leafIndex) {
            if (!activeLeaves[leafIndex]) {
                continue;
            }

            const float3 center = leafCenters[leafIndex];
            for (size_t listIndex = 0; listIndex < ambientSurfaceLights.size(); ++listIndex) {
                const size_t lightIndex = ambientSurfaceLights[listIndex];
                const ::BSP::DWorldLight& light = bsp.get_worldlights()[lightIndex];
                const float3 lightOrigin = make_float3(light.origin);
                const float3 delta = lightOrigin - center;
                const float dist = len(delta);
                if (dist <= 1e-6f) {
                    continue;
                }

                OptixRT::SunRay ray{};
                ray.origin = center;
                ray.direction = delta / dist;
                ray.tmin = HOST_RAY_TMIN;
                ray.tmax = dist - HOST_RAY_TMIN;
                ray.visibilityMask = 0xff;
                if (!is_valid_ray(ray)) {
                    continue;
                }

                emitRays.push_back(ray);
                emitContributions.push_back(EmitSurfaceContribution{
                    leafIndex,
                    lightIndex,
                });
            }
        }

        std::cout << "Tracing " << emitRays.size()
            << " core leaf ambient emit-surface rays..." << std::endl;

        std::vector<OptixRT::RayHit> emitHits;
        tracer.trace_batch(emitRays, emitHits);

        for (size_t i = 0; i < emitHits.size(); ++i) {
            if (emitHits[i].hit) {
                continue;
            }

            const EmitSurfaceContribution& contribution = emitContributions[i];
            const ::BSP::DWorldLight& light = bsp.get_worldlights()[contribution.lightIndex];
            const float3 center = leafCenters[contribution.leafIndex];
            const float3 lightOrigin = make_float3(light.origin);
            const float3 delta = lightOrigin - center;
            const float3 deltaNormal = safe_normalized(delta);
            if (!is_finite(deltaNormal) || len(deltaNormal) <= 1e-6f) {
                continue;
            }

            const float distanceScale = engine_world_light_distance_falloff(light, delta);
            const float angleScale = engine_world_light_angle(light, deltaNormal);
            const float ratio = distanceScale * angleScale;
            if (ratio <= 0.0f || !std::isfinite(ratio)) {
                continue;
            }

            const float3 intensity = make_float3(light.intensity) * 255.0f;
            for (size_t side = 0; side < 6; ++side) {
                const float t = dot(BOX_DIRECTIONS[side], deltaNormal);
                if (t > 0.0f) {
                    leafColors[contribution.leafIndex][side] += intensity * (t * ratio);
                }
            }
        }

        for (size_t leafIndex = 0; leafIndex < bsp.get_leaves().size(); ++leafIndex) {
            for (size_t side = 0; side < 6; ++side) {
                leafAmbient[leafIndex].color[side] =
                    rgbexp32_from_float3(leafColors[leafIndex][side]);
            }
        }
    }
}
