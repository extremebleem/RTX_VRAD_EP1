#include "direct_lighting.h"
#include "ambient_lighting.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../../cudautils.h"
#include "../bridge/backend.h"
#include "../common/math.h"

namespace SilkRAD::V2::Lighting {
    namespace {
        static constexpr uint32_t RTX_ROLE_SKY = 1u << 2;
        static constexpr size_t HOST_SKY_AMBIENT_SAMPLES = 16;
        static constexpr size_t HOST_GI_SAMPLES = 16;
        static constexpr float HOST_COORD_EXTENT = 32768.0f;
        static constexpr float HOST_INDIRECT_SCALE = 0.75f;
        static constexpr float HOST_SKY_GI_SCALE = 0.35f;
        static constexpr float HOST_AMBIENT_FLOOR_SCALE = 0.06f;
        static constexpr float HOST_RAY_BIAS = 0.03125f;
        static constexpr float HOST_RAY_TMIN = 0.001f;
        static constexpr size_t HOST_DIRECT_SUBSAMPLES = 16;
        static constexpr float HOST_SUN_ANGULAR_RADIUS = 0.02f;
        static constexpr float HOST_LEAF_SURFACE_AMBIENT_SCALE = 0.08f;
        static constexpr float HOST_LEAF_SKY_AMBIENT_SCALE = 0.75f;
        static constexpr float HOST_LEAF_AMBIENT_MAX = 96.0f;

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

        float3 soft_sun_direction(float3 sunDir, size_t sampleIndex)
        {
            const float2 aperture = luxel_subsample_offset(sampleIndex);
            const float3 tangent = make_tangent(sunDir);
            const float3 bitangent = make_bitangent(sunDir, tangent);
            return safe_normalized(
                sunDir
                + tangent * (aperture.x * HOST_SUN_ANGULAR_RADIUS)
                + bitangent * (aperture.y * HOST_SUN_ANGULAR_RADIUS)
            );
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

        bool v2_receiver_sample(
            const Bridge::BackendState& state,
            size_t faceIndex,
            size_t s,
            size_t t,
            float2 sampleOffset,
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
                const float fu = 0.5f + sampleOffset.x;
                const float fv = 0.5f + sampleOffset.y;
                const float u = sample.mins.x + (sample.maxs.x - sample.mins.x) * fu;
                const float v = sample.mins.y + (sample.maxs.y - sample.mins.y) * fv;

                Common::Vec3f pos;
                Common::Vec3f normal;
                if (!Geometry::disp_uv_to_surf_point(dispGeometry, u, v, 1.0f, pos)) {
                    return false;
                }
                if (!Geometry::disp_uv_to_surf_normal(dispGeometry, u, v, normal)) {
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
            const float fu = 0.5f + sampleOffset.x;
            const float fv = 0.5f + sampleOffset.y;
            const float ss = sample.mins.x + (sample.maxs.x - sample.mins.x) * fu;
            const float tt = sample.mins.y + (sample.maxs.y - sample.mins.y) * fv;

            outNormal = safe_normalized(to_float3(sample.normal));
            const float ds = ss - sample.coord.x;
            const float dt = tt - sample.coord.y;
            const Common::Vec3f shiftedPos = Common::add(
                sample.pos,
                Common::add(
                    Common::scale(faceGeometry.luxelToWorldSpace[0], ds),
                    Common::scale(faceGeometry.luxelToWorldSpace[1], dt)
                )
            );
            outPos = to_float3(shiftedPos) + outNormal * HOST_RAY_BIAS;
            return is_finite(outPos) && is_finite(outNormal);
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
        return result;
    }

    void build_runtime_world(
        const Bridge::BackendState& state,
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
        const Bridge::BackendState& state,
        const std::vector<OptixRT::Triangle>& triangles,
        OptixRT::OptixSunLosTracer& tracer
    )
    {
        using Clock = std::chrono::high_resolution_clock;

        enum class DirectVisibilityMode : uint8_t {
            Unblocked,
            SkyFirst,
        };

        struct RayContribution {
            size_t lightmapIndex;
            float3 color;
            DirectVisibilityMode visibilityMode;
        };

        struct GiRayContribution {
            size_t lightmapIndex;
            float3 reflectivity;
            float weight;
        };

        auto clampf_host = [](float v, float lo, float hi) {
            return std::max(lo, std::min(v, hi));
        };

        auto attenuate_host = [](const ::BSP::DWorldLight& light, float dist) {
            return light.constantAtten
                + light.linearAtten * dist
                + light.quadraticAtten * dist * dist;
        };

        std::vector<float3> lightSamples(
            bsp.get_lightsamples().size(),
            make_float3()
        );
        std::vector<::BSP::DFace> dFaces = bsp.get_dfaces();
        std::vector<OptixRT::SunRay> rays;
        std::vector<RayContribution> contributions;

        const size_t numFaces = bsp.get_faces().size();
        size_t processedFaces = 0;

        std::cout << "Building v2 OptiX direct-lighting ray batch for "
            << numFaces << " faces at " << HOST_DIRECT_SUBSAMPLES
            << " spp/luxel..." << std::endl;

        const auto startTime = Clock::now();

        for (size_t faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
            const ::BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.lightOffset < 0) {
                ++processedFaces;
                continue;
            }

            const Geometry::DispGeometry& dispGeometry = state.dispGeometry[faceIndex];
            const Geometry::FaceGeometry& faceGeometry = state.faceGeometry[faceIndex];
            const bool validReceiver =
                dispGeometry.valid || (faceGeometry.valid && !faceGeometry.isDisplacement);
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
                    const float subSampleWeight =
                        1.0f / static_cast<float>(HOST_DIRECT_SUBSAMPLES);

                    for (size_t directSample = 0; directSample < HOST_DIRECT_SUBSAMPLES; ++directSample) {
                        const float2 sampleOffset = luxel_subsample_offset(directSample);
                        float3 samplePos;
                        float3 sampleNormal;
                        if (!v2_receiver_sample(
                            state,
                            faceIndex,
                            s,
                            t,
                            sampleOffset,
                            samplePos,
                            sampleNormal
                        )) {
                            continue;
                        }

                        const ::BSP::Vec3<float> sampleVec{
                            samplePos.x,
                            samplePos.y,
                            samplePos.z,
                        };
                        const int16_t sampleCluster = bsp.cluster_for_pos(sampleVec);

                        for (const ::BSP::DWorldLight& light : bsp.get_worldlights()) {
                            if (light.type == ::BSP::EMIT_SKYAMBIENT) {
                                continue;
                            }

                            if (light.type == ::BSP::EMIT_SKYLIGHT) {
                                const float3 lightNormal = make_float3(light.normal);
                                const float3 sunDir =
                                    soft_sun_direction(safe_normalized(lightNormal) * -1.0f, directSample);
                                const float ndotl = dot(sampleNormal, sunDir);
                                if (ndotl <= 0.0f || !is_finite(sunDir)) {
                                    continue;
                                }

                                OptixRT::SunRay ray{};
                                ray.origin = samplePos + sunDir * HOST_RAY_TMIN;
                                ray.direction = sunDir;
                                ray.tmin = HOST_RAY_TMIN;
                                ray.tmax = HOST_COORD_EXTENT;
                                ray.visibilityMask = 0xff;
                                if (!is_valid_ray(ray)) {
                                    continue;
                                }

                                rays.push_back(ray);
                                contributions.push_back(RayContribution{
                                    lightmapIndex,
                                    make_float3(light.intensity) * ndotl * 255.0f * subSampleWeight,
                                    DirectVisibilityMode::SkyFirst,
                                });
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
                            if (dot(diff, sampleNormal) >= 0.0f) {
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

                            const float3 shadowStart = samplePos + sampleNormal * HOST_RAY_BIAS;
                            const float3 shadowDelta = lightPos - shadowStart;
                            const float shadowDist = len(shadowDelta);
                            if (shadowDist <= 1e-6f) {
                                continue;
                            }

                            const float attenuation = attenuate_host(light, distToLight);
                            if (!std::isfinite(attenuation) || attenuation <= 1e-6f) {
                                continue;
                            }

                            float3 lightContribution = make_float3(light.intensity);
                            lightContribution *=
                                penumbraScale * 255.0f
                                * subSampleWeight / attenuation;

                            OptixRT::SunRay ray{};
                            ray.origin = shadowStart;
                            ray.direction = shadowDelta / shadowDist;
                            ray.tmin = HOST_RAY_TMIN;
                            ray.tmax = shadowDist - HOST_RAY_TMIN;
                            ray.visibilityMask = 0xff;
                            if (!is_valid_ray(ray)) {
                                continue;
                            }

                            rays.push_back(ray);
                            contributions.push_back(RayContribution{
                                lightmapIndex,
                                lightContribution,
                                DirectVisibilityMode::Unblocked,
                            });
                        }
                    }
                }
            }

            ++processedFaces;
            if (processedFaces % 32 == 0 || processedFaces == numFaces) {
                std::cout << "    " << processedFaces << "/"
                    << numFaces << " faces batched..." << std::endl;
            }
        }

        std::cout << "Tracing " << rays.size()
            << " v2 direct-lighting rays with OptiX..." << std::endl;

        std::vector<OptixRT::RayHit> directHits;
        tracer.trace_batch(rays, directHits);

        for (size_t i = 0; i < contributions.size(); ++i) {
            const uint32_t currentHitRole = hit_role(triangles, directHits[i]);
            const RayContribution& contribution = contributions[i];
            const bool visible =
                contribution.visibilityMode == DirectVisibilityMode::SkyFirst
                    ? ((currentHitRole & RTX_ROLE_SKY) != 0)
                    : (!directHits[i].hit || (currentHitRole & RTX_ROLE_SKY) != 0);

            if (visible) {
                lightSamples[contribution.lightmapIndex] += contribution.color;
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

        const auto endTime = Clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime
        );
        std::cout << "Done! (" << ms.count() << " ms)" << std::endl;
    }
}
