#include "disp_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "face_geometry.h"

namespace SilkRAD::Core::Geometry {
    namespace {
        constexpr float TRIEDGE_EPSILON = 0.001f;

        Common::Vec3f from_bsp_vec3(const ::BSP::Vec3<float>& v)
        {
            return Common::make_vec3(v.x, v.y, v.z);
        }

        Common::Vec3f bilerp(
            Common::Vec3f p00,
            Common::Vec3f p10,
            Common::Vec3f p11,
            Common::Vec3f p01,
            float u,
            float v
        )
        {
            const Common::Vec3f pu0 = Common::add(
                p00,
                Common::scale(Common::sub(p10, p00), u)
            );
            const Common::Vec3f pu1 = Common::add(
                p01,
                Common::scale(Common::sub(p11, p01), u)
            );
            return Common::add(
                pu0,
                Common::scale(Common::sub(pu1, pu0), v)
            );
        }

        size_t disp_index(const DispGeometry& geometry, size_t x, size_t y)
        {
            return y * geometry.gridSize + x;
        }

        size_t find_surf_point_start_index(
            const std::vector<Common::Vec3f>& points,
            Common::Vec3f pointStart
        )
        {
            size_t minIndex = 0;
            float minDistance = std::numeric_limits<float>::max();

            for (size_t i = 0; i < points.size(); ++i) {
                const Common::Vec3f segment = Common::sub(pointStart, points[i]);
                const float distanceSq = Common::dot(segment, segment);
                if (distanceSq < minDistance) {
                    minDistance = distanceSq;
                    minIndex = i;
                }
            }

            return minIndex;
        }

        void adjust_surf_point_data(
            std::vector<Common::Vec3f>& points,
            size_t pointStartIndex
        )
        {
            std::vector<Common::Vec3f> tmpPoints = points;
            for (size_t i = 0; i < points.size(); ++i) {
                points[i] = tmpPoints[(i + pointStartIndex) % points.size()];
            }
        }

        void orient_normal(Common::Vec3f faceNormal, Common::Vec3f& normal)
        {
            if (Common::dot(normal, faceNormal) < 0.0f) {
                normal = Common::scale(normal, -1.0f);
            }
        }

        bool does_edge_exist(size_t indexRow, size_t indexCol, int direction, size_t postSpacing)
        {
            switch (direction) {
            case 0:
                return indexRow > 0;
            case 1:
                return (indexCol + 1) < postSpacing;
            case 2:
                return (indexRow + 1) < postSpacing;
            case 3:
                return indexCol > 0;
            default:
                return false;
            }
        }

        Common::Vec3f calc_normal_from_edges(
            const DispGeometry& geometry,
            size_t indexRow,
            size_t indexCol
        )
        {
            const size_t postSpacing = geometry.gridSize;
            Common::Vec3f accumNormal = Common::make_vec3(0.0f, 0.0f, 0.0f);
            int normalCount = 0;

            auto at = [&](size_t col, size_t row) -> const Common::Vec3f& {
                return geometry.surfaceVertices[col * postSpacing + row].pos;
            };

            const bool edge0 = does_edge_exist(indexRow, indexCol, 0, postSpacing);
            const bool edge1 = does_edge_exist(indexRow, indexCol, 1, postSpacing);
            const bool edge2 = does_edge_exist(indexRow, indexCol, 2, postSpacing);
            const bool edge3 = does_edge_exist(indexRow, indexCol, 3, postSpacing);

            Common::Vec3f tmpVect0;
            Common::Vec3f tmpVect1;
            Common::Vec3f tmpNormal;

            if (edge1 && edge2) {
                tmpVect0 = Common::sub(at(indexCol + 1, indexRow), at(indexCol, indexRow));
                tmpVect1 = Common::sub(at(indexCol, indexRow + 1), at(indexCol, indexRow));
                tmpNormal = Common::normalized(Common::cross(tmpVect1, tmpVect0));
                accumNormal = Common::add(accumNormal, tmpNormal);
                ++normalCount;

                tmpVect0 = Common::sub(at(indexCol + 1, indexRow), at(indexCol, indexRow + 1));
                tmpVect1 = Common::sub(at(indexCol + 1, indexRow + 1), at(indexCol, indexRow + 1));
                tmpNormal = Common::normalized(Common::cross(tmpVect1, tmpVect0));
                accumNormal = Common::add(accumNormal, tmpNormal);
                ++normalCount;
            }

            if (edge0 && edge1) {
                tmpVect0 = Common::sub(at(indexCol + 1, indexRow - 1), at(indexCol, indexRow - 1));
                tmpVect1 = Common::sub(at(indexCol, indexRow), at(indexCol, indexRow - 1));
                tmpNormal = Common::normalized(Common::cross(tmpVect1, tmpVect0));
                accumNormal = Common::add(accumNormal, tmpNormal);
                ++normalCount;

                tmpVect0 = Common::sub(at(indexCol + 1, indexRow - 1), at(indexCol, indexRow));
                tmpVect1 = Common::sub(at(indexCol + 1, indexRow), at(indexCol, indexRow));
                tmpNormal = Common::normalized(Common::cross(tmpVect1, tmpVect0));
                accumNormal = Common::add(accumNormal, tmpNormal);
                ++normalCount;
            }

            if (edge0 && edge3) {
                tmpVect0 = Common::sub(at(indexCol, indexRow - 1), at(indexCol - 1, indexRow - 1));
                tmpVect1 = Common::sub(at(indexCol - 1, indexRow), at(indexCol - 1, indexRow - 1));
                tmpNormal = Common::normalized(Common::cross(tmpVect1, tmpVect0));
                accumNormal = Common::add(accumNormal, tmpNormal);
                ++normalCount;

                tmpVect0 = Common::sub(at(indexCol, indexRow - 1), at(indexCol - 1, indexRow));
                tmpVect1 = Common::sub(at(indexCol, indexRow), at(indexCol - 1, indexRow));
                tmpNormal = Common::normalized(Common::cross(tmpVect1, tmpVect0));
                accumNormal = Common::add(accumNormal, tmpNormal);
                ++normalCount;
            }

            if (edge2 && edge3) {
                tmpVect0 = Common::sub(at(indexCol, indexRow), at(indexCol - 1, indexRow));
                tmpVect1 = Common::sub(at(indexCol - 1, indexRow + 1), at(indexCol - 1, indexRow));
                tmpNormal = Common::normalized(Common::cross(tmpVect1, tmpVect0));
                accumNormal = Common::add(accumNormal, tmpNormal);
                ++normalCount;

                tmpVect0 = Common::sub(at(indexCol, indexRow), at(indexCol - 1, indexRow + 1));
                tmpVect1 = Common::sub(at(indexCol, indexRow + 1), at(indexCol - 1, indexRow + 1));
                tmpNormal = Common::normalized(Common::cross(tmpVect1, tmpVect0));
                accumNormal = Common::add(accumNormal, tmpNormal);
                ++normalCount;
            }

            if (normalCount <= 0) {
                return geometry.faceNormal;
            }

            Common::Vec3f normal = Common::scale(accumNormal, 1.0f / static_cast<float>(normalCount));
            normal = Common::normalized(normal);
            if (!Common::is_finite(normal) || Common::length(normal) <= 1e-6f) {
                normal = geometry.faceNormal;
            }
            orient_normal(geometry.faceNormal, normal);
            return normal;
        }

        bool disp_triangle_vertices(
            const DispGeometry& geometry,
            uint16_t triangleIndex,
            size_t& i0,
            size_t& i1,
            size_t& i2
        )
        {
            if (geometry.gridSize < 2) {
                return false;
            }

            const size_t cellsPerRow = geometry.gridSize - 1;
            const size_t quadCount = cellsPerRow * cellsPerRow;
            const size_t triIndex = static_cast<size_t>(triangleIndex);
            if (triIndex >= quadCount * 2) {
                return false;
            }

            const size_t cellIndex = triIndex / 2;
            const size_t triInCell = triIndex % 2;
            const size_t y = cellIndex / cellsPerRow;
            const size_t x = cellIndex % cellsPerRow;
            const bool odd = (((y * geometry.gridSize) + x) & 1u) != 0;

            const size_t q0 = disp_index(geometry, x, y);
            const size_t q1 = disp_index(geometry, x, y + 1);
            const size_t q2 = disp_index(geometry, x + 1, y + 1);
            const size_t q3 = disp_index(geometry, x + 1, y);

            if (odd) {
                if (triInCell == 0) {
                    i0 = q0; i1 = q1; i2 = q3;
                }
                else {
                    i0 = q1; i1 = q2; i2 = q3;
                }
            }
            else {
                if (triInCell == 0) {
                    i0 = q0; i1 = q1; i2 = q2;
                }
                else {
                    i0 = q0; i1 = q2; i2 = q3;
                }
            }

            return true;
        }

        bool disp_sample_mapping_position(
            const DispGeometry& geometry,
            const DispSampleMapping& mapping,
            float pushEps,
            Common::Vec3f& outPos,
            Common::Vec3f& outNormal
        )
        {
            if (!mapping.valid) {
                return false;
            }

            size_t i0 = 0;
            size_t i1 = 0;
            size_t i2 = 0;
            if (!disp_triangle_vertices(geometry, mapping.triangleIndex, i0, i1, i2)) {
                return false;
            }

            Common::Vec3f bary = mapping.barycentric;
            const float barySum = bary.x + bary.y + bary.z;
            if (barySum <= 1e-6f) {
                return false;
            }

            bary = Common::scale(bary, 1.0f / barySum);

            const Common::Vec3f p0 = geometry.surfaceVertices[i0].pos;
            const Common::Vec3f p1 = geometry.surfaceVertices[i1].pos;
            const Common::Vec3f p2 = geometry.surfaceVertices[i2].pos;
            const Common::Vec3f n0 = geometry.surfaceVertices[i0].normal;
            const Common::Vec3f n1 = geometry.surfaceVertices[i1].normal;
            const Common::Vec3f n2 = geometry.surfaceVertices[i2].normal;

            Common::Vec3f normal = Common::add(
                Common::add(
                    Common::scale(n0, bary.x),
                    Common::scale(n1, bary.y)
                ),
                Common::scale(n2, bary.z)
            );
            normal = Common::normalized(normal);
            if (!Common::is_finite(normal) || Common::length(normal) <= 1e-6f) {
                normal = geometry.faceNormal;
            }
            orient_normal(geometry.faceNormal, normal);

            const Common::Vec3f point = Common::add(
                Common::add(
                    Common::scale(p0, bary.x),
                    Common::scale(p1, bary.y)
                ),
                Common::scale(p2, bary.z)
            );

            outNormal = normal;
            outPos = Common::add(point, Common::scale(normal, pushEps));
            return Common::is_finite(outPos) && Common::is_finite(outNormal);
        }

        bool barycentric_coordinates(
            Common::Vec3f a,
            Common::Vec3f b,
            Common::Vec3f c,
            Common::Vec3f p,
            Common::Vec3f& barycentric,
            float& planeDistance
        )
        {
            const Common::Vec3f v0 = Common::sub(b, a);
            const Common::Vec3f v1 = Common::sub(c, a);
            const Common::Vec3f v2 = Common::sub(p, a);
            const Common::Vec3f normal = Common::cross(v0, v1);
            const float normalLen = Common::length(normal);
            if (!Common::is_finite(normal) || normalLen <= 1e-6f) {
                return false;
            }

            const Common::Vec3f unitNormal = Common::scale(normal, 1.0f / normalLen);
            planeDistance = std::fabs(Common::dot(v2, unitNormal));

            const float d00 = Common::dot(v0, v0);
            const float d01 = Common::dot(v0, v1);
            const float d11 = Common::dot(v1, v1);
            const float d20 = Common::dot(v2, v0);
            const float d21 = Common::dot(v2, v1);
            const float denom = d00 * d11 - d01 * d01;
            if (std::fabs(denom) <= 1e-12f) {
                return false;
            }

            const float v = (d11 * d20 - d01 * d21) / denom;
            const float w = (d00 * d21 - d01 * d20) / denom;
            const float u = 1.0f - v - w;
            barycentric = Common::make_vec3(u, v, w);
            return Common::is_finite(barycentric);
        }

        std::vector<DispSampleMapping> decode_disp_sample_mappings(
            const BSP::SourceMap& sourceMap,
            const ::BSP::DispInfo& dispInfo,
            const Common::LightmapDimensions& lightmap
        )
        {
            std::vector<DispSampleMapping> mappings;
            const std::vector<uint8_t>& raw = sourceMap.disp_lightmap_sample_positions();
            const size_t sampleCount = lightmap.sample_count();
            mappings.reserve(sampleCount);

            size_t cursor = dispInfo.lightmapSamplePositionStart >= 0
                ? static_cast<size_t>(dispInfo.lightmapSamplePositionStart)
                : raw.size();

            for (size_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
                DispSampleMapping mapping;

                if (cursor + 4 > raw.size()) {
                    mappings.push_back(mapping);
                    continue;
                }

                uint16_t triIndex = raw[cursor++];
                if (triIndex == 255) {
                    if (cursor >= raw.size()) {
                        mappings.push_back(mapping);
                        continue;
                    }
                    triIndex = static_cast<uint16_t>(255 + raw[cursor++]);
                }

                if (cursor + 3 > raw.size()) {
                    mappings.push_back(mapping);
                    continue;
                }

                const uint8_t bary0 = raw[cursor++];
                const uint8_t bary1 = raw[cursor++];
                const uint8_t bary2 = raw[cursor++];

                mapping.valid = !(triIndex == 0 && bary0 == 0 && bary1 == 0 && bary2 == 0);
                mapping.triangleIndex = triIndex;
                mapping.barycentric = Common::make_vec3(
                    static_cast<float>(bary0) / 255.0f,
                    static_cast<float>(bary1) / 255.0f,
                    static_cast<float>(bary2) / 255.0f
                );
                mappings.push_back(mapping);
            }

            return mappings;
        }
    }

    bool disp_uv_to_surf_point(
        const DispGeometry& geometry,
        float u,
        float v,
        float pushEps,
        Common::Vec3f& outPos
    )
    {
        if (!geometry.valid || geometry.gridSize < 2 || geometry.surfaceVertices.empty()) {
            return false;
        }

        if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
            return false;
        }

        const float flU = u * static_cast<float>(geometry.gridSize - 1.000001f);
        const float flV = v * static_cast<float>(geometry.gridSize - 1.000001f);
        const size_t snapU = static_cast<size_t>(flU);
        const size_t snapV = static_cast<size_t>(flV);
        const size_t nextU = std::min(snapU + 1, geometry.gridSize - 1);
        const size_t nextV = std::min(snapV + 1, geometry.gridSize - 1);
        const float fracU = flU - static_cast<float>(snapU);
        const float fracV = flV - static_cast<float>(snapV);
        const bool odd = (((snapV * geometry.gridSize) + snapU) & 1u) != 0;

        Common::Vec3f point;
        Common::Vec3f triNormal = geometry.faceNormal;

        if (odd) {
            if ((fracU + fracV) >= (1.0f + TRIEDGE_EPSILON)) {
                const size_t i0 = disp_index(geometry, snapU, nextV);
                const size_t i1 = disp_index(geometry, nextU, nextV);
                const size_t i2 = disp_index(geometry, nextU, snapV);
                const Common::Vec3f edgeU = Common::sub(geometry.surfaceVertices[i0].pos, geometry.surfaceVertices[i1].pos);
                const Common::Vec3f edgeV = Common::sub(geometry.surfaceVertices[i2].pos, geometry.surfaceVertices[i1].pos);
                point = Common::add(
                    geometry.surfaceVertices[i1].pos,
                    Common::add(
                        Common::scale(edgeU, 1.0f - fracU),
                        Common::scale(edgeV, 1.0f - fracV)
                    )
                );
                triNormal = Common::normalized(Common::cross(edgeU, edgeV));
            }
            else {
                const size_t i0 = disp_index(geometry, snapU, snapV);
                const size_t i1 = disp_index(geometry, snapU, nextV);
                const size_t i2 = disp_index(geometry, nextU, snapV);
                const Common::Vec3f edgeU = Common::sub(geometry.surfaceVertices[i2].pos, geometry.surfaceVertices[i0].pos);
                const Common::Vec3f edgeV = Common::sub(geometry.surfaceVertices[i1].pos, geometry.surfaceVertices[i0].pos);
                point = Common::add(
                    geometry.surfaceVertices[i0].pos,
                    Common::add(
                        Common::scale(edgeU, fracU),
                        Common::scale(edgeV, fracV)
                    )
                );
                triNormal = Common::normalized(Common::cross(edgeU, edgeV));
            }
        }
        else {
            if (fracU < fracV) {
                const size_t i0 = disp_index(geometry, snapU, snapV);
                const size_t i1 = disp_index(geometry, snapU, nextV);
                const size_t i2 = disp_index(geometry, nextU, nextV);
                const Common::Vec3f edgeU = Common::sub(geometry.surfaceVertices[i2].pos, geometry.surfaceVertices[i1].pos);
                const Common::Vec3f edgeV = Common::sub(geometry.surfaceVertices[i0].pos, geometry.surfaceVertices[i1].pos);
                point = Common::add(
                    geometry.surfaceVertices[i1].pos,
                    Common::add(
                        Common::scale(edgeU, fracU),
                        Common::scale(edgeV, 1.0f - fracV)
                    )
                );
                triNormal = Common::normalized(Common::cross(edgeV, edgeU));
            }
            else {
                const size_t i0 = disp_index(geometry, snapU, snapV);
                const size_t i1 = disp_index(geometry, nextU, nextV);
                const size_t i2 = disp_index(geometry, nextU, snapV);
                const Common::Vec3f edgeU = Common::sub(geometry.surfaceVertices[i0].pos, geometry.surfaceVertices[i2].pos);
                const Common::Vec3f edgeV = Common::sub(geometry.surfaceVertices[i1].pos, geometry.surfaceVertices[i2].pos);
                point = Common::add(
                    geometry.surfaceVertices[i2].pos,
                    Common::add(
                        Common::scale(edgeU, 1.0f - fracU),
                        Common::scale(edgeV, fracV)
                    )
                );
                triNormal = Common::normalized(Common::cross(edgeV, edgeU));
            }
        }

        if (!Common::is_finite(triNormal) || Common::length(triNormal) <= 1e-6f) {
            triNormal = geometry.faceNormal;
        }
        orient_normal(geometry.faceNormal, triNormal);
        outPos = Common::add(point, Common::scale(triNormal, pushEps));
        return Common::is_finite(outPos);
    }

    bool disp_uv_to_surf_normal(
        const DispGeometry& geometry,
        float u,
        float v,
        Common::Vec3f& outNormal
    )
    {
        if (!geometry.valid || geometry.gridSize < 2 || geometry.surfaceVertices.empty()) {
            return false;
        }

        if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
            return false;
        }

        const float flU = u * static_cast<float>(geometry.gridSize - 1.000001f);
        const float flV = v * static_cast<float>(geometry.gridSize - 1.000001f);
        const size_t snapU = static_cast<size_t>(flU);
        const size_t snapV = static_cast<size_t>(flV);
        const size_t nextU = std::min(snapU + 1, geometry.gridSize - 1);
        const size_t nextV = std::min(snapV + 1, geometry.gridSize - 1);
        const float fracU = flU - static_cast<float>(snapU);
        const float fracV = flV - static_cast<float>(snapV);

        const size_t iQuad[4] = {
            disp_index(geometry, snapU, snapV),
            disp_index(geometry, snapU, nextV),
            disp_index(geometry, nextU, nextV),
            disp_index(geometry, nextU, snapV),
        };

        const Common::Vec3f& n0 = geometry.surfaceVertices[iQuad[0]].normal;
        const Common::Vec3f& n1 = geometry.surfaceVertices[iQuad[1]].normal;
        const Common::Vec3f& n2 = geometry.surfaceVertices[iQuad[2]].normal;
        const Common::Vec3f& n3 = geometry.surfaceVertices[iQuad[3]].normal;

        Common::Vec3f blended0 = Common::add(
            Common::scale(n0, 1.0f - fracU),
            Common::scale(n3, fracU)
        );
        blended0 = Common::normalized(blended0);

        Common::Vec3f blended1 = Common::add(
            Common::scale(n1, 1.0f - fracU),
            Common::scale(n2, fracU)
        );
        blended1 = Common::normalized(blended1);

        Common::Vec3f normal = Common::add(
            Common::scale(blended0, 1.0f - fracV),
            Common::scale(blended1, fracV)
        );
        normal = Common::normalized(normal);
        if (!Common::is_finite(normal) || Common::length(normal) <= 1e-6f) {
            normal = geometry.faceNormal;
        }
        orient_normal(geometry.faceNormal, normal);
        outNormal = normal;
        return true;
    }

    bool disp_world_to_uv(
        const DispGeometry& geometry,
        Common::Vec3f worldPoint,
        Common::Vec2f& outUv
    )
    {
        if (!geometry.valid || geometry.gridSize < 2 || geometry.surfaceVertices.empty()) {
            return false;
        }

        static constexpr float kBaryEpsilon = 1e-4f;
        static constexpr float kPlaneEpsilon = 0.25f;
        const size_t triangleCount = (geometry.gridSize - 1) * (geometry.gridSize - 1) * 2;

        for (size_t triIndex = 0; triIndex < triangleCount; ++triIndex) {
            size_t i0 = 0;
            size_t i1 = 0;
            size_t i2 = 0;
            if (!disp_triangle_vertices(
                geometry,
                static_cast<uint16_t>(triIndex),
                i0,
                i1,
                i2
            )) {
                continue;
            }

            Common::Vec3f barycentric;
            float planeDistance = 0.0f;
            if (!barycentric_coordinates(
                geometry.surfaceVertices[i0].pos,
                geometry.surfaceVertices[i1].pos,
                geometry.surfaceVertices[i2].pos,
                worldPoint,
                barycentric,
                planeDistance
            )) {
                continue;
            }

            if (planeDistance > kPlaneEpsilon) {
                continue;
            }

            if (barycentric.x < -kBaryEpsilon
                || barycentric.y < -kBaryEpsilon
                || barycentric.z < -kBaryEpsilon
                || barycentric.x > 1.0f + kBaryEpsilon
                || barycentric.y > 1.0f + kBaryEpsilon
                || barycentric.z > 1.0f + kBaryEpsilon) {
                continue;
            }

            const size_t x0 = i0 % geometry.gridSize;
            const size_t y0 = i0 / geometry.gridSize;
            const size_t x1 = i1 % geometry.gridSize;
            const size_t y1 = i1 / geometry.gridSize;
            const size_t x2 = i2 % geometry.gridSize;
            const size_t y2 = i2 / geometry.gridSize;
            const float scale = 1.0f / static_cast<float>(geometry.gridSize - 1);

            const Common::Vec2f uv0 = Common::make_vec2(
                static_cast<float>(x0) * scale,
                static_cast<float>(y0) * scale
            );
            const Common::Vec2f uv1 = Common::make_vec2(
                static_cast<float>(x1) * scale,
                static_cast<float>(y1) * scale
            );
            const Common::Vec2f uv2 = Common::make_vec2(
                static_cast<float>(x2) * scale,
                static_cast<float>(y2) * scale
            );

            outUv = Common::make_vec2(
                uv0.x * barycentric.x + uv1.x * barycentric.y + uv2.x * barycentric.z,
                uv0.y * barycentric.x + uv1.y * barycentric.y + uv2.y * barycentric.z
            );
            return true;
        }

        return false;
    }

    DispGeometry build_disp_geometry(
        const BSP::SourceMap& sourceMap,
        size_t faceIndex
    )
    {
        DispGeometry geometry;
        geometry.key.faceIndex = faceIndex;

        const ::BSP::DFace& face = sourceMap.dface(faceIndex);
        const ::BSP::DispInfo* dispInfo = sourceMap.dispinfo_for_face(faceIndex);
        if (dispInfo == nullptr) {
            return geometry;
        }

        geometry.key.dispInfoIndex = static_cast<size_t>(face.dispInfo);
        geometry.lightmap.width =
            static_cast<size_t>(face.lightmapTextureSizeInLuxels[0] + 1);
        geometry.lightmap.height =
            static_cast<size_t>(face.lightmapTextureSizeInLuxels[1] + 1);
        geometry.power = dispInfo->power;
        geometry.gridSize = static_cast<size_t>((1 << dispInfo->power) + 1);
        geometry.lightmapSamplePositionStart = dispInfo->lightmapSamplePositionStart;
        geometry.faceNormal = Common::normalized(from_bsp_vec3(
            sourceMap.plane(static_cast<size_t>(face.planeNum)).normal
        ));

        if (geometry.gridSize < 2 || geometry.lightmap.width == 0 || geometry.lightmap.height == 0) {
            return geometry;
        }

        FaceGeometry baseFace = build_face_geometry(sourceMap, faceIndex);
        if (!baseFace.valid) {
            return geometry;
        }

        const std::vector<Common::Vec3f> winding =
            face_world_winding(sourceMap, faceIndex, baseFace.modelOrigin);
        if (winding.size() != 4) {
            return geometry;
        }

        std::vector<Common::Vec3f> surfacePoints = winding;
        const size_t pointStartIndex = find_surf_point_start_index(
            surfacePoints,
            from_bsp_vec3(dispInfo->startPosition)
        );
        adjust_surf_point_data(surfacePoints, pointStartIndex);

        const Common::Vec3f& p00 = surfacePoints[0];
        const Common::Vec3f& p10 = surfacePoints[1];
        const Common::Vec3f& p11 = surfacePoints[2];
        const Common::Vec3f& p01 = surfacePoints[3];

        const std::vector<::BSP::DispVert>& dispVerts = sourceMap.dispverts();
        const size_t vertexCount = geometry.gridSize * geometry.gridSize;
        if (dispInfo->dispVertStart < 0
            || static_cast<size_t>(dispInfo->dispVertStart) + vertexCount > dispVerts.size()) {
            return geometry;
        }

        geometry.surfaceVertices.resize(vertexCount);
        const float ooInt = 1.0f / static_cast<float>(geometry.gridSize - 1);
        const Common::Vec3f edgeInt0 = Common::scale(Common::sub(p10, p00), ooInt);
        const Common::Vec3f edgeInt1 = Common::scale(Common::sub(p11, p01), ooInt);

        for (size_t i = 0; i < geometry.gridSize; ++i) {
            const Common::Vec3f endPt0 = Common::add(p00, Common::scale(edgeInt0, static_cast<float>(i)));
            const Common::Vec3f endPt1 = Common::add(p01, Common::scale(edgeInt1, static_cast<float>(i)));
            const Common::Vec3f seg = Common::sub(endPt1, endPt0);
            const Common::Vec3f segInt = Common::scale(seg, ooInt);

            for (size_t j = 0; j < geometry.gridSize; ++j) {
                const size_t index = i * geometry.gridSize + j;
                const ::BSP::DispVert& dispVert =
                    dispVerts[static_cast<size_t>(dispInfo->dispVertStart) + index];
                const Common::Vec3f flatVert = Common::add(
                    endPt0,
                    Common::scale(segInt, static_cast<float>(j))
                );
                const Common::Vec3f offset = Common::scale(from_bsp_vec3(dispVert.vector), dispVert.dist);
                geometry.surfaceVertices[index].pos = Common::add(flatVert, offset);
                geometry.surfaceVertices[index].normal = Common::make_vec3(0.0f, 0.0f, 0.0f);
            }
        }

        for (size_t i = 0; i < geometry.gridSize; ++i) {
            for (size_t j = 0; j < geometry.gridSize; ++j) {
                const size_t index = i * geometry.gridSize + j;
                geometry.surfaceVertices[index].normal = calc_normal_from_edges(geometry, j, i);
            }
        }

        geometry.sampleMappings = decode_disp_sample_mappings(sourceMap, *dispInfo, geometry.lightmap);

        const float stepSampleU = 1.0f / static_cast<float>(geometry.lightmap.width);
        const float stepSampleV = 1.0f / static_cast<float>(geometry.lightmap.height);
        const float halfStepU = stepSampleU * 0.5f;
        const float halfStepV = stepSampleV * 0.5f;

        std::vector<Common::Vec3f> worldPoints((geometry.lightmap.width + 1) * (geometry.lightmap.height + 1));
        for (size_t y = 0; y < geometry.lightmap.height + 1; ++y) {
            for (size_t x = 0; x < geometry.lightmap.width + 1; ++x) {
                Common::Vec3f point;
                const float u = static_cast<float>(x) * stepSampleU;
                const float v = static_cast<float>(y) * stepSampleV;
                if (!disp_uv_to_surf_point(geometry, u, v, 0.0f, point)) {
                    point = Common::make_vec3(0.0f, 0.0f, 0.0f);
                }
                worldPoints[y * (geometry.lightmap.width + 1) + x] = point;
            }
        }

        geometry.samples.resize(geometry.lightmap.sample_count());
        for (size_t y = 0; y < geometry.lightmap.height; ++y) {
            for (size_t x = 0; x < geometry.lightmap.width; ++x) {
                const size_t sampleIndex = y * geometry.lightmap.width + x;
                Common::ReceiverSample& sample = geometry.samples[sampleIndex];
                sample.s = x;
                sample.t = y;
                sample.coord = Common::make_vec2(
                    static_cast<float>(x) * stepSampleU + halfStepU,
                    static_cast<float>(y) * stepSampleV + halfStepV
                );
                sample.mins = Common::make_vec2(
                    static_cast<float>(x) * stepSampleU,
                    static_cast<float>(y) * stepSampleV
                );
                sample.maxs = Common::make_vec2(
                    static_cast<float>(x + 1) * stepSampleU,
                    static_cast<float>(y + 1) * stepSampleV
                );

                const Common::Vec3f& p0 = worldPoints[y * (geometry.lightmap.width + 1) + x];
                const Common::Vec3f& p1 = worldPoints[(y + 1) * (geometry.lightmap.width + 1) + x];
                const Common::Vec3f& p2 = worldPoints[(y + 1) * (geometry.lightmap.width + 1) + (x + 1)];
                const Common::Vec3f& p3 = worldPoints[y * (geometry.lightmap.width + 1) + (x + 1)];
                sample.area = 0.5f * Common::length(Common::cross(Common::sub(p1, p0), Common::sub(p2, p0)))
                    + 0.5f * Common::length(Common::cross(Common::sub(p2, p0), Common::sub(p3, p0)));

                if (!disp_uv_to_surf_point(geometry, sample.coord.x, sample.coord.y, 1.0f, sample.pos)) {
                    sample.pos = Common::make_vec3(0.0f, 0.0f, 0.0f);
                }
                if (!disp_uv_to_surf_normal(geometry, sample.coord.x, sample.coord.y, sample.normal)) {
                    sample.normal = geometry.faceNormal;
                }
            }
        }

        const float stepLuxelU = geometry.lightmap.width > 1
            ? 1.0f / static_cast<float>(geometry.lightmap.width - 1)
            : 0.0f;
        const float stepLuxelV = geometry.lightmap.height > 1
            ? 1.0f / static_cast<float>(geometry.lightmap.height - 1)
            : 0.0f;

        geometry.luxels.resize(geometry.lightmap.sample_count());
        for (size_t y = 0; y < geometry.lightmap.height; ++y) {
            for (size_t x = 0; x < geometry.lightmap.width; ++x) {
                const size_t luxelIndex = y * geometry.lightmap.width + x;
                Common::ReceiverLuxel& luxel = geometry.luxels[luxelIndex];
                luxel.s = x;
                luxel.t = y;
                Common::Vec3f point;
                if (!disp_uv_to_surf_point(
                    geometry,
                    static_cast<float>(x) * stepLuxelU,
                    static_cast<float>(y) * stepLuxelV,
                    1.0f,
                    point
                )) {
                    point = Common::make_vec3(0.0f, 0.0f, 0.0f);
                }
                luxel.pos = point;
            }
        }

        geometry.valid = true;
        return geometry;
    }
}
