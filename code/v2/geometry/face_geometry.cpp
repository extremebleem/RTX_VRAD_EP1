#include "face_geometry.h"

#include <algorithm>
#include <cmath>

namespace SilkRAD::V2::Geometry {
    namespace {
        bool vec3_equal(
            const ::BSP::Vec3<float>& a,
            const ::BSP::Vec3<float>& b
        )
        {
            return a.x == b.x && a.y == b.y && a.z == b.z;
        }

        Common::Vec3f from_bsp_vec3(const ::BSP::Vec3<float>& v)
        {
            return Common::make_vec3(v.x, v.y, v.z);
        }

        Common::Vec3f bsp_plus_vec3(
            const ::BSP::Vec3<float>& a,
            Common::Vec3f b
        )
        {
            return Common::make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
        }

        Common::Vec3f mul_add(
            Common::Vec3f a,
            float scalar,
            Common::Vec3f b
        )
        {
            return Common::make_vec3(
                a.x + scalar * b.x,
                a.y + scalar * b.y,
                a.z + scalar * b.z
            );
        }

        std::vector<Common::Vec2f> clip_poly_min_x(
            const std::vector<Common::Vec2f>& poly,
            float minX
        )
        {
            std::vector<Common::Vec2f> out;
            if (poly.empty()) {
                return out;
            }

            for (size_t i = 0; i < poly.size(); ++i) {
                const Common::Vec2f a = poly[i];
                const Common::Vec2f b = poly[(i + 1) % poly.size()];
                const bool aInside = a.x >= minX;
                const bool bInside = b.x >= minX;

                if (aInside && bInside) {
                    out.push_back(b);
                }
                else if (aInside && !bInside) {
                    const float dx = b.x - a.x;
                    if (std::fabs(dx) > 1e-6f) {
                        const float t = (minX - a.x) / dx;
                        out.push_back(Common::make_vec2(
                            minX,
                            a.y + (b.y - a.y) * t
                        ));
                    }
                }
                else if (!aInside && bInside) {
                    const float dx = b.x - a.x;
                    if (std::fabs(dx) > 1e-6f) {
                        const float t = (minX - a.x) / dx;
                        out.push_back(Common::make_vec2(
                            minX,
                            a.y + (b.y - a.y) * t
                        ));
                    }
                    out.push_back(b);
                }
            }

            return out;
        }

        std::vector<Common::Vec2f> clip_poly_max_x(
            const std::vector<Common::Vec2f>& poly,
            float maxX
        )
        {
            std::vector<Common::Vec2f> out;
            if (poly.empty()) {
                return out;
            }

            for (size_t i = 0; i < poly.size(); ++i) {
                const Common::Vec2f a = poly[i];
                const Common::Vec2f b = poly[(i + 1) % poly.size()];
                const bool aInside = a.x <= maxX;
                const bool bInside = b.x <= maxX;

                if (aInside && bInside) {
                    out.push_back(b);
                }
                else if (aInside && !bInside) {
                    const float dx = b.x - a.x;
                    if (std::fabs(dx) > 1e-6f) {
                        const float t = (maxX - a.x) / dx;
                        out.push_back(Common::make_vec2(
                            maxX,
                            a.y + (b.y - a.y) * t
                        ));
                    }
                }
                else if (!aInside && bInside) {
                    const float dx = b.x - a.x;
                    if (std::fabs(dx) > 1e-6f) {
                        const float t = (maxX - a.x) / dx;
                        out.push_back(Common::make_vec2(
                            maxX,
                            a.y + (b.y - a.y) * t
                        ));
                    }
                    out.push_back(b);
                }
            }

            return out;
        }

        std::vector<Common::Vec2f> clip_poly_min_y(
            const std::vector<Common::Vec2f>& poly,
            float minY
        )
        {
            std::vector<Common::Vec2f> out;
            if (poly.empty()) {
                return out;
            }

            for (size_t i = 0; i < poly.size(); ++i) {
                const Common::Vec2f a = poly[i];
                const Common::Vec2f b = poly[(i + 1) % poly.size()];
                const bool aInside = a.y >= minY;
                const bool bInside = b.y >= minY;

                if (aInside && bInside) {
                    out.push_back(b);
                }
                else if (aInside && !bInside) {
                    const float dy = b.y - a.y;
                    if (std::fabs(dy) > 1e-6f) {
                        const float t = (minY - a.y) / dy;
                        out.push_back(Common::make_vec2(
                            a.x + (b.x - a.x) * t,
                            minY
                        ));
                    }
                }
                else if (!aInside && bInside) {
                    const float dy = b.y - a.y;
                    if (std::fabs(dy) > 1e-6f) {
                        const float t = (minY - a.y) / dy;
                        out.push_back(Common::make_vec2(
                            a.x + (b.x - a.x) * t,
                            minY
                        ));
                    }
                    out.push_back(b);
                }
            }

            return out;
        }

        std::vector<Common::Vec2f> clip_poly_max_y(
            const std::vector<Common::Vec2f>& poly,
            float maxY
        )
        {
            std::vector<Common::Vec2f> out;
            if (poly.empty()) {
                return out;
            }

            for (size_t i = 0; i < poly.size(); ++i) {
                const Common::Vec2f a = poly[i];
                const Common::Vec2f b = poly[(i + 1) % poly.size()];
                const bool aInside = a.y <= maxY;
                const bool bInside = b.y <= maxY;

                if (aInside && bInside) {
                    out.push_back(b);
                }
                else if (aInside && !bInside) {
                    const float dy = b.y - a.y;
                    if (std::fabs(dy) > 1e-6f) {
                        const float t = (maxY - a.y) / dy;
                        out.push_back(Common::make_vec2(
                            a.x + (b.x - a.x) * t,
                            maxY
                        ));
                    }
                }
                else if (!aInside && bInside) {
                    const float dy = b.y - a.y;
                    if (std::fabs(dy) > 1e-6f) {
                        const float t = (maxY - a.y) / dy;
                        out.push_back(Common::make_vec2(
                            a.x + (b.x - a.x) * t,
                            maxY
                        ));
                    }
                    out.push_back(b);
                }
            }

            return out;
        }

        std::vector<Common::Vec2f> clip_poly_rect(
            const std::vector<Common::Vec2f>& poly,
            float minX,
            float maxX,
            float minY,
            float maxY
        )
        {
            std::vector<Common::Vec2f> out = clip_poly_min_x(poly, minX);
            out = clip_poly_max_x(out, maxX);
            out = clip_poly_min_y(out, minY);
            out = clip_poly_max_y(out, maxY);
            return out;
        }

        float polygon_area_and_centroid(
            const std::vector<Common::Vec2f>& poly,
            Common::Vec2f& centroid
        )
        {
            centroid = Common::make_vec2(0.0f, 0.0f);
            if (poly.size() < 3) {
                return 0.0f;
            }

            float twiceArea = 0.0f;
            float cx = 0.0f;
            float cy = 0.0f;
            for (size_t i = 0; i < poly.size(); ++i) {
                const Common::Vec2f a = poly[i];
                const Common::Vec2f b = poly[(i + 1) % poly.size()];
                const float cp = Common::cross(a, b);
                twiceArea += cp;
                cx += (a.x + b.x) * cp;
                cy += (a.y + b.y) * cp;
            }

            if (std::fabs(twiceArea) <= 1e-6f) {
                return 0.0f;
            }

            centroid.x = cx / (3.0f * twiceArea);
            centroid.y = cy / (3.0f * twiceArea);
            return std::fabs(twiceArea) * 0.5f;
        }

        void polygon_bounds(
            const std::vector<Common::Vec2f>& poly,
            Common::Vec2f& mins,
            Common::Vec2f& maxs
        )
        {
            mins = poly[0];
            maxs = poly[0];
            for (size_t i = 1; i < poly.size(); ++i) {
                mins.x = std::min(mins.x, poly[i].x);
                mins.y = std::min(mins.y, poly[i].y);
                maxs.x = std::max(maxs.x, poly[i].x);
                maxs.y = std::max(maxs.y, poly[i].y);
            }
        }

        void append_fallback_luxel_sample(
            FaceGeometry& geometry,
            const ::BSP::DFace& face,
            int s,
            int t
        )
        {
            Common::ReceiverSample sample;
            sample.s = static_cast<size_t>(s);
            sample.t = static_cast<size_t>(t);
            sample.coord = Common::make_vec2(
                static_cast<float>(s) + 0.5f,
                static_cast<float>(t) + 0.5f
            );
            sample.mins = sample.coord;
            sample.maxs = sample.coord;
            sample.area = geometry.worldAreaPerLuxel;
            sample.pos = luxel_space_to_world(
                geometry,
                sample.coord.x,
                sample.coord.y,
                face
            );
            sample.normal = geometry.faceNormal;
            geometry.samples.values.push_back(sample);
        }
    }

    Common::Vec3f face_model_origin(
        const BSP::SourceMap& sourceMap,
        size_t faceIndex
    )
    {
        for (size_t modelIndex = 0; modelIndex < sourceMap.model_count(); ++modelIndex) {
            const ::BSP::DModel& model = sourceMap.model(modelIndex);
            const size_t firstFace = static_cast<size_t>(model.firstFace);
            const size_t numFaces = static_cast<size_t>(model.numFaces);
            if (faceIndex >= firstFace && faceIndex < firstFace + numFaces) {
                return from_bsp_vec3(model.origin);
            }
        }

        return Common::make_vec3(0.0f, 0.0f, 0.0f);
    }

    std::vector<Common::Vec3f> face_world_winding(
        const BSP::SourceMap& sourceMap,
        size_t faceIndex,
        Common::Vec3f modelOrigin
    )
    {
        std::vector<Common::Vec3f> verts;
        const std::vector<::BSP::Edge>& edges = sourceMap.face(faceIndex).get_edges();
        if (edges.size() < 3) {
            return verts;
        }

        verts.reserve(edges.size());
        verts.push_back(bsp_plus_vec3(edges[0].vertex1, modelOrigin));

        ::BSP::Vec3<float> current = edges[0].vertex2;
        verts.push_back(bsp_plus_vec3(current, modelOrigin));

        for (size_t edgeIndex = 1; edgeIndex < edges.size(); ++edgeIndex) {
            const ::BSP::Edge& edge = edges[edgeIndex];
            ::BSP::Vec3<float> next;

            if (vec3_equal(edge.vertex1, current)) {
                next = edge.vertex2;
            }
            else if (vec3_equal(edge.vertex2, current)) {
                next = edge.vertex1;
            }
            else {
                next = edge.vertex1;
            }

            if (edgeIndex + 1 < edges.size()) {
                verts.push_back(bsp_plus_vec3(next, modelOrigin));
            }

            current = next;
        }

        return verts;
    }

    void world_to_luxel_space(
        const FaceGeometry& geometry,
        Common::Vec3f world,
        Common::Vec2f& coord,
        const ::BSP::DFace& face
    )
    {
        const Common::Vec3f pos = Common::sub(world, geometry.luxelOrigin);
        coord.x = Common::dot(pos, geometry.worldToLuxelSpace[0])
            - static_cast<float>(face.lightmapTextureMinsInLuxels[0]);
        coord.y = Common::dot(pos, geometry.worldToLuxelSpace[1])
            - static_cast<float>(face.lightmapTextureMinsInLuxels[1]);
    }

    Common::Vec3f luxel_space_to_world(
        const FaceGeometry& geometry,
        float s,
        float t,
        const ::BSP::DFace& face
    )
    {
        s += static_cast<float>(face.lightmapTextureMinsInLuxels[0]);
        t += static_cast<float>(face.lightmapTextureMinsInLuxels[1]);

        Common::Vec3f pos = mul_add(geometry.luxelOrigin, s, geometry.luxelToWorldSpace[0]);
        return mul_add(pos, t, geometry.luxelToWorldSpace[1]);
    }

    FaceGeometry build_face_geometry(
        const BSP::SourceMap& sourceMap,
        size_t faceIndex
    )
    {
        const ::BSP::DFace& face = sourceMap.dface(faceIndex);
        FaceGeometry geometry;
        geometry.key.faceIndex = faceIndex;
        geometry.isDisplacement = face.dispInfo >= 0;
        geometry.lightmap.width = static_cast<size_t>(face.lightmapTextureSizeInLuxels[0] + 1);
        geometry.lightmap.height = static_cast<size_t>(face.lightmapTextureSizeInLuxels[1] + 1);

        if (geometry.lightmap.width == 0 || geometry.lightmap.height == 0) {
            return geometry;
        }

        const ::BSP::TexInfo& texInfo = sourceMap.texinfo(static_cast<size_t>(face.texInfo));
        const ::BSP::DPlane& plane = sourceMap.plane(static_cast<size_t>(face.planeNum));

        geometry.faceNormal = from_bsp_vec3(plane.normal);
        geometry.faceDist = plane.dist;
        geometry.modelOrigin = face_model_origin(sourceMap, faceIndex);

        geometry.worldToLuxelSpace[0] = Common::make_vec3(
            texInfo.lightmapVecs[0][0],
            texInfo.lightmapVecs[0][1],
            texInfo.lightmapVecs[0][2]
        );
        geometry.worldToLuxelSpace[1] = Common::make_vec3(
            texInfo.lightmapVecs[1][0],
            texInfo.lightmapVecs[1][1],
            texInfo.lightmapVecs[1][2]
        );

        const Common::Vec3f luxelSpaceCross = Common::make_vec3(
            texInfo.lightmapVecs[1][1] * texInfo.lightmapVecs[0][2]
                - texInfo.lightmapVecs[1][2] * texInfo.lightmapVecs[0][1],
            texInfo.lightmapVecs[1][2] * texInfo.lightmapVecs[0][0]
                - texInfo.lightmapVecs[1][0] * texInfo.lightmapVecs[0][2],
            texInfo.lightmapVecs[1][0] * texInfo.lightmapVecs[0][1]
                - texInfo.lightmapVecs[1][1] * texInfo.lightmapVecs[0][0]
        );

        const float det = -Common::dot(geometry.faceNormal, luxelSpaceCross);
        if (std::fabs(det) <= 1e-12f) {
            return geometry;
        }

        geometry.luxelToWorldSpace[0] = Common::make_vec3(
            (geometry.faceNormal.z * geometry.worldToLuxelSpace[1].y
                - geometry.faceNormal.y * geometry.worldToLuxelSpace[1].z) / det,
            (geometry.faceNormal.x * geometry.worldToLuxelSpace[1].z
                - geometry.faceNormal.z * geometry.worldToLuxelSpace[1].x) / det,
            (geometry.faceNormal.y * geometry.worldToLuxelSpace[1].x
                - geometry.faceNormal.x * geometry.worldToLuxelSpace[1].y) / det
        );
        geometry.luxelToWorldSpace[1] = Common::make_vec3(
            (geometry.faceNormal.y * geometry.worldToLuxelSpace[0].z
                - geometry.faceNormal.z * geometry.worldToLuxelSpace[0].y) / det,
            (geometry.faceNormal.z * geometry.worldToLuxelSpace[0].x
                - geometry.faceNormal.x * geometry.worldToLuxelSpace[0].z) / det,
            (geometry.faceNormal.x * geometry.worldToLuxelSpace[0].y
                - geometry.faceNormal.y * geometry.worldToLuxelSpace[0].x) / det
        );

        geometry.luxelOrigin = Common::make_vec3(
            -(geometry.faceDist * luxelSpaceCross.x) / det,
            -(geometry.faceDist * luxelSpaceCross.y) / det,
            -(geometry.faceDist * luxelSpaceCross.z) / det
        );
        geometry.luxelOrigin = mul_add(
            geometry.luxelOrigin,
            -texInfo.lightmapVecs[0][3],
            geometry.luxelToWorldSpace[0]
        );
        geometry.luxelOrigin = mul_add(
            geometry.luxelOrigin,
            -texInfo.lightmapVecs[1][3],
            geometry.luxelToWorldSpace[1]
        );
        geometry.luxelOrigin = Common::add(geometry.luxelOrigin, geometry.modelOrigin);

        const float lenS = Common::length(geometry.worldToLuxelSpace[0]);
        const float lenT = Common::length(geometry.worldToLuxelSpace[1]);
        if (lenS <= 1e-12f || lenT <= 1e-12f) {
            return geometry;
        }
        geometry.worldAreaPerLuxel = 1.0f / (lenS * lenT);

        const std::vector<Common::Vec3f> worldWinding =
            face_world_winding(sourceMap, faceIndex, geometry.modelOrigin);
        if (worldWinding.size() < 3) {
            return geometry;
        }

        geometry.faceWindingLuxel.reserve(worldWinding.size());
        for (const Common::Vec3f& point : worldWinding) {
            Common::Vec2f coord;
            world_to_luxel_space(geometry, point, coord, face);
            geometry.faceWindingLuxel.push_back(coord);
        }

        geometry.luxels.values.reserve(geometry.lightmap.sample_count());
        for (size_t t = 0; t < geometry.lightmap.height; ++t) {
            for (size_t s = 0; s < geometry.lightmap.width; ++s) {
                Common::ReceiverLuxel luxel;
                luxel.s = s;
                luxel.t = t;
                luxel.pos = luxel_space_to_world(
                    geometry,
                    static_cast<float>(s),
                    static_cast<float>(t),
                    face
                );
                geometry.luxels.values.push_back(luxel);
            }
        }

        if (!geometry.isDisplacement) {
            for (size_t t = 0; t < geometry.lightmap.height; ++t) {
                for (size_t s = 0; s < geometry.lightmap.width; ++s) {
                    const std::vector<Common::Vec2f> cell = clip_poly_rect(
                        geometry.faceWindingLuxel,
                        static_cast<float>(s),
                        static_cast<float>(s + 1),
                        static_cast<float>(t),
                        static_cast<float>(t + 1)
                    );

                    if (cell.size() < 3) {
                        append_fallback_luxel_sample(
                            geometry,
                            face,
                            static_cast<int>(s),
                            static_cast<int>(t)
                        );
                        continue;
                    }

                    Common::Vec2f center;
                    const float area = polygon_area_and_centroid(cell, center);
                    if (area <= 1e-6f) {
                        append_fallback_luxel_sample(
                            geometry,
                            face,
                            static_cast<int>(s),
                            static_cast<int>(t)
                        );
                        continue;
                    }

                    Common::Vec2f mins;
                    Common::Vec2f maxs;
                    polygon_bounds(cell, mins, maxs);

                    Common::ReceiverSample sample;
                    sample.s = s;
                    sample.t = t;
                    sample.coord = center;
                    sample.mins = mins;
                    sample.maxs = maxs;
                    sample.area = area * geometry.worldAreaPerLuxel;
                    sample.pos = luxel_space_to_world(
                        geometry,
                        center.x,
                        center.y,
                        face
                    );
                    sample.normal = geometry.faceNormal;
                    geometry.samples.values.push_back(sample);
                }
            }
        }

        geometry.valid = true;
        return geometry;
    }
}
