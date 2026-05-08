#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cuda_runtime.h"

#include "bsp.h"

namespace VradFaceGeometry {
    inline bool vec3_equal(
        const BSP::Vec3<float>& a,
        const BSP::Vec3<float>& b
    )
    {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    struct Sample {
        int s = 0;
        int t = 0;
        float2 coord = make_float2(0.0f, 0.0f);
        float2 mins = make_float2(0.0f, 0.0f);
        float2 maxs = make_float2(0.0f, 0.0f);
        float3 pos = make_float3(0.0f, 0.0f, 0.0f);
        float area = 0.0f;
    };

    struct FaceGeometry {
        bool valid = false;
        bool isDisplacement = false;
        size_t faceIndex = 0;
        int width = 0;
        int height = 0;
        float worldAreaPerLuxel = 0.0f;
        float3 faceNormal = make_float3(0.0f, 0.0f, 1.0f);
        float faceDist = 0.0f;
        float3 modelOrg = make_float3(0.0f, 0.0f, 0.0f);
        float3 luxelOrigin = make_float3(0.0f, 0.0f, 0.0f);
        float3 worldToLuxelSpace[2] = {
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 0.0f),
        };
        float3 luxelToWorldSpace[2] = {
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 0.0f),
        };
        std::vector<float2> faceWindingLuxel;
        std::vector<Sample> samples;
        std::vector<float3> luxels;
    };

    inline float3 operator+(const BSP::Vec3<float>& a, const float3& b)
    {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline float3 make_float3_from_vec3(const BSP::Vec3<float>& v)
    {
        return make_float3(v.x, v.y, v.z);
    }

    inline float3 add3(float3 a, float3 b)
    {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline float3 sub3(float3 a, float3 b)
    {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    inline float3 scale3(float3 v, float scalar)
    {
        return make_float3(v.x * scalar, v.y * scalar, v.z * scalar);
    }

    inline float length3(float3 v)
    {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    inline float2 make_float2_from_vec(float x, float y)
    {
        return make_float2(x, y);
    }

    inline float cross2(float2 a, float2 b)
    {
        return a.x * b.y - a.y * b.x;
    }

    inline float dot3(float3 a, float3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline float3 mul_add(float3 a, float scalar, float3 b)
    {
        return make_float3(
            a.x + scalar * b.x,
            a.y + scalar * b.y,
            a.z + scalar * b.z
        );
    }

    inline float3 face_model_origin(
        const BSP::BSP& bsp,
        size_t faceIndex
    )
    {
        for (const BSP::DModel& model : bsp.get_models()) {
            const size_t firstFace = static_cast<size_t>(model.firstFace);
            const size_t numFaces = static_cast<size_t>(model.numFaces);
            if (faceIndex >= firstFace && faceIndex < firstFace + numFaces) {
                return make_float3_from_vec3(model.origin);
            }
        }

        return make_float3(0.0f, 0.0f, 0.0f);
    }

    inline std::vector<float3> face_world_winding(
        const BSP::BSP& bsp,
        size_t faceIndex,
        float3 modelOrg
    )
    {
        std::vector<float3> verts;
        const std::vector<BSP::Edge>& edges = bsp.get_faces()[faceIndex].get_edges();
        if (edges.size() < 3) {
            return verts;
        }

        verts.reserve(edges.size());
        verts.push_back(edges[0].vertex1 + modelOrg);

        BSP::Vec3<float> current = edges[0].vertex2;
        verts.push_back(current + modelOrg);

        for (size_t ei = 1; ei < edges.size(); ++ei) {
            const BSP::Edge& edge = edges[ei];
            BSP::Vec3<float> next;

            if (vec3_equal(edge.vertex1, current)) {
                next = edge.vertex2;
            }
            else if (vec3_equal(edge.vertex2, current)) {
                next = edge.vertex1;
            }
            else {
                next = edge.vertex1;
            }

            if (ei + 1 < edges.size()) {
                verts.push_back(next + modelOrg);
            }

            current = next;
        }

        return verts;
    }

    inline void world_to_luxel_space(
        const FaceGeometry& g,
        float3 world,
        float2& coord,
        const BSP::DFace& face
    )
    {
        float3 pos = sub3(world, g.luxelOrigin);
        coord.x = dot3(pos, g.worldToLuxelSpace[0])
            - static_cast<float>(face.lightmapTextureMinsInLuxels[0]);
        coord.y = dot3(pos, g.worldToLuxelSpace[1])
            - static_cast<float>(face.lightmapTextureMinsInLuxels[1]);
    }

    inline float3 luxel_space_to_world(
        const FaceGeometry& g,
        float s,
        float t,
        const BSP::DFace& face
    )
    {
        s += static_cast<float>(face.lightmapTextureMinsInLuxels[0]);
        t += static_cast<float>(face.lightmapTextureMinsInLuxels[1]);

        float3 pos = mul_add(g.luxelOrigin, s, g.luxelToWorldSpace[0]);
        return mul_add(pos, t, g.luxelToWorldSpace[1]);
    }

    inline std::vector<float2> clip_poly_min_x(
        const std::vector<float2>& poly,
        float minX
    )
    {
        std::vector<float2> out;
        if (poly.empty()) {
            return out;
        }

        for (size_t i = 0; i < poly.size(); ++i) {
            const float2 a = poly[i];
            const float2 b = poly[(i + 1) % poly.size()];
            const bool aInside = a.x >= minX;
            const bool bInside = b.x >= minX;

            if (aInside && bInside) {
                out.push_back(b);
            }
            else if (aInside && !bInside) {
                float dx = b.x - a.x;
                if (std::fabs(dx) > 1e-6f) {
                    float t = (minX - a.x) / dx;
                    out.push_back(make_float2(
                        minX,
                        a.y + (b.y - a.y) * t
                    ));
                }
            }
            else if (!aInside && bInside) {
                float dx = b.x - a.x;
                if (std::fabs(dx) > 1e-6f) {
                    float t = (minX - a.x) / dx;
                    out.push_back(make_float2(
                        minX,
                        a.y + (b.y - a.y) * t
                    ));
                }
                out.push_back(b);
            }
        }

        return out;
    }

    inline std::vector<float2> clip_poly_max_x(
        const std::vector<float2>& poly,
        float maxX
    )
    {
        std::vector<float2> out;
        if (poly.empty()) {
            return out;
        }

        for (size_t i = 0; i < poly.size(); ++i) {
            const float2 a = poly[i];
            const float2 b = poly[(i + 1) % poly.size()];
            const bool aInside = a.x <= maxX;
            const bool bInside = b.x <= maxX;

            if (aInside && bInside) {
                out.push_back(b);
            }
            else if (aInside && !bInside) {
                float dx = b.x - a.x;
                if (std::fabs(dx) > 1e-6f) {
                    float t = (maxX - a.x) / dx;
                    out.push_back(make_float2(
                        maxX,
                        a.y + (b.y - a.y) * t
                    ));
                }
            }
            else if (!aInside && bInside) {
                float dx = b.x - a.x;
                if (std::fabs(dx) > 1e-6f) {
                    float t = (maxX - a.x) / dx;
                    out.push_back(make_float2(
                        maxX,
                        a.y + (b.y - a.y) * t
                    ));
                }
                out.push_back(b);
            }
        }

        return out;
    }

    inline std::vector<float2> clip_poly_min_y(
        const std::vector<float2>& poly,
        float minY
    )
    {
        std::vector<float2> out;
        if (poly.empty()) {
            return out;
        }

        for (size_t i = 0; i < poly.size(); ++i) {
            const float2 a = poly[i];
            const float2 b = poly[(i + 1) % poly.size()];
            const bool aInside = a.y >= minY;
            const bool bInside = b.y >= minY;

            if (aInside && bInside) {
                out.push_back(b);
            }
            else if (aInside && !bInside) {
                float dy = b.y - a.y;
                if (std::fabs(dy) > 1e-6f) {
                    float t = (minY - a.y) / dy;
                    out.push_back(make_float2(
                        a.x + (b.x - a.x) * t,
                        minY
                    ));
                }
            }
            else if (!aInside && bInside) {
                float dy = b.y - a.y;
                if (std::fabs(dy) > 1e-6f) {
                    float t = (minY - a.y) / dy;
                    out.push_back(make_float2(
                        a.x + (b.x - a.x) * t,
                        minY
                    ));
                }
                out.push_back(b);
            }
        }

        return out;
    }

    inline std::vector<float2> clip_poly_max_y(
        const std::vector<float2>& poly,
        float maxY
    )
    {
        std::vector<float2> out;
        if (poly.empty()) {
            return out;
        }

        for (size_t i = 0; i < poly.size(); ++i) {
            const float2 a = poly[i];
            const float2 b = poly[(i + 1) % poly.size()];
            const bool aInside = a.y <= maxY;
            const bool bInside = b.y <= maxY;

            if (aInside && bInside) {
                out.push_back(b);
            }
            else if (aInside && !bInside) {
                float dy = b.y - a.y;
                if (std::fabs(dy) > 1e-6f) {
                    float t = (maxY - a.y) / dy;
                    out.push_back(make_float2(
                        a.x + (b.x - a.x) * t,
                        maxY
                    ));
                }
            }
            else if (!aInside && bInside) {
                float dy = b.y - a.y;
                if (std::fabs(dy) > 1e-6f) {
                    float t = (maxY - a.y) / dy;
                    out.push_back(make_float2(
                        a.x + (b.x - a.x) * t,
                        maxY
                    ));
                }
                out.push_back(b);
            }
        }

        return out;
    }

    inline std::vector<float2> clip_poly_rect(
        const std::vector<float2>& poly,
        float minX,
        float maxX,
        float minY,
        float maxY
    )
    {
        std::vector<float2> out = clip_poly_min_x(poly, minX);
        out = clip_poly_max_x(out, maxX);
        out = clip_poly_min_y(out, minY);
        out = clip_poly_max_y(out, maxY);
        return out;
    }

    inline float polygon_area_and_centroid(
        const std::vector<float2>& poly,
        float2& centroid
    )
    {
        centroid = make_float2(0.0f, 0.0f);
        if (poly.size() < 3) {
            return 0.0f;
        }

        float twiceArea = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;

        for (size_t i = 0; i < poly.size(); ++i) {
            const float2 a = poly[i];
            const float2 b = poly[(i + 1) % poly.size()];
            const float cross = cross2(a, b);
            twiceArea += cross;
            cx += (a.x + b.x) * cross;
            cy += (a.y + b.y) * cross;
        }

        if (std::fabs(twiceArea) <= 1e-6f) {
            return 0.0f;
        }

        centroid.x = cx / (3.0f * twiceArea);
        centroid.y = cy / (3.0f * twiceArea);

        return std::fabs(twiceArea) * 0.5f;
    }

    inline void polygon_bounds(
        const std::vector<float2>& poly,
        float2& mins,
        float2& maxs
    )
    {
        mins = make_float2(poly[0].x, poly[0].y);
        maxs = mins;

        for (size_t i = 1; i < poly.size(); ++i) {
            mins.x = std::min(mins.x, poly[i].x);
            mins.y = std::min(mins.y, poly[i].y);
            maxs.x = std::max(maxs.x, poly[i].x);
            maxs.y = std::max(maxs.y, poly[i].y);
        }
    }

    inline void append_fallback_luxel_sample(
        FaceGeometry& out,
        const BSP::DFace& face,
        int s,
        int t
    )
    {
        const float2 center = make_float2(
            static_cast<float>(s) + 0.5f,
            static_cast<float>(t) + 0.5f
        );
        Sample sample;
        sample.s = s;
        sample.t = t;
        sample.coord = center;
        sample.mins = center;
        sample.maxs = center;
        sample.area = out.worldAreaPerLuxel;
        sample.pos = luxel_space_to_world(out, center.x, center.y, face);
        out.samples.push_back(sample);
    }

    inline bool build_face_geometry(
        const BSP::BSP& bsp,
        size_t faceIndex,
        FaceGeometry& out
    )
    {
        out = FaceGeometry{};
        out.faceIndex = faceIndex;

        const BSP::DFace& face = bsp.get_dfaces()[faceIndex];
        out.isDisplacement = face.dispInfo >= 0;
        out.width = face.lightmapTextureSizeInLuxels[0] + 1;
        out.height = face.lightmapTextureSizeInLuxels[1] + 1;

        if (out.width <= 0 || out.height <= 0) {
            return false;
        }

        const BSP::TexInfo& texInfo = bsp.get_texinfos()[face.texInfo];
        const BSP::DPlane& plane = bsp.get_planes()[face.planeNum];

        out.faceNormal = make_float3_from_vec3(plane.normal);
        out.faceDist = plane.dist;
        out.modelOrg = face_model_origin(bsp, faceIndex);

        out.worldToLuxelSpace[0] = make_float3(
            texInfo.lightmapVecs[0][0],
            texInfo.lightmapVecs[0][1],
            texInfo.lightmapVecs[0][2]
        );
        out.worldToLuxelSpace[1] = make_float3(
            texInfo.lightmapVecs[1][0],
            texInfo.lightmapVecs[1][1],
            texInfo.lightmapVecs[1][2]
        );

        const float3 crossVec = make_float3(
            texInfo.lightmapVecs[1][1] * texInfo.lightmapVecs[0][2]
                - texInfo.lightmapVecs[1][2] * texInfo.lightmapVecs[0][1],
            texInfo.lightmapVecs[1][2] * texInfo.lightmapVecs[0][0]
                - texInfo.lightmapVecs[1][0] * texInfo.lightmapVecs[0][2],
            texInfo.lightmapVecs[1][0] * texInfo.lightmapVecs[0][1]
                - texInfo.lightmapVecs[1][1] * texInfo.lightmapVecs[0][0]
        );

        const float det = -dot3(out.faceNormal, crossVec);
        if (std::fabs(det) <= 1e-12f) {
            return false;
        }

        out.luxelToWorldSpace[0] = make_float3(
            (out.faceNormal.z * out.worldToLuxelSpace[1].y
                - out.faceNormal.y * out.worldToLuxelSpace[1].z) / det,
            (out.faceNormal.x * out.worldToLuxelSpace[1].z
                - out.faceNormal.z * out.worldToLuxelSpace[1].x) / det,
            (out.faceNormal.y * out.worldToLuxelSpace[1].x
                - out.faceNormal.x * out.worldToLuxelSpace[1].y) / det
        );
        out.luxelToWorldSpace[1] = make_float3(
            (out.faceNormal.y * out.worldToLuxelSpace[0].z
                - out.faceNormal.z * out.worldToLuxelSpace[0].y) / det,
            (out.faceNormal.z * out.worldToLuxelSpace[0].x
                - out.faceNormal.x * out.worldToLuxelSpace[0].z) / det,
            (out.faceNormal.x * out.worldToLuxelSpace[0].y
                - out.faceNormal.y * out.worldToLuxelSpace[0].x) / det
        );

        out.luxelOrigin = make_float3(
            -(out.faceDist * crossVec.x) / det,
            -(out.faceDist * crossVec.y) / det,
            -(out.faceDist * crossVec.z) / det
        );
        out.luxelOrigin = mul_add(
            out.luxelOrigin,
            -texInfo.lightmapVecs[0][3],
            out.luxelToWorldSpace[0]
        );
        out.luxelOrigin = mul_add(
            out.luxelOrigin,
            -texInfo.lightmapVecs[1][3],
            out.luxelToWorldSpace[1]
        );
        out.luxelOrigin = add3(out.luxelOrigin, out.modelOrg);

        const float lenS = length3(out.worldToLuxelSpace[0]);
        const float lenT = length3(out.worldToLuxelSpace[1]);
        if (lenS <= 1e-12f || lenT <= 1e-12f) {
            return false;
        }
        out.worldAreaPerLuxel = 1.0f / (lenS * lenT);

        const std::vector<float3> worldWinding =
            face_world_winding(bsp, faceIndex, out.modelOrg);
        if (worldWinding.size() < 3) {
            return false;
        }

        out.faceWindingLuxel.reserve(worldWinding.size());
        for (const float3& p : worldWinding) {
            float2 coord;
            world_to_luxel_space(out, p, coord, face);
            out.faceWindingLuxel.push_back(coord);
        }

        out.luxels.reserve(static_cast<size_t>(out.width * out.height));
        for (int t = 0; t < out.height; ++t) {
            for (int s = 0; s < out.width; ++s) {
                out.luxels.push_back(luxel_space_to_world(
                    out,
                    static_cast<float>(s),
                    static_cast<float>(t),
                    face
                ));
            }
        }

        if (!out.isDisplacement) {
            for (int t = 0; t < out.height; ++t) {
                for (int s = 0; s < out.width; ++s) {
                    std::vector<float2> cell = clip_poly_rect(
                        out.faceWindingLuxel,
                        static_cast<float>(s),
                        static_cast<float>(s + 1),
                        static_cast<float>(t),
                        static_cast<float>(t + 1)
                    );

                    if (cell.size() < 3) {
                        append_fallback_luxel_sample(out, face, s, t);
                        continue;
                    }

                    float2 center;
                    float area = polygon_area_and_centroid(cell, center);
                    if (area <= 1e-6f) {
                        append_fallback_luxel_sample(out, face, s, t);
                        continue;
                    }

                    float2 mins;
                    float2 maxs;
                    polygon_bounds(cell, mins, maxs);

                    Sample sample;
                    sample.s = s;
                    sample.t = t;
                    sample.coord = center;
                    sample.mins = mins;
                    sample.maxs = maxs;
                    sample.area = area * out.worldAreaPerLuxel;
                    sample.pos = luxel_space_to_world(
                        out,
                        center.x,
                        center.y,
                        face
                    );
                    out.samples.push_back(sample);
                }
            }
        }

        out.valid = true;
        return true;
    }
}
