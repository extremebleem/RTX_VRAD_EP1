#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "cudarad.h"

#include "bsp.h"
#include "bsp_shared.h"

#include "cudabsp.h"
#include "cudamatrix.h"
#include "geometry_rules.h"
#include "raytracer.h"
#include "raytracer_optix.h"
#include "vrad_face_geometry.h"

#include "cudautils.h"


#ifndef MAX_LIGHTSTYLES
#define MAX_LIGHTSTYLES 4
#endif

#ifndef NUM_BUMP_VECTS
#define NUM_BUMP_VECTS 3
#endif

enum { NUM_VERTEX_NORMALS = 162 };
enum { NUM_CUBE_SIDES = 6 };

static __device__ const int DWL_FLAGS_INAMBIENTCUBE = 0x0001;
static __device__ const float ON_EPSILON = 0.1f;
static __device__ const float COORD_EXTENT = 16384.0f;
static __device__ const float WORLD_LIGHT_MIN_EMIT_SURFACE = 0.005f;

static __device__ float3 safe_normalized(const float3& v) {
    float vLen = len(v);

    if (vLen <= 1e-20f) {
        return make_float3();
    }

    return v / vLen;
}

static __device__ inline float3 sr_zero3() {
    return make_float3(0.0f, 0.0f, 0.0f);
}

static __device__ inline float3 sr_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __device__ inline float3 sr_mul(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

static __device__ inline float3 sr_mul(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __device__ inline int sr_clampi(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

static __device__ inline float tex_light_to_linear(uint8_t c, int8_t exponent) {
    return float(c) * exp2f(float(exponent));
}

static __device__ inline float3 decode_rgbexp32(BSP::RGBExp32 c) {
    return make_float3(
        tex_light_to_linear(c.r, c.exp),
        tex_light_to_linear(c.g, c.exp),
        tex_light_to_linear(c.b, c.exp)
    );
}

namespace CUDARAD {
    static std::unique_ptr<RayTracer::CUDARayTracer> g_pRayTracer;
    static __device__ RayTracer::CUDARayTracer* g_pDeviceRayTracer;
    static std::unique_ptr<OptixRT::OptixSunLosTracer> g_pOptixTracer;
    static std::vector<OptixRT::Triangle> g_optixTriangles;
    static std::vector<VradFaceGeometry::FaceGeometry> g_hostFaceGeometry;
    struct HostDisplacementSurface {
        bool valid = false;
        size_t faceIndex = 0;
        size_t dispInfoIndex = 0;
        int gridSize = 0;
        int lightmapWidth = 0;
        int lightmapHeight = 0;
        std::vector<float3> vertices;
        std::vector<float3> normals;
    };
    static std::vector<HostDisplacementSurface> g_hostDispSurfaces;
    static std::string g_assetRoot = ".";
    static std::vector<std::string> g_assetSearchRoots;

    static std::string normalize_asset_path(std::string path) {
        for (char& c : path) {
            if (c == '/') {
                c = '\\';
            }
        }

        while (path.size() > 1 && (path.back() == '\\' || path.back() == '/')) {
            path.pop_back();
        }

        return path.empty() ? "." : path;
    }

    static std::string lower_ascii(std::string text) {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
        );
        return text;
    }

    static bool path_ends_with_gameinfo(const std::string& path) {
        const std::string lower = lower_ascii(normalize_asset_path(path));
        const std::string suffix = "\\gameinfo.gi";

        return lower == "gameinfo.gi"
            || (lower.size() >= suffix.size()
                && lower.compare(lower.size() - suffix.size(), suffix.size(), suffix) == 0);
    }

    static std::string dirname_from_path(const std::string& path) {
        const std::string normalized = normalize_asset_path(path);
        const size_t slash = normalized.find_last_of('\\');
        if (slash == std::string::npos) {
            return ".";
        }

        return normalized.substr(0, slash);
    }

    static std::string gameinfo_path_from_root(const std::string& root) {
        return normalize_asset_path(root) + "\\gameinfo.gi";
    }

    static bool read_text_file(const std::string& path, std::string& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }

        std::ostringstream ss;
        ss << f.rdbuf();
        out = ss.str();
        return true;
    }

    static void add_unique_search_root(
        std::vector<std::string>& roots,
        const std::string& root
    ) {
        const std::string normalized = normalize_asset_path(root);
        const std::string lowered = lower_ascii(normalized);

        for (const std::string& existing : roots) {
            if (lower_ascii(existing) == lowered) {
                return;
            }
        }

        roots.push_back(normalized);
    }

    static bool parse_gameinfo_token(
        const std::string& line,
        size_t& offset,
        std::string& token
    ) {
        token.clear();

        while (offset < line.size()
            && (line[offset] == ' ' || line[offset] == '\t')) {
            ++offset;
        }

        if (offset >= line.size()) {
            return false;
        }

        if (line[offset] == '"') {
            ++offset;
            const size_t start = offset;
            while (offset < line.size() && line[offset] != '"') {
                ++offset;
            }
            token = line.substr(start, offset - start);
            if (offset < line.size() && line[offset] == '"') {
                ++offset;
            }
            return true;
        }

        const size_t start = offset;
        while (offset < line.size()
            && line[offset] != ' '
            && line[offset] != '\t'
            && line[offset] != '{'
            && line[offset] != '}') {
            ++offset;
        }

        token = line.substr(start, offset - start);
        return !token.empty();
    }

    static std::vector<std::string> parse_gameinfo_search_roots(
        const std::string& gameInfoPath,
        const std::string& gameRoot
    ) {
        std::vector<std::string> roots;
        add_unique_search_root(roots, gameRoot);

        std::string text;
        if (!read_text_file(gameInfoPath, text)) {
            return roots;
        }

        std::string parentRoot;
        const size_t slash = gameRoot.find_last_of('\\');
        if (slash != std::string::npos) {
            parentRoot = gameRoot.substr(0, slash);
        }

        std::istringstream stream(text);
        std::string line;
        bool inSearchPaths = false;
        int braceDepth = 0;

        while (std::getline(stream, line)) {
            const size_t comment = line.find("//");
            if (comment != std::string::npos) {
                line.resize(comment);
            }

            std::string lower = lower_ascii(line);
            if (!inSearchPaths && lower.find("searchpaths") != std::string::npos) {
                inSearchPaths = true;
            }

            if (!inSearchPaths) {
                continue;
            }

            for (char c : line) {
                if (c == '{') {
                    ++braceDepth;
                }
                else if (c == '}') {
                    --braceDepth;
                }
            }

            if (braceDepth <= 0) {
                if (braceDepth <= 0 && lower.find('}') != std::string::npos) {
                    inSearchPaths = false;
                }
                continue;
            }

            size_t tokenOffset = 0;
            std::string key;
            std::string value;
            if (!parse_gameinfo_token(line, tokenOffset, key)
                || lower_ascii(key) != "game"
                || !parse_gameinfo_token(line, tokenOffset, value)) {
                continue;
            }

            if (value.empty() || value == "|gameinfo_path|.") {
                add_unique_search_root(roots, gameRoot);
                continue;
            }

            const std::string marker = "|gameinfo_path|";
            if (value.find(marker) == 0) {
                add_unique_search_root(roots, gameRoot + "\\" + value.substr(marker.size()));
                continue;
            }

            const std::string allPathsMarker = "|all_source_engine_paths|";
            if (value.find(allPathsMarker) == 0 && !parentRoot.empty()) {
                add_unique_search_root(roots, parentRoot + "\\" + value.substr(allPathsMarker.size()));
                continue;
            }

            if (value.size() > 1 && value[1] == ':') {
                add_unique_search_root(roots, value);
            }
            else {
                add_unique_search_root(roots, gameRoot + "\\" + value);
            }
        }

        return roots;
    }

    void set_asset_root(const std::string& assetRoot) {
        const std::string requested = normalize_asset_path(assetRoot);
        const std::string gameInfoPath =
            path_ends_with_gameinfo(requested)
                ? requested
                : gameinfo_path_from_root(requested);

        g_assetRoot =
            path_ends_with_gameinfo(requested)
                ? dirname_from_path(requested)
                : requested;

        g_assetSearchRoots = parse_gameinfo_search_roots(gameInfoPath, g_assetRoot);

        if (g_assetSearchRoots.empty()) {
            add_unique_search_root(g_assetSearchRoots, g_assetRoot);
        }
    }

    __device__ FaceInfo::FaceInfo() {};

    __device__ FaceInfo::FaceInfo(
            CUDABSP::CUDABSP& cudaBSP,
            size_t faceIndex
            ) :
            faceIndex(faceIndex),
            face(cudaBSP.faces[faceIndex]),
            plane(cudaBSP.planes[face.planeNum]),
            texInfo(cudaBSP.texInfos[face.texInfo]),
            Ainv(cudaBSP.xyzMatrices[faceIndex]),
            faceNorm(
                make_float3(plane.normal.x, plane.normal.y, plane.normal.z)
            ),
            lightmapWidth(face.lightmapTextureSizeInLuxels[0] + 1),
            lightmapHeight(face.lightmapTextureSizeInLuxels[1] + 1),
            lightmapSize(lightmapWidth * lightmapHeight),
            lightmapStartIndex(face.lightOffset / sizeof(BSP::RGBExp32)),
            totalLight(make_float3()) {}

    __device__ float3 FaceInfo::xyz_from_st(float s, float t) {
        float sOffset = this->texInfo.lightmapVecs[0][3];
        float tOffset = this->texInfo.lightmapVecs[1][3];
        float sMin = this->face.lightmapTextureMinsInLuxels[0];
        float tMin = this->face.lightmapTextureMinsInLuxels[1];

        CUDAMatrix::CUDAMatrix<double, 3, 1> B;

        B[0][0] = s - sOffset + sMin;
        B[1][0] = t - tOffset + tMin;
        B[2][0] = this->plane.dist;

        CUDAMatrix::CUDAMatrix<double, 3, 1> result = this->Ainv * B;

        return make_float3(result[0][0], result[1][0], result[2][0]);
    }
}

namespace LeafAmbient {



    static __device__ const float3 BOX_DIRECTIONS[NUM_CUBE_SIDES] = {
        {  1.0f,  0.0f,  0.0f },
        { -1.0f,  0.0f,  0.0f },
        {  0.0f,  1.0f,  0.0f },
        {  0.0f, -1.0f,  0.0f },
        {  0.0f,  0.0f,  1.0f },
        {  0.0f,  0.0f, -1.0f },
    };

    static __device__ const float3 ANORMS[NUM_VERTEX_NORMALS] = {
        { -0.525731f,  0.000000f,  0.850651f },
        { -0.442863f,  0.238856f,  0.864188f },
        { -0.295242f,  0.000000f,  0.955423f },
        { -0.309017f,  0.500000f,  0.809017f },
        { -0.162460f,  0.262866f,  0.951056f },
        {  0.000000f,  0.000000f,  1.000000f },
        {  0.000000f,  0.850651f,  0.525731f },
        { -0.147621f,  0.716567f,  0.681718f },
        {  0.147621f,  0.716567f,  0.681718f },
        {  0.000000f,  0.525731f,  0.850651f },
        {  0.309017f,  0.500000f,  0.809017f },
        {  0.525731f,  0.000000f,  0.850651f },
        {  0.295242f,  0.000000f,  0.955423f },
        {  0.442863f,  0.238856f,  0.864188f },
        {  0.162460f,  0.262866f,  0.951056f },
        { -0.681718f,  0.147621f,  0.716567f },
        { -0.809017f,  0.309017f,  0.500000f },
        { -0.587785f,  0.425325f,  0.688191f },
        { -0.850651f,  0.525731f,  0.000000f },
        { -0.864188f,  0.442863f,  0.238856f },
        { -0.716567f,  0.681718f,  0.147621f },
        { -0.688191f,  0.587785f,  0.425325f },
        { -0.500000f,  0.809017f,  0.309017f },
        { -0.238856f,  0.864188f,  0.442863f },
        { -0.425325f,  0.688191f,  0.587785f },
        { -0.716567f,  0.681718f, -0.147621f },
        { -0.500000f,  0.809017f, -0.309017f },
        { -0.525731f,  0.850651f,  0.000000f },
        {  0.000000f,  0.850651f, -0.525731f },
        { -0.238856f,  0.864188f, -0.442863f },
        {  0.000000f,  0.955423f, -0.295242f },
        { -0.262866f,  0.951056f, -0.162460f },
        {  0.000000f,  1.000000f,  0.000000f },
        {  0.000000f,  0.955423f,  0.295242f },
        { -0.262866f,  0.951056f,  0.162460f },
        {  0.238856f,  0.864188f,  0.442863f },
        {  0.262866f,  0.951056f,  0.162460f },
        {  0.500000f,  0.809017f,  0.309017f },
        {  0.238856f,  0.864188f, -0.442863f },
        {  0.262866f,  0.951056f, -0.162460f },
        {  0.500000f,  0.809017f, -0.309017f },
        {  0.850651f,  0.525731f,  0.000000f },
        {  0.716567f,  0.681718f,  0.147621f },
        {  0.716567f,  0.681718f, -0.147621f },
        {  0.525731f,  0.850651f,  0.000000f },
        {  0.425325f,  0.688191f,  0.587785f },
        {  0.864188f,  0.442863f,  0.238856f },
        {  0.688191f,  0.587785f,  0.425325f },
        {  0.809017f,  0.309017f,  0.500000f },
        {  0.681718f,  0.147621f,  0.716567f },
        {  0.587785f,  0.425325f,  0.688191f },
        {  0.955423f,  0.295242f,  0.000000f },
        {  1.000000f,  0.000000f,  0.000000f },
        {  0.951056f,  0.162460f,  0.262866f },
        {  0.850651f, -0.525731f,  0.000000f },
        {  0.955423f, -0.295242f,  0.000000f },
        {  0.864188f, -0.442863f,  0.238856f },
        {  0.951056f, -0.162460f,  0.262866f },
        {  0.809017f, -0.309017f,  0.500000f },
        {  0.681718f, -0.147621f,  0.716567f },
        {  0.850651f,  0.000000f,  0.525731f },
        {  0.864188f,  0.442863f, -0.238856f },
        {  0.809017f,  0.309017f, -0.500000f },
        {  0.951056f,  0.162460f, -0.262866f },
        {  0.525731f,  0.000000f, -0.850651f },
        {  0.681718f,  0.147621f, -0.716567f },
        {  0.681718f, -0.147621f, -0.716567f },
        {  0.850651f,  0.000000f, -0.525731f },
        {  0.809017f, -0.309017f, -0.500000f },
        {  0.864188f, -0.442863f, -0.238856f },
        {  0.951056f, -0.162460f, -0.262866f },
        {  0.147621f,  0.716567f, -0.681718f },
        {  0.309017f,  0.500000f, -0.809017f },
        {  0.425325f,  0.688191f, -0.587785f },
        {  0.442863f,  0.238856f, -0.864188f },
        {  0.587785f,  0.425325f, -0.688191f },
        {  0.688191f,  0.587785f, -0.425325f },
        { -0.147621f,  0.716567f, -0.681718f },
        { -0.309017f,  0.500000f, -0.809017f },
        {  0.000000f,  0.525731f, -0.850651f },
        { -0.525731f,  0.000000f, -0.850651f },
        { -0.442863f,  0.238856f, -0.864188f },
        { -0.295242f,  0.000000f, -0.955423f },
        { -0.162460f,  0.262866f, -0.951056f },
        {  0.000000f,  0.000000f, -1.000000f },
        {  0.295242f,  0.000000f, -0.955423f },
        {  0.162460f,  0.262866f, -0.951056f },
        { -0.442863f, -0.238856f, -0.864188f },
        { -0.309017f, -0.500000f, -0.809017f },
        { -0.162460f, -0.262866f, -0.951056f },
        {  0.000000f, -0.850651f, -0.525731f },
        { -0.147621f, -0.716567f, -0.681718f },
        {  0.147621f, -0.716567f, -0.681718f },
        {  0.000000f, -0.525731f, -0.850651f },
        {  0.309017f, -0.500000f, -0.809017f },
        {  0.442863f, -0.238856f, -0.864188f },
        {  0.162460f, -0.262866f, -0.951056f },
        {  0.238856f, -0.864188f, -0.442863f },
        {  0.500000f, -0.809017f, -0.309017f },
        {  0.425325f, -0.688191f, -0.587785f },
        {  0.716567f, -0.681718f, -0.147621f },
        {  0.688191f, -0.587785f, -0.425325f },
        {  0.587785f, -0.425325f, -0.688191f },
        {  0.000000f, -0.955423f, -0.295242f },
        {  0.000000f, -1.000000f,  0.000000f },
        {  0.262866f, -0.951056f, -0.162460f },
        {  0.000000f, -0.850651f,  0.525731f },
        {  0.000000f, -0.955423f,  0.295242f },
        {  0.238856f, -0.864188f,  0.442863f },
        {  0.262866f, -0.951056f,  0.162460f },
        {  0.500000f, -0.809017f,  0.309017f },
        {  0.716567f, -0.681718f,  0.147621f },
        {  0.525731f, -0.850651f,  0.000000f },
        { -0.238856f, -0.864188f, -0.442863f },
        { -0.500000f, -0.809017f, -0.309017f },
        { -0.262866f, -0.951056f, -0.162460f },
        { -0.850651f, -0.525731f,  0.000000f },
        { -0.716567f, -0.681718f, -0.147621f },
        { -0.716567f, -0.681718f,  0.147621f },
        { -0.525731f, -0.850651f,  0.000000f },
        { -0.500000f, -0.809017f,  0.309017f },
        { -0.238856f, -0.864188f,  0.442863f },
        { -0.262866f, -0.951056f,  0.162460f },
        { -0.864188f, -0.442863f,  0.238856f },
        { -0.809017f, -0.309017f,  0.500000f },
        { -0.688191f, -0.587785f,  0.425325f },
        { -0.681718f, -0.147621f,  0.716567f },
        { -0.442863f, -0.238856f,  0.864188f },
        { -0.587785f, -0.425325f,  0.688191f },
        { -0.309017f, -0.500000f,  0.809017f },
        { -0.147621f, -0.716567f,  0.681718f },
        { -0.425325f, -0.688191f,  0.587785f },
        { -0.162460f, -0.262866f,  0.951056f },
        {  0.442863f, -0.238856f,  0.864188f },
        {  0.162460f, -0.262866f,  0.951056f },
        {  0.309017f, -0.500000f,  0.809017f },
        {  0.147621f, -0.716567f,  0.681718f },
        {  0.000000f, -0.525731f,  0.850651f },
        {  0.425325f, -0.688191f,  0.587785f },
        {  0.587785f, -0.425325f,  0.688191f },
        {  0.688191f, -0.587785f,  0.425325f },
        { -0.955423f,  0.295242f,  0.000000f },
        { -0.951056f,  0.162460f,  0.262866f },
        { -1.000000f,  0.000000f,  0.000000f },
        { -0.850651f,  0.000000f,  0.525731f },
        { -0.955423f, -0.295242f,  0.000000f },
        { -0.951056f, -0.162460f,  0.262866f },
        { -0.864188f,  0.442863f, -0.238856f },
        { -0.951056f,  0.162460f, -0.262866f },
        { -0.809017f,  0.309017f, -0.500000f },
        { -0.864188f, -0.442863f, -0.238856f },
        { -0.951056f, -0.162460f, -0.262866f },
        { -0.809017f, -0.309017f, -0.500000f },
        { -0.681718f,  0.147621f, -0.716567f },
        { -0.681718f, -0.147621f, -0.716567f },
        { -0.850651f,  0.000000f, -0.525731f },
        { -0.688191f,  0.587785f, -0.425325f },
        { -0.587785f,  0.425325f, -0.688191f },
        { -0.425325f,  0.688191f, -0.587785f },
        { -0.425325f, -0.688191f, -0.587785f },
        { -0.587785f, -0.425325f, -0.688191f },
        { -0.688191f, -0.587785f, -0.425325f },
    };

    static __device__ float length_squared(const float3& v) {
        return dot(v, v);
    }

    static __device__ float3 component_multiply(
            const float3& a,
            const float3& b
            ) {
        return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    static __device__ float3 find_sky_ambient(
        const CUDABSP::CUDABSP& cudaBSP
    ) {
        for (size_t i = 0; i < cudaBSP.numWorldLights; ++i) {
            const BSP::DWorldLight& light = cudaBSP.worldLights[i];

            if (light.type == BSP::EMIT_SKYAMBIENT && light.style == 0) {
                return make_float3(light.intensity);
            }
        }

        return make_float3();
    }

    static __device__ float3 compute_lightmap_color_from_average(
            const CUDABSP::CUDABSP& cudaBSP,
            size_t faceIndex,
            const float3& skyAmbient
            ) {
        if (faceIndex >= cudaBSP.numFaces) {
            return make_float3();
        }

        const BSP::DFace& face = cudaBSP.faces[faceIndex];

        if (face.texInfo < 0
                || static_cast<size_t>(face.texInfo) >= cudaBSP.numTexInfos) {
            return make_float3();
        }

        const BSP::TexInfo& texInfo = cudaBSP.texInfos[face.texInfo];

        if (texInfo.flags & BSP::SURF_SKY) {
            return skyAmbient;
        }

        if (face.lightOffset <= 0 || face.styles[0] == 255) {
            return make_float3();
        }

        size_t lightmapStart =
            static_cast<size_t>(face.lightOffset) / sizeof(BSP::RGBExp32);

        if (lightmapStart == 0 || lightmapStart > cudaBSP.numLightSamples) {
            return make_float3();
        }

        float3 color = cudaBSP.lightSamples[lightmapStart - 1];

        if (texInfo.texData >= 0
                && static_cast<size_t>(texInfo.texData)
                    < cudaBSP.numTexDatas) {
            const BSP::DTexData& texData = cudaBSP.texDatas[texInfo.texData];
            color = component_multiply(color, make_float3(texData.reflectivity));
        }

        return color;
    }

    static __device__ float3 compute_ambient_from_surface(
        const CUDABSP::CUDABSP& cudaBSP,
        int faceId,
        const float3& skyAmbient,
        const float3& radcolor
    ) {
        const BSP::DFace& face = cudaBSP.faces[faceId];
        const BSP::TexInfo& texInfo = cudaBSP.texInfos[face.texInfo];

        if (texInfo.flags & BSP::SURF_SKY) {
            return skyAmbient;
        }

        const BSP::DTexData& texData = cudaBSP.texDatas[texInfo.texData];

        return sr_mul(radcolor, make_float3(
            texData.reflectivity.x,
            texData.reflectivity.y,
            texData.reflectivity.z
        ));
    }

    static __device__ float3 compute_lightmap_color_from_average_fallback(
        const CUDABSP::CUDABSP& cudaBSP,
        int faceId,
        const float3& skyAmbient
    ) {
        const BSP::DFace& face = cudaBSP.faces[faceId];
        const BSP::TexInfo& texInfo = cudaBSP.texInfos[face.texInfo];

        if (texInfo.flags & BSP::SURF_SKY) {
            return skyAmbient;
        }

        if (face.lightOffset < int(sizeof(BSP::RGBExp32))) {
            return sr_zero3();
        }

        // Your SilkRAD path stores average lighting one RGBExp32 before lightOffset.
        int base = face.lightOffset / int(sizeof(BSP::RGBExp32));

        float3 color = cudaBSP.lightSamples[base - 1];
        return compute_ambient_from_surface(cudaBSP, faceId, skyAmbient, color);
    }

    static __device__ bool world_to_face_luxel(
        const CUDABSP::CUDABSP& cudaBSP,
        int faceId,
        const float3& p,
        float2& outLuxel
    ) {
        const BSP::DFace& face = cudaBSP.faces[faceId];
        const BSP::TexInfo& texInfo = cudaBSP.texInfos[face.texInfo];

        if (texInfo.flags & BSP::SURF_NOLIGHT)
            return false;

        float s =
            p.x * texInfo.lightmapVecs[0][0] +
            p.y * texInfo.lightmapVecs[0][1] +
            p.z * texInfo.lightmapVecs[0][2] +
            texInfo.lightmapVecs[0][3];

        float t =
            p.x * texInfo.lightmapVecs[1][0] +
            p.y * texInfo.lightmapVecs[1][1] +
            p.z * texInfo.lightmapVecs[1][2] +
            texInfo.lightmapVecs[1][3];

        outLuxel.x = s - float(face.lightmapTextureMinsInLuxels[0]);
        outLuxel.y = t - float(face.lightmapTextureMinsInLuxels[1]);

        return true;
    }

    static __device__ bool surf_has_bumped_lightmaps(
        const CUDABSP::CUDABSP& cudaBSP,
        int faceId
    ) {
        const BSP::DFace& face = cudaBSP.faces[faceId];
        const BSP::TexInfo& texInfo = cudaBSP.texInfos[face.texInfo];

        return ((texInfo.flags & BSP::SURF_BUMPLIGHT) && !(texInfo.flags & BSP::SURF_NOLIGHT));
    }

    static __device__ float3 compute_lightmap_color_from_luxel(
        const CUDABSP::CUDABSP& cudaBSP,
        int faceId,
        const float2& luxelCoord,
        const float3& skyAmbient
    ) {
        const BSP::DFace& face = cudaBSP.faces[faceId];
        const BSP::TexInfo& texInfo = cudaBSP.texInfos[face.texInfo];

        if (texInfo.flags & BSP::SURF_SKY) {
            return skyAmbient;
        }

        if (face.lightOffset < 0) {
            return sr_zero3();
        }

        int smax = face.lightmapTextureSizeInLuxels[0] + 1;
        int tmax = face.lightmapTextureSizeInLuxels[1] + 1;

        int ds = sr_clampi(int(floorf(luxelCoord.x)), 0, smax - 1);
        int dt = sr_clampi(int(floorf(luxelCoord.y)), 0, tmax - 1);

        int stylePlaneSize = smax * tmax;
        if (surf_has_bumped_lightmaps(cudaBSP, faceId)) {
            stylePlaneSize *= (NUM_BUMP_VECTS + 1);
        }

        // Source dface.lightofs/lightOffset is byte offset into LIGHTING lump.
        // cudaBSP.lightSamples must be BSP::RGBExp32* for this indexing.
        int base = face.lightOffset / int(sizeof(BSP::RGBExp32));
        int luxelIndex = dt * smax + ds;

        float3 colorSum = sr_zero3();

        for (int map = 0; map < MAX_LIGHTSTYLES && face.styles[map] != 255; ++map) {
            int sampleIndex = base + map * stylePlaneSize + luxelIndex;

            float3 color = cudaBSP.lightSamples[sampleIndex];

            color = compute_ambient_from_surface(cudaBSP, faceId, skyAmbient, color);
            colorSum = sr_add(colorSum, color);
        }

        return colorSum;
    }

    static __device__ float3 calc_ray_ambient_lighting(
            const CUDABSP::CUDABSP& cudaBSP,
            const float3& start,
            const float3& end,
            const float3& skyAmbient
            ) {
        RayTracer::RayHit hit = CUDARAD::g_pDeviceRayTracer->trace_closest(start, end);

        if (!hit.hit) {
            return sr_zero3();
        }

        int faceId = int(hit.faceId);

        float2 luxel;
        if (!world_to_face_luxel(cudaBSP, faceId, hit.position, luxel)) {
            return compute_lightmap_color_from_average_fallback(cudaBSP, faceId, skyAmbient);
        }

        return compute_lightmap_color_from_luxel(cudaBSP, faceId, luxel, skyAmbient);
    }

    static __device__ float inv_r_squared(const float3& delta) {
        float distSquared = length_squared(delta);

        if (distSquared <= 1e-20f) {
            return 0.0f;
        }

        return 1.0f / distSquared;
    }

    static __device__ bool is_leaf_ambient_surface_light(
            const BSP::DWorldLight& light
            ) {
        if (light.type != BSP::EMIT_SURFACE || light.style != 0) {
            return false;
        }

        float intensity = fmaxf(
            light.intensity.x,
            fmaxf(light.intensity.y, light.intensity.z)
        );

        return intensity * inv_r_squared(make_float3(0.0f, 0.0f, 512.0f))
            < WORLD_LIGHT_MIN_EMIT_SURFACE;
    }

    static __device__ float engine_world_light_distance_falloff(
            const BSP::DWorldLight& light,
            const float3& delta
            ) {
        if (light.radius != 0.0f
                && length_squared(delta) > light.radius * light.radius) {
            return 0.0f;
        }

        return inv_r_squared(delta);
    }

    static __device__ float engine_world_light_angle(
            const float3& lightNormal,
            const float3& sampleNormal,
            const float3& deltaNormal
            ) {
        float dotSample = dot(sampleNormal, deltaNormal);

        if (dotSample < 0.0f) {
            return 0.0f;
        }

        float dotLight = -dot(deltaNormal, lightNormal);

        if (dotLight <= ON_EPSILON / 10.0f) {
            return 0.0f;
        }

        return dotSample * dotLight;
    }

    static __device__ void add_emit_surface_lights(
            const CUDABSP::CUDABSP& cudaBSP,
            const float3& start,
            float3 lightBoxColor[NUM_CUBE_SIDES]
            ) {

        for (size_t i=0; i<cudaBSP.numWorldLights; ++i) {
            const BSP::DWorldLight& light = cudaBSP.worldLights[i];

            if (light.type != BSP::EMIT_SURFACE) {
                continue;
            }

            if (!(light.flags & DWL_FLAGS_INAMBIENTCUBE)) { // && !is_leaf_ambient_surface_light(light)
                continue;
            }

            float3 lightOrigin = make_float3(light.origin);

            if (CUDARAD::g_pDeviceRayTracer->LOS_blocked(start, lightOrigin)) {
                continue;
            }

            float3 delta = lightOrigin - start;
            float distanceScale =
                engine_world_light_distance_falloff(light, delta);

            if (distanceScale == 0.0f) {
                continue;
            }

            printf("zxc\n");

            float3 deltaNormal = safe_normalized(delta);
            float angleScale = engine_world_light_angle(
                safe_normalized(make_float3(light.normal)),
                deltaNormal,
                deltaNormal
            );

            float ratio = distanceScale * angleScale;

            if (ratio == 0.0f) {
                continue;
            }

            float3 intensity = make_float3(light.intensity);

            for (int side=0; side<NUM_CUBE_SIDES; ++side) {
                float directionScale = dot(BOX_DIRECTIONS[side], deltaNormal);

                if (directionScale > 0.0f) {
                    lightBoxColor[side] +=
                        intensity * (directionScale * ratio);
                }

                printf("side %.1f %.1f %.1f", lightBoxColor[side].x, lightBoxColor[side].y, lightBoxColor[side].z);
            }
        }
    }

    static __device__ void compute_ambient_from_spherical_samples(
            const CUDABSP::CUDABSP& cudaBSP,
            const float3& start,
            const float3& skyAmbient,
            float3 lightBoxColor[NUM_CUBE_SIDES]
            ) {
        float weights[NUM_CUBE_SIDES];

        for (int side=0; side<NUM_CUBE_SIDES; ++side) {
            lightBoxColor[side] = make_float3();
            weights[side] = 0.0f;
        }

        for (int i=0; i<NUM_VERTEX_NORMALS; ++i) {
            float3 direction = ANORMS[i];
            float3 end = start + direction * (COORD_EXTENT * 1.74f);

            float3 radcolor = calc_ray_ambient_lighting(
                cudaBSP, start, end, skyAmbient
            );

            for (int side=0; side<NUM_CUBE_SIDES; ++side) {
                float directionScale = dot(direction, BOX_DIRECTIONS[side]);

                if (directionScale > 0.0f) {
                    weights[side] += directionScale;
                    lightBoxColor[side] += radcolor * directionScale;
                }
            }
        }

        for (int side=0; side<NUM_CUBE_SIDES; ++side) {
            if (weights[side] > 0.0f) {
                lightBoxColor[side] /= weights[side];
            }
        }

        add_emit_surface_lights(cudaBSP, start, lightBoxColor);
    }

    __global__ void compute_leaf_ambient(
        CUDABSP::CUDABSP* pCudaBSP,
        size_t* pLeavesCompleted
    ) {
        size_t leafIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (leafIndex >= pCudaBSP->numLeaves)
            return;

        BSP::CompressedLightCube out;

        for (int side = 0; side < NUM_CUBE_SIDES; ++side) {
            out.color[side] = BSP::RGBExp32{ 0, 0, 0, 0 };
        }

        const BSP::DLeaf& leaf = pCudaBSP->leaves[leafIndex];

        if (!(leaf.contents & BSP::CONTENTS_SOLID)) {
            float3 center = make_float3(
                (float(leaf.mins[0]) + float(leaf.maxs[0])) * 0.5f,
                (float(leaf.mins[1]) + float(leaf.maxs[1])) * 0.5f,
                (float(leaf.mins[2]) + float(leaf.maxs[2])) * 0.5f
            );

            float3 skyAmbient = find_sky_ambient(*pCudaBSP);
            float3 cube[NUM_CUBE_SIDES];

            compute_ambient_from_spherical_samples(
                *pCudaBSP, center, skyAmbient, cube
            );

            for (int side = 0; side < NUM_CUBE_SIDES; ++side) {

                out.color[side] = CUDABSP::rgbexp32_from_float3(cube[side]);
            }
        }

        pCudaBSP->ambientLightSamples[leafIndex] = out;

        atomicAdd(
            reinterpret_cast<unsigned long long*>(pLeavesCompleted),
            1ULL
        );
    }

    void run(CUDABSP::CUDABSP* pCudaBSP) {
        CUDABSP::CUDABSP cudaBSP;

        CUDA_CHECK_ERROR(cudaMemcpy(
            &cudaBSP,
            pCudaBSP,
            sizeof(CUDABSP::CUDABSP),
            cudaMemcpyDeviceToHost
        ));

        volatile size_t* pLeavesCompleted;
        CUDA_CHECK_ERROR(cudaHostAlloc(
            &pLeavesCompleted,
            sizeof(size_t),
            cudaHostAllocMapped
        ));

        *pLeavesCompleted = 0;

        volatile size_t* pDeviceLeavesCompleted;
        CUDA_CHECK_ERROR(cudaHostGetDevicePointer(
            const_cast<size_t**>(&pDeviceLeavesCompleted),
            const_cast<size_t*>(pLeavesCompleted),
            0
        ));

        const size_t BLOCK_WIDTH = 128;
        size_t numBlocks = div_ceil(cudaBSP.numLeaves, BLOCK_WIDTH);

        std::cout << "Launching "
            << cudaBSP.numLeaves << " leaf ambient threads..."
            << std::endl;

        cudaEvent_t startEvent;
        cudaEvent_t stopEvent;

        CUDA_CHECK_ERROR(cudaEventCreate(&startEvent));
        CUDA_CHECK_ERROR(cudaEventCreate(&stopEvent));

        CUDA_CHECK_ERROR(cudaEventRecord(startEvent));

        KERNEL_LAUNCH(
            LeafAmbient::compute_leaf_ambient,
            numBlocks,
            BLOCK_WIDTH,
            pCudaBSP,
            const_cast<size_t*>(pDeviceLeavesCompleted)
        );

        flush_wddm_queue();

        size_t lastLeavesCompleted = 0;
        size_t leavesCompleted = 0;

        do {
            CUDA_CHECK_ERROR(cudaPeekAtLastError());

            leavesCompleted = *pLeavesCompleted;

            if (leavesCompleted > lastLeavesCompleted) {
                std::cout << "    " << leavesCompleted << "/"
                    << cudaBSP.numLeaves
                    << " leaves processed..." << std::endl;
            }

            lastLeavesCompleted = leavesCompleted;

            std::this_thread::sleep_for(std::chrono::milliseconds(5));

        } while (leavesCompleted < cudaBSP.numLeaves);

        CUDA_CHECK_ERROR(cudaEventRecord(stopEvent));
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        float time;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&time, startEvent, stopEvent));

        std::cout << "Done! (" << time << " ms)" << std::endl;

        CUDA_CHECK_ERROR(cudaEventDestroy(startEvent));
        CUDA_CHECK_ERROR(cudaEventDestroy(stopEvent));

        CUDA_CHECK_ERROR(cudaFreeHost(const_cast<size_t*>(pLeavesCompleted)));
    }
}

namespace DirectLighting {
    static __device__ inline float attenuate(
            BSP::DWorldLight& light,
            float dist
            ) {

        float c = light.constantAtten;
        float l = light.linearAtten;
        float q = light.quadraticAtten;

        return c + l * dist + q * dist * dist;
    }

    __device__ float3 sample_at(
            CUDABSP::CUDABSP& cudaBSP,
            float3 samplePos,
            float3 sampleNormal=make_float3()
            ) {

        uint8_t* pvs = CUDABSP::pvs_for_pos(cudaBSP, samplePos);
        size_t numClusters = cudaBSP.numVisClusters;

        float3 result = make_float3();

        for (size_t lightIndex=0;
                 lightIndex<cudaBSP.numWorldLights;
                 lightIndex++
                 ) {

            BSP::DWorldLight& light = cudaBSP.worldLights[lightIndex];

            //if (light.type == BSP::EMIT_SKYLIGHT) {
            //    printf("SKY HIT intensity %.2f %.2f %.2f normal %.2f %.2f %.2f\n",
            //        light.intensity.x, light.intensity.y, light.intensity.z,
            //        light.normal.x, light.normal.y, light.normal.z);
            //}

            //if (light.type == BSP::EMIT_SKYLIGHT) {
            //    result += make_float3(64.0f, 64.0f, 64.0f);
            //    continue;
            //}

            if (light.type == BSP::EMIT_SKYLIGHT) {
                float3 n = sampleNormal;
                float3 sunDir = make_float3(light.normal);

                float ndotl = dot(n, sunDir * -1.f);
                if (ndotl <= 0.0f)
                {
                    //printf("normal %.3f %.3f %.3f sunDir %.3f %.3f %.3f dot %.2f\n", n.x, n.y, n.z, sunDir.x, sunDir.y, sunDir.z, ndotl);
                    continue;
                }


                float3 start = samplePos + n * 0.001f;
                float3 end = start + sunDir * COORD_EXTENT;

                if (!CUDARAD::g_pDeviceRayTracer->LOS_blocked_sun(start, end))
                {
                    result += make_float3(light.intensity) * ndotl * 255.0f;
                }

                continue;
            }

            if (!CUDABSP::cluster_in_pvs(light.cluster, pvs, numClusters)) {
                // This light isn't within the sample's PVS. Skip it.
                continue;
            }

            float3 lightPos = make_float3(light.origin);
            float3 diff = samplePos - lightPos;

            /*
             * This light is on the wrong side of the current sample.
             * There's no way it could possibly light it.
             */
            if (len(sampleNormal) > 0.0f && dot(diff, sampleNormal) >= 0.0f) {
                continue;
            }

            float dist = len(diff);
            float3 dir = diff / dist;

            float penumbraScale = 1.0f;

            if (light.type == BSP::EMIT_SPOTLIGHT) {
                float3 lightNorm = make_float3(light.normal);
                float lightDot = dot(dir, lightNorm);

                if (lightDot < light.stopdot2) {
                    /* This sample is outside the spotlight cone. */
                    continue;
                }
                else if (lightDot < light.stopdot) {
                    /* This sample is within the spotlight's penumbra. */
                    penumbraScale = (
                        (lightDot - light.stopdot2)
                        / (light.stopdot - light.stopdot2)
                    );
                    //penumbraScale = 100.0;
                }

                //if (lightIndex == cudaBSP.numWorldLights - 1) {
                //    printf(
                //        "(%f, %f, %f) is within spotlight!\n"
                //        "Pos: (%f, %f, %f)\n"
                //        "Norm: <%f, %f, %f> (<%f, %f, %f>)\n"
                //        "stopdot: %f; stopdot2: %f\n"
                //        "Dot between light and sample: %f\n",
                //        samplePos.x, samplePos.y, samplePos.z,
                //        lightPos.x, lightPos.y, lightPos.z,
                //        lightNorm.x, lightNorm.y, lightNorm.z,
                //        light.normal.x, light.normal.y, light.normal.z,
                //        light.stopdot, light.stopdot2,
                //        lightDot
                //    );
                //}
            }

            const float EPSILON = 0.0325f;

            // Nudge the sample position towards the light slightly, to avoid
            // colliding with triangles that directly contain the sample
            // position.
            float3 nudgedSamplePos = samplePos - dir * EPSILON;

            const float SHADOW_EPSILON = 0.05f;

            float3 shadowStart = samplePos + sampleNormal * SHADOW_EPSILON;
            float3 shadowEnd = lightPos;

            bool lightBlocked = CUDARAD::g_pDeviceRayTracer->LOS_blocked(
                shadowStart,
                shadowEnd
            );

            if (lightBlocked) {
                // This light can't be seen from the position of the sample.
                // Ignore it.
                continue;
            }

            /* I CAN SEE THE LIGHT */
            float attenuation = attenuate(light, dist);

            if (!isfinite(attenuation) || attenuation <= 1e-6f) {
                continue;
            }

            float3 lightContribution = make_float3(light.intensity);
            lightContribution *= penumbraScale * 255.0f / attenuation;

            result += lightContribution;
        }

        //printf(
        //    "Sample at (%u, %u) for Face %u: (%f, %f, %f)\n",
        //    static_cast<unsigned int>(s),
        //    static_cast<unsigned int>(t),
        //    static_cast<unsigned int>(faceIndex),
        //    result.x, result.y, result.z
        //);

        return result;
    }

    static __device__ inline float sr_clampf(float v, float lo, float hi)
    {
        return fmaxf(lo, fminf(v, hi));
    }

    static __device__ inline float3 safe_luxel_pos(
        CUDARAD::FaceInfo& faceInfo,
        float s,
        float t
    ) {
        // Do not sample exactly on the border luxel.
        // 0.5f samples at the luxel center and greatly reduces seam leaks.
        float ss = sr_clampf(s + 0.5f, 0.5f, float(faceInfo.lightmapWidth) - 1.5f);
        float tt = sr_clampf(t + 0.5f, 0.5f, float(faceInfo.lightmapHeight) - 1.5f);

        float3 p = faceInfo.xyz_from_st(ss, tt);

        float3 n = faceInfo.faceNorm;

        // Push slightly off the surface so the trace does not start exactly on a plane or edge.
        p.x += n.x * 0.25f;
        p.y += n.y * 0.25f;
        p.z += n.z * 0.25f;

        return p;
    }

    __device__ float3 sample_at(
        CUDABSP::CUDABSP& cudaBSP,
        CUDARAD::FaceInfo& faceInfo,
        float s,
        float t
    )
    {
        float ss = fminf(fmaxf(s + 0.5f, 0.5f), float(faceInfo.lightmapWidth) - 1.5f);
        float tt = fminf(fmaxf(t + 0.5f, 0.5f), float(faceInfo.lightmapHeight) - 1.5f);

        float3 n = faceInfo.faceNorm;

        float3 samplePos = faceInfo.xyz_from_st(ss, tt);

        samplePos.x += n.x * 0.03125f;
        samplePos.y += n.y * 0.03125f;
        samplePos.z += n.z * 0.03125f;

        return sample_at(cudaBSP, samplePos, n);
    }

    __global__ void map_faces(
            CUDABSP::CUDABSP* pCudaBSP,
            size_t* pFacesCompleted
            ) {

        bool primaryThread = (threadIdx.x == 0 && threadIdx.y == 0);

        if (pCudaBSP->tag != CUDABSP::TAG) {
            if (primaryThread) {
                printf("Invalid CUDABSP Tag: %x\n", pCudaBSP->tag);
            }
            return;
        }

        __shared__ CUDARAD::FaceInfo faceInfo;

        if (primaryThread) {
            faceInfo = CUDARAD::FaceInfo(*pCudaBSP, blockIdx.x);
        }

        __syncthreads();

        if (faceInfo.face.lightOffset < 0) {
            if (primaryThread) {
                //printf("skip cuz lightOffset < 0\n");
                atomicAdd(pFacesCompleted, 1);
            }
            return;
        }

        /* Take a sample at each lightmap luxel. */
        for (size_t i=0; i<faceInfo.lightmapHeight; i+=blockDim.y) {
            size_t t = i + threadIdx.y;

            if (t >= faceInfo.lightmapHeight) {
                continue;
            }

            for (size_t j=0; j<faceInfo.lightmapWidth; j+=blockDim.x) {
                size_t s = j + threadIdx.x;

                if (s >= faceInfo.lightmapWidth) {
                    continue;
                }

                float3 color = sample_at(
                    *pCudaBSP, faceInfo,
                    static_cast<float>(s),
                    static_cast<float>(t)
                );

                size_t& lightmapStart = faceInfo.lightmapStartIndex;
                size_t sampleIndex = t * faceInfo.lightmapWidth + s;

                pCudaBSP->lightSamples[lightmapStart + sampleIndex] = color;

                atomicAdd(&faceInfo.totalLight.x, color.x);
                atomicAdd(&faceInfo.totalLight.y, color.y);
                atomicAdd(&faceInfo.totalLight.z, color.z);
            }
        }

        __syncthreads();

        if (primaryThread) {
            faceInfo.avgLight = faceInfo.totalLight;
            faceInfo.avgLight /= static_cast<float>(faceInfo.lightmapSize);

            if (faceInfo.lightmapStartIndex > 0) {
                pCudaBSP->lightSamples[faceInfo.lightmapStartIndex - 1] =
                    faceInfo.avgLight;
            }

            faceInfo.face.styles[0] = 0x00;
            faceInfo.face.styles[1] = 0xFF;
            faceInfo.face.styles[2] = 0xFF;
            faceInfo.face.styles[3] = 0xFF;

            /* Copy our changes back to the CUDABSP. */
            pCudaBSP->faces[faceInfo.faceIndex] = faceInfo.face;

            atomicAdd(reinterpret_cast<unsigned int*>(pFacesCompleted), 1);
            __threadfence_system();
        }

        //printf(
        //    "Lightmap offset for face %u: %u\n",
        //    static_cast<unsigned int>(faceIndex),
        //    static_cast<unsigned int>(lightmapStartIndex)
        //);

        //printf("%u\n", static_cast<unsigned int>(*pFacesCompleted));
    }
}


namespace AA {
    static __device__ const float INV_GAMMA = 1.0f / 2.2f;

    static __device__ inline float perceptual_from_linear(float linear) {
        return powf(linear, INV_GAMMA);
    }

    static __device__ float intensity(float3 rgb) {
        return perceptual_from_linear(
            dot(
                rgb / 255.0f,
                make_float3(1.0f)
                //make_float3(0.299, 0.587, 0.114)
            )
        );
    }

    //static __device__ const float MIN_AA_GRADIENT = 1.0f / 8.0f;
    static __device__ const float MIN_AA_GRADIENT = 1.0f / 16.0f;

    const size_t MAP_FACES_AA_BLOCK_WIDTH = 16;
    const size_t MAP_FACES_AA_BLOCK_HEIGHT = 16;
    const size_t MAP_FACES_AA_NUM_THREADS =
        MAP_FACES_AA_BLOCK_WIDTH * MAP_FACES_AA_BLOCK_HEIGHT;

    __global__ void map_faces_AA(CUDABSP::CUDABSP* pCudaBSP, uint32_t* aaTargetsGlobal,
        uint32_t* scannedAATargetsGlobal,
        uint32_t* aaTargetIndicesGlobal,
        float3* finalSamplesGlobal) {
        int threadID = threadIdx.y * blockDim.x + threadIdx.x;
        bool primaryThread = (threadID == 0);

        __shared__ size_t threadsPerBlock;

        __shared__ size_t faceNum;
        __shared__ CUDARAD::FaceInfo faceInfo;

        __shared__ size_t lightmapStart;
        __shared__ size_t width;
        __shared__ size_t height;

        uint32_t* aaTargets =
            &aaTargetsGlobal[blockIdx.x * CUDABSP::MAX_LUXELS_PER_FACE];

        if (primaryThread) {
            threadsPerBlock = blockDim.x * blockDim.y;

            // Map block numbers to faces.
            faceNum = blockIdx.x;
            faceInfo = CUDARAD::FaceInfo(*pCudaBSP, faceNum);

            lightmapStart = faceInfo.lightmapStartIndex;
            width = faceInfo.lightmapWidth;
            height = faceInfo.lightmapHeight;
        }

        __syncthreads();

        if (width * height > CUDABSP::MAX_LUXELS_PER_FACE)
            printf("width %llu height %llu * %llu | CUDABSP::MAX_LUXELS_PER_FACE %llu\n", width, height, width * height, CUDABSP::MAX_LUXELS_PER_FACE);

        assert(width * height <= CUDABSP::MAX_LUXELS_PER_FACE);

        /* Initialize the AA targets array. */
        for (size_t i=0; i<CUDABSP::MAX_LUXELS_PER_FACE; i+=threadsPerBlock) {
            size_t index = i + threadID;
            if (index >= CUDABSP::MAX_LUXELS_PER_FACE) {
                continue;
            }

            aaTargets[index] = 0;
        }

        __syncthreads();

        /* Select luxels on this face that are good candidates for AA. */
        for (size_t i=0; i<height; i+=blockDim.y) {
            size_t t = i + threadIdx.y;

            if (t >= height) {
                continue;
            }

            for (size_t j=0; j<width; j+=blockDim.x) {
                size_t s = j + threadIdx.x;

                if (s >= width) {
                    continue;
                }

                size_t sampleIndex = t * width + s;

                float3 sampleColor
                    = pCudaBSP->lightSamples[lightmapStart + sampleIndex];

                float sampleIntensity = intensity(sampleColor);

                /* Calculate the maximum gradient of this luxel. */
                float gradient = 0.0;

                for (int tOffset=-1; tOffset<=1; tOffset++) {
                    int neighborT = t + tOffset;

                    if (!(0 <= neighborT && neighborT < height)) {
                        continue;
                    }

                    for (int sOffset=-1; sOffset<=1; sOffset++) {
                        if (sOffset == 0 && tOffset == 0) {
                            continue;
                        }

                        int neighborS = s + sOffset;

                        if (!(0 <= neighborS && neighborS < width)) {
                            continue;
                        }

                        int neighborIndex
                            = neighborT * width + neighborS;

                        float neighborIntensity = intensity(
                            pCudaBSP->lightSamples[
                                lightmapStart + neighborIndex
                            ]
                        );

                        gradient = fmaxf(
                            gradient,
                            fabsf(neighborIntensity - sampleIntensity)
                        );
                    }
                }

                assert(sampleIndex < CUDABSP::MAX_LUXELS_PER_FACE);
                aaTargets[sampleIndex] = (gradient > MIN_AA_GRADIENT);
            }
        }

        __syncthreads();

        uint32_t* scannedAATargets =
            &scannedAATargetsGlobal[blockIdx.x * CUDABSP::MAX_LUXELS_PER_FACE];

        prefix_sum<
            uint32_t, CUDABSP::MAX_LUXELS_PER_FACE,
            MAP_FACES_AA_BLOCK_WIDTH, MAP_FACES_AA_BLOCK_HEIGHT
        >(aaTargets, scannedAATargets);

        __syncthreads();

        __shared__ size_t numAATargets;
        uint32_t* aaTargetIndices =
            &aaTargetIndicesGlobal[blockIdx.x * CUDABSP::MAX_LUXELS_PER_FACE];

        if (primaryThread) {
            numAATargets =
                scannedAATargets[CUDABSP::MAX_LUXELS_PER_FACE - 1]
                + aaTargets[CUDABSP::MAX_LUXELS_PER_FACE - 1];
        }

        __syncthreads();

        /* Gather all the AA targets into the final target array. */
        for (size_t i=0; i<CUDABSP::MAX_LUXELS_PER_FACE; i+=threadsPerBlock) {
            size_t index = i + threadID;
            if (index >= CUDABSP::MAX_LUXELS_PER_FACE) {
                continue;
            }

            if (aaTargets[index]) {
                size_t finalPosition = scannedAATargets[index];
                aaTargetIndices[finalPosition] = index;
            }
        }

        __syncthreads();

        float3* finalSamples =
            &finalSamplesGlobal[blockIdx.x * CUDABSP::MAX_LUXELS_PER_FACE];

        /* Zero out all the final samples. */
        for (size_t i=0; i<numAATargets; i+=threadsPerBlock) {
            size_t index = i + threadID;
            if (index >= numAATargets) {
                continue;
            }

            finalSamples[index] = make_float3();
        }

        __syncthreads();

        const size_t SUPERSAMPLE_WIDTH = 4;
        const size_t SUPERSAMPLES_PER_TARGET =
            SUPERSAMPLE_WIDTH * SUPERSAMPLE_WIDTH;

        const size_t numSupersamples = numAATargets * SUPERSAMPLES_PER_TARGET;

        /* Supersample all the target positions. */
        for (size_t i=0; i<numSupersamples; i+=threadsPerBlock) {
            size_t index = i + threadID;
            if (index >= numSupersamples) {
                continue;
            }

            size_t aaTargetNumber = index / SUPERSAMPLES_PER_TARGET;
            size_t targetSupersampleNumber = index % SUPERSAMPLES_PER_TARGET;

            size_t aaTargetIndex = aaTargetIndices[aaTargetNumber];

            float s = static_cast<float>(aaTargetIndex % width);
            float t = static_cast<float>(aaTargetIndex / width);

            size_t sOffsetIndex = targetSupersampleNumber % SUPERSAMPLE_WIDTH;
            size_t tOffsetIndex = targetSupersampleNumber / SUPERSAMPLE_WIDTH;

            float sStep = 2.0f / static_cast<float>(SUPERSAMPLE_WIDTH);
            float tStep = 2.0f / static_cast<float>(SUPERSAMPLE_WIDTH);

            float sOffset = sStep * sOffsetIndex - 1.0f;
            float tOffset = tStep * tOffsetIndex - 1.0f;

            float3 color = DirectLighting::sample_at(
                *pCudaBSP, faceInfo,
                s + sOffset, t + tOffset
            );

            float3& sample = finalSamples[aaTargetNumber];

            atomicAdd(&sample.x, color.x);
            atomicAdd(&sample.y, color.y);
            atomicAdd(&sample.z, color.z);
        }

        __threadfence_block();
        __syncthreads();

        /* Average out all the supersamples. */
        for (size_t i=0; i<numAATargets; i+=threadsPerBlock) {
            size_t index = i + threadID;
            if (index >= numAATargets) {
                continue;
            }

            finalSamples[index] /= SUPERSAMPLES_PER_TARGET;
        }

        __syncthreads();

        /* Scatter the final samples to their lightmap positions. */
        for (size_t i=0; i<numAATargets; i+=threadsPerBlock) {
            size_t index = i + threadID;
            if (index >= numAATargets) {
                continue;
            }

            size_t targetIndex = aaTargetIndices[index];
            size_t lightmapIndex = lightmapStart + targetIndex;

            pCudaBSP->lightSamples[lightmapIndex] = finalSamples[index];
        }
    }
}


namespace BouncedLighting {
    static __device__ const float PI = 3.14159265358979323846264f;
    static __device__ const float INV_PI = 0.31830988618379067153715f;

    /**
     * Computes the form factor from a differential patch to a convex
     * polygonal patch.
     *
     * Thankfully, Source's polygons are always convex.
     *
     * Formula graciously stolen from Formula 81 of this book:
     * https://people.cs.kuleuven.be/~philip.dutre/GI/TotalCompendium.pdf
     *
     * ... and Formula 4.16 of this one:
     * https://books.google.com/books?id=zALK286TFXgC&lpg=PP1&pg=PA72#v=onepage&q&f=false
     */
    static __device__ float ff_diff_poly(
            float3 diffPos, float3 diffNorm,
            float3* vertices, size_t numVertices
            ) {

        float result = 0.0f;

        for (size_t i=0; i<4; i++) {
            float3 vertex1 = vertices[i] - diffPos;
            float3 vertex2 = vertices[(i + 1) % numVertices] - diffPos;
            float3 vertexCross = cross(vertex1, vertex2);
            float crossLen = len(vertexCross);

            vertexCross /= crossLen;

            float v1Len = len(vertex1);
            float v2Len = len(vertex2);

            float theta =  asinf(crossLen / (v1Len * v2Len));

            result += dot(diffNorm, vertexCross) * theta;
        }

        result *= 0.5f * INV_PI;

        return result;
    }


    /** Computes the form factor between two differential patches. */
    static __device__ float ff_diff_diff(
            float3 diff1Pos, float3 diff1Norm,
            float3 diff2Pos, float3 diff2Norm
            ) {

        float3 delta = diff2Pos - diff1Pos;
        float invDist = 1.0f / len(delta);

        float3 dir = delta * invDist;

        return (
            dot(diff1Norm, dir) * -dot(diff2Norm, dir)
            * INV_PI * invDist * invDist
        );
    }
}


namespace AmbientLighting {
    static __device__ const float AMBIENT_SCALE = 1.0f / 128.0f;

    __global__ void map_leaves(CUDABSP::CUDABSP* pCudaBSP) {
        size_t leafIndex = blockIdx.x;

        if (leafIndex >= pCudaBSP->numLeaves) {
            return;
        }

        BSP::DLeaf& leaf = pCudaBSP->leaves[leafIndex];

        if (leaf.contents & BSP::CONTENTS_SOLID) {
            return;
        }

        //BSP::CompressedLightCube* ambientSamples
        //    = &pCudaBSP->ambientLightSamples[ambientIndex.firstAmbientSample];

        //for (size_t i=threadIdx.x;
        //        i<ambientIndex.ambientSampleCount;
        //        i+=blockDim.x) {

        //    if (i >= ambientIndex.ambientSampleCount) {
        //        return;
        //    }

        //    BSP::CompressedLightCube& sample = ambientSamples[i];

        //    float3 leafMins = make_float3(
        //        leaf.mins[0], leaf.mins[1], leaf.mins[2]
        //    );

        //    float3 leafMaxs = make_float3(
        //        leaf.maxs[0], leaf.maxs[1], leaf.maxs[2]
        //    );

        //    float3 leafSize = leafMaxs - leafMins;

        //    float3 samplePos = leafMins + make_float3(
        //        leafSize.x,
        //        leafSize.y,
        //        leafSize.z
        //    );

        //    //sample.cube.color[0] = BSP::RGBExp32 {1, 1, 1, -3};
        //    //sample.cube.color[1] = BSP::RGBExp32 {1, 1, 1, -3};
        //    //sample.cube.color[2] = BSP::RGBExp32 {1, 1, 1, -3};
        //    //sample.cube.color[3] = BSP::RGBExp32 {1, 1, 1, -3};
        //    //sample.cube.color[4] = BSP::RGBExp32 {1, 1, 1, -3};
        //    //sample.cube.color[5] = BSP::RGBExp32 {1, 1, 1, -3};

        //    /*
        //     * Note: This isn't really the correct way to do ambient lighting.
        //     * Actual ambient lighting would sample lightmaps visible from this
        //     * point in a sphere, and use that information to accumulate
        //     * lighting data into a light cube.
        //     * TODO: Write an actual ambient lighting algorithm.
        //     */

        //    // +X
        //    sample.cube.color[0] = CUDABSP::rgbexp32_from_float3(
        //        DirectLighting::sample_at(
        //            *pCudaBSP,
        //            samplePos,
        //            make_float3(1.0f, 0.0f, 0.0f)
        //        ) * AMBIENT_SCALE
        //    );

        //    // -X
        //    sample.cube.color[1] = CUDABSP::rgbexp32_from_float3(
        //        DirectLighting::sample_at(
        //            *pCudaBSP,
        //            samplePos,
        //            make_float3(-1.0f, 0.0f, 0.0f)
        //        ) * AMBIENT_SCALE
        //    );

        //    // +Y
        //    sample.cube.color[2] = CUDABSP::rgbexp32_from_float3(
        //        DirectLighting::sample_at(
        //            *pCudaBSP,
        //            samplePos,
        //            make_float3(0.0f, 1.0f, 0.0f)
        //        ) * AMBIENT_SCALE
        //    );

        //    // -Y
        //    sample.cube.color[3] = CUDABSP::rgbexp32_from_float3(
        //        DirectLighting::sample_at(
        //            *pCudaBSP,
        //            samplePos,
        //            make_float3(0.0f, -1.0f, 0.0f)
        //        ) * AMBIENT_SCALE
        //    );

        //    // +Z
        //    sample.cube.color[4] = CUDABSP::rgbexp32_from_float3(
        //        DirectLighting::sample_at(
        //            *pCudaBSP,
        //            samplePos,
        //            make_float3(0.0f, 0.0f, 1.0f)
        //        ) * AMBIENT_SCALE
        //    );

        //    // -Z
        //    sample.cube.color[5] = CUDABSP::rgbexp32_from_float3(
        //        DirectLighting::sample_at(
        //            *pCudaBSP,
        //            samplePos,
        //            make_float3(0.0f, 0.0f, -1.0f)
        //        ) * AMBIENT_SCALE
        //    );
        //}
    }
}


namespace BouncedLighting
{
    static __device__ const int PT_SPP = 16;
    static __device__ const int PT_DEPTH = 2;
    static __device__ const float PT_EPSILON = 0.5f;
    static __device__ const float PT_BOUNCE_SCALE = 1.0f;
    static __device__ const float PT_MAX_LIGHT = 4096.0f;

    static __device__ inline uint32_t hash_u32(uint32_t x)
    {
        x ^= x >> 17;
        x *= 0xed5ad4bbU;
        x ^= x >> 11;
        x *= 0xac4c1b51U;
        x ^= x >> 15;
        x *= 0x31848babU;
        x ^= x >> 14;
        return x;
    }

    static __device__ inline float rand01(uint32_t& state)
    {
        state = hash_u32(state);
        return float(state & 0x00ffffffU) / float(0x01000000U);
    }

    static __device__ inline float3 safe_norm3(float3 v)
    {
        float l = len(v);
        if (l <= 1e-20f)
            return make_float3(0.0f, 0.0f, 1.0f);
        return v / l;
    }

    static __device__ inline float3 cosine_hemisphere(float3 n, uint32_t& rng)
    {
        float u1 = rand01(rng);
        float u2 = rand01(rng);

        float r = sqrtf(u1);
        float phi = 2.0f * PI * u2;

        float x = r * cosf(phi);
        float y = r * sinf(phi);
        float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

        float3 up = fabsf(n.z) < 0.999f
            ? make_float3(0.0f, 0.0f, 1.0f)
            : make_float3(1.0f, 0.0f, 0.0f);

        float3 tangent = safe_norm3(cross(up, n));
        float3 bitangent = cross(n, tangent);

        return safe_norm3(tangent * x + bitangent * y + n * z);
    }

    static __device__ inline float3 face_reflectivity(
        const CUDABSP::CUDABSP& cudaBSP,
        int faceId
    ) {
        if (faceId < 0 || faceId >= cudaBSP.numFaces)
            return make_float3(0.0f);

        const BSP::DFace& face = cudaBSP.faces[faceId];

        if (face.texInfo < 0 || face.texInfo >= cudaBSP.numTexInfos)
            return make_float3(0.0f);

        const BSP::TexInfo& texInfo = cudaBSP.texInfos[face.texInfo];

        if (texInfo.texData < 0 || texInfo.texData >= cudaBSP.numTexDatas)
            return make_float3(0.5f);

        const BSP::DTexData& texData = cudaBSP.texDatas[texInfo.texData];

        return make_float3(
            texData.reflectivity.x,
            texData.reflectivity.y,
            texData.reflectivity.z
        );
    }

    static __device__ inline float3 clamp_light(float3 c)
    {
        c.x = fminf(fmaxf(c.x, 0.0f), PT_MAX_LIGHT);
        c.y = fminf(fmaxf(c.y, 0.0f), PT_MAX_LIGHT);
        c.z = fminf(fmaxf(c.z, 0.0f), PT_MAX_LIGHT);
        return c;
    }

    static __device__ float3 sample_indirect_path(
        CUDABSP::CUDABSP& cudaBSP,
        float3 start,
        float3 normal,
        int startFaceId,
        float3 skyAmbient,
        uint32_t& rng
    ) {
        float3 radiance = make_float3(0.0f);
        float3 throughput = face_reflectivity(cudaBSP, startFaceId);

        float3 rayOrigin = start + normal * PT_EPSILON;
        float3 rayDir = cosine_hemisphere(normal, rng);

        for (int depth = 0; depth < PT_DEPTH; ++depth)
        {
            RayTracer::RayHit hit =
                CUDARAD::g_pDeviceRayTracer->trace_closest(
                    rayOrigin,
                    rayOrigin + rayDir * 32768.0f
                );

            if (!hit.hit)
                break;

            int hitFaceId = int(hit.faceId);

            float2 luxel;
            float3 hitLight;

            if (LeafAmbient::world_to_face_luxel(cudaBSP, hitFaceId, hit.position, luxel))
            {
                hitLight = LeafAmbient::compute_lightmap_color_from_luxel(
                    cudaBSP,
                    hitFaceId,
                    luxel,
                    skyAmbient
                );
            }
            else
            {
                hitLight = LeafAmbient::compute_lightmap_color_from_average_fallback(
                    cudaBSP,
                    hitFaceId,
                    skyAmbient
                );
            }

            radiance += sr_mul(throughput, hitLight);

            const BSP::DFace& hitFace = cudaBSP.faces[hitFaceId];
            const BSP::DPlane& hitPlane = cudaBSP.planes[hitFace.planeNum];

            float3 hitNormal = make_float3(
                hitPlane.normal.x,
                hitPlane.normal.y,
                hitPlane.normal.z
            );

            hitNormal = safe_norm3(hitNormal);

            throughput = sr_mul(throughput, face_reflectivity(cudaBSP, hitFaceId));

            float rr = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            rr = fminf(fmaxf(rr, 0.05f), 0.95f);

            if (depth > 0)
            {
                if (rand01(rng) > rr)
                    break;

                throughput /= rr;
            }

            rayOrigin = hit.position + hitNormal * PT_EPSILON;
            rayDir = cosine_hemisphere(hitNormal, rng);
        }

        return radiance;
    }

    __global__ void map_faces(
        CUDABSP::CUDABSP* pCudaBSP,
        size_t* pFacesCompleted
    ) {
        bool primaryThread = (threadIdx.x == 0 && threadIdx.y == 0);

        if (pCudaBSP->tag != CUDABSP::TAG)
        {
            if (primaryThread)
                printf("Invalid CUDABSP Tag: %x\n", pCudaBSP->tag);
            return;
        }

        __shared__ CUDARAD::FaceInfo faceInfo;
        __shared__ float3 skyAmbient;

        if (primaryThread)
        {
            faceInfo = CUDARAD::FaceInfo(*pCudaBSP, blockIdx.x);
            skyAmbient = LeafAmbient::find_sky_ambient(*pCudaBSP);
        }

        __syncthreads();

        if (faceInfo.face.lightOffset < 0)
        {
            if (primaryThread)
            {
                atomicAdd(reinterpret_cast<unsigned long long*>(pFacesCompleted), 1ULL);
                __threadfence_system();
            }
            return;
        }

        for (size_t i = threadIdx.y; i < faceInfo.lightmapHeight; i += blockDim.y)
        {
            for (size_t j = threadIdx.x; j < faceInfo.lightmapWidth; j += blockDim.x)
            {
                size_t s = j;
                size_t t = i;

                float3 samplePos = faceInfo.xyz_from_st(float(s), float(t));
                float3 sampleNormal = faceInfo.faceNorm;

                sampleNormal = safe_norm3(sampleNormal);

                uint32_t rng =
                    hash_u32(
                        uint32_t(blockIdx.x * 9781u) ^
                        uint32_t(s * 6271u) ^
                        uint32_t(t * 7919u) ^
                        0x1234abcdU
                    );

                float3 indirect = make_float3(0.0f);

                for (int sample = 0; sample < PT_SPP; ++sample)
                {
                    indirect += sample_indirect_path(
                        *pCudaBSP,
                        samplePos,
                        sampleNormal,
                        int(faceInfo.faceIndex),
                        skyAmbient,
                        rng
                    );
                }

                indirect /= float(PT_SPP);
                indirect *= PT_BOUNCE_SCALE;
                indirect = clamp_light(indirect);

                size_t sampleIndex = t * faceInfo.lightmapWidth + s;
                size_t lightmapIndex = faceInfo.lightmapStartIndex + sampleIndex;

                pCudaBSP->lightSamples[lightmapIndex] += indirect;
            }
        }

        __syncthreads();

        if (primaryThread)
        {
            atomicAdd(reinterpret_cast<unsigned long long*>(pFacesCompleted), 1ULL);
            __threadfence_system();
        }
    }
}



namespace CUDARAD {
    static constexpr float HOST_COORD_EXTENT = 16384.0f;
    static constexpr size_t HOST_SKY_AMBIENT_SAMPLES = 16;
    static constexpr size_t HOST_GI_SAMPLES = 16;
    static constexpr float HOST_INDIRECT_SCALE = 0.75f;
    static constexpr float HOST_SKY_GI_SCALE = 0.35f;
    static constexpr float HOST_AMBIENT_FLOOR_SCALE = 0.06f;
    static constexpr float HOST_RAY_BIAS = 0.03125f;
    static constexpr float HOST_RAY_TMIN = 0.001f;
    static constexpr float HOST_OCCLUDER_WIDEN_EPSILON = 0.001f;
    static constexpr size_t HOST_DIRECT_SUBSAMPLES = 16;
    static constexpr float HOST_SUN_ANGULAR_RADIUS = 0.02f;
    static constexpr float HOST_LEAF_SURFACE_AMBIENT_SCALE = 0.08f;
    static constexpr float HOST_LEAF_SKY_AMBIENT_SCALE = 0.75f;
    static constexpr float HOST_LEAF_AMBIENT_MAX = 96.0f;

    enum HostSurfaceRole : uint32_t {
        RTX_ROLE_RECEIVER = 1u << 0,
        RTX_ROLE_OCCLUDER = 1u << 1,
        RTX_ROLE_SKY = 1u << 2,
        RTX_ROLE_TOOL = 1u << 3,
        RTX_ROLE_TRANSLUCENT = 1u << 4,
        RTX_ROLE_DISPLACEMENT = 1u << 5,
        RTX_ROLE_STATIC_PROP = 1u << 6,
    };

    static bool host_starts_with(const std::string& text, const char* prefix)
    {
        const size_t prefixLen = std::strlen(prefix);
        return text.size() >= prefixLen
            && text.compare(0, prefixLen, prefix) == 0;
    }

    static std::string host_lower_material_name(std::string name)
    {
        std::replace(name.begin(), name.end(), '\\', '/');
        std::transform(
            name.begin(),
            name.end(),
            name.begin(),
            [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            }
        );
        return name;
    }

    static bool host_is_service_material(const std::string& materialName)
    {
        const std::string name = host_lower_material_name(materialName);

        if (!host_starts_with(name, "tools/")) {
            return false;
        }

        return name.find("trigger") != std::string::npos
            || name.find("clip") != std::string::npos
            || name.find("skip") != std::string::npos
            || name.find("hint") != std::string::npos
            || name.find("nodraw") != std::string::npos
            || name.find("origin") != std::string::npos
            || name.find("areaportal") != std::string::npos
            || name.find("occluder") != std::string::npos
            || name.find("invisible") != std::string::npos;
    }

    static uint32_t host_surface_role_from_flags(
        int32_t flags,
        bool displacement,
        const std::string& materialName
    )
    {
        uint32_t role = 0;

        if (displacement) {
            role |= RTX_ROLE_DISPLACEMENT;
        }

        if (flags & BSP::SURF_SKY) {
            return role | RTX_ROLE_SKY;
        }

        if (flags & (BSP::SURF_TRIGGER | BSP::SURF_SKIP | BSP::SURF_HINT)) {
            return role | RTX_ROLE_TOOL;
        }

        if ((flags & BSP::SURF_NODRAW)
            || (flags & BSP::SURF_NOLIGHT)
            || host_is_service_material(materialName)) {
            return role | RTX_ROLE_TOOL;
        }

        if (flags & BSP::SURF_TRANS) {
            return role | RTX_ROLE_TRANSLUCENT;
        }

        if (!(flags & BSP::SURF_NOLIGHT)) {
            role |= RTX_ROLE_RECEIVER;
        }

        if (!(flags & BSP::SURF_NOSHADOWS)) {
            role |= RTX_ROLE_OCCLUDER;
        }

        return role;
    }

    static std::vector<char> load_optix_ptx()
    {
        const char* candidates[] = {
            "raytracer_optix.ptx",
            "x64\\Debug\\raytracer_optix.ptx",
            "x64\\Release\\raytracer_optix.ptx",
            "code\\x64\\Debug\\raytracer_optix.ptx",
            "code\\x64\\Release\\raytracer_optix.ptx",
        };

        for (const char* candidate : candidates) {
            std::ifstream file(candidate, std::ios::binary);
            if (!file) {
                continue;
            }

            return std::vector<char>(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>()
            );
        }

        std::ostringstream error;
        error << "Unable to find raytracer_optix.ptx. Tried:";
        for (const char* candidate : candidates) {
            error << " " << candidate;
        }
        throw std::runtime_error(error.str());
    }

    static float3 host_normalized(float3 v);

    static bool host_brush_blocks_light(const BSP::DBrush& brush)
    {
        return GeometryRules::brush_contents_block_light(brush.contents);
    }

    static bool host_brush_side_skip_tool_volume_occluder(
        const BSP::BSP& bsp,
        const BSP::TexInfo& texInfo
    )
    {
        if (texInfo.flags
            & (BSP::SURF_TRIGGER | BSP::SURF_SKIP | BSP::SURF_HINT)) {
            return true;
        }

        const std::string materialName =
            bsp.get_texture_name(texInfo.texData);
        const std::string name = host_lower_material_name(materialName);

        if (!host_starts_with(name, "tools/")) {
            return false;
        }

        // Volume / gameplay collision tools — no visible surface; do not add
        // Phantom shadow hulls. (tools/toolsnodraw is structural sealing and
        // MUST stay — VRAD treats nodraw solids as blocking light.)
        if (name.find("trigger") != std::string::npos) {
            return true;
        }
        if (name.find("clip") != std::string::npos) {
            return true;
        }
        if (name.find("skip") != std::string::npos) {
            return true;
        }
        if (name.find("hint") != std::string::npos) {
            return true;
        }
        if (name.find("origin") != std::string::npos) {
            return true;
        }
        if (name.find("areaportal") != std::string::npos) {
            return true;
        }
        if (name.find("occluder") != std::string::npos) {
            return true;
        }

        if (name.find("blocklight") != std::string::npos) {
            return false;
        }
        if (name.find("block") != std::string::npos) {
            return true;
        }

        if (name.find("invisible") != std::string::npos
            && name.find("nodraw") == std::string::npos) {
            return true;
        }

        return false;
    }

    static bool host_brush_side_blocks_light(
        const BSP::BSP& bsp,
        const BSP::DBrush& brush,
        const BSP::DBrushSide& side
    )
    {
        if (side.texInfo < 0
            || static_cast<size_t>(side.texInfo) >= bsp.get_texinfos().size()) {
            // Unknown side: err on sealing the world (matches old VRAD-style
            // brush inclusion) to avoid sun/sky leaks through missing tris.
            return true;
        }

        const BSP::TexInfo& texInfo = bsp.get_texinfos()[side.texInfo];

        if (!GeometryRules::world_brush_side_blocks_light(
            brush.contents,
            texInfo.flags
        )) {
            return false;
        }

        if (host_brush_side_skip_tool_volume_occluder(bsp, texInfo)) {
            return false;
        }

        return true;
    }

    static void host_mark_model_brushes_r(
        const BSP::BSP& bsp,
        int32_t nodeIndex,
        std::vector<uint8_t>& usedBrushes
    )
    {
        if (nodeIndex < 0) {
            const int32_t leafIndex = -nodeIndex - 1;
            const std::vector<BSP::DLeaf>& leaves = bsp.get_leaves();
            const std::vector<uint16_t>& leafBrushes = bsp.get_leafbrushes();

            if (leafIndex < 0
                || static_cast<size_t>(leafIndex) >= leaves.size()) {
                return;
            }

            const BSP::DLeaf& leaf = leaves[leafIndex];
            const size_t firstLeafBrush = leaf.firstLeafBrush;
            const size_t numLeafBrushes = leaf.numLeafBrushes;

            if (firstLeafBrush + numLeafBrushes > leafBrushes.size()) {
                return;
            }

            for (size_t i = 0; i < numLeafBrushes; ++i) {
                const size_t brushIndex = leafBrushes[firstLeafBrush + i];
                if (brushIndex < usedBrushes.size()) {
                    usedBrushes[brushIndex] = 1;
                }
            }

            return;
        }

        const std::vector<BSP::DNode>& nodes = bsp.get_nodes();
        if (static_cast<size_t>(nodeIndex) >= nodes.size()) {
            return;
        }

        const BSP::DNode& node = nodes[nodeIndex];
        host_mark_model_brushes_r(bsp, node.children[0], usedBrushes);
        host_mark_model_brushes_r(bsp, node.children[1], usedBrushes);
    }

    static std::vector<uint8_t> host_collect_world_model_brushes(
        const BSP::BSP& bsp
    )
    {
        std::vector<uint8_t> usedBrushes(bsp.get_brushes().size(), 0);
        const std::vector<BSP::DModel>& models = bsp.get_models();

        if (models.empty()) {
            return usedBrushes;
        }

        host_mark_model_brushes_r(bsp, models[0].headNode, usedBrushes);
        return usedBrushes;
    }

    static float3 host_lerp3(float3 a, float3 b, float t)
    {
        return a + (b - a) * t;
    }

    static float3 host_bilerp3(
        float3 p00,
        float3 p10,
        float3 p11,
        float3 p01,
        float u,
        float v
    )
    {
        const float3 pu0 = host_lerp3(p00, p10, u);
        const float3 pu1 = host_lerp3(p01, p11, u);
        return host_lerp3(pu0, pu1, v);
    }

    static size_t host_disp_index(
        const HostDisplacementSurface& disp,
        int x,
        int y
    )
    {
        return static_cast<size_t>(y) * static_cast<size_t>(disp.gridSize)
            + static_cast<size_t>(x);
    }

    static size_t host_closest_corner_index(
        const std::vector<float3>& corners,
        float3 point
    )
    {
        size_t bestIndex = 0;
        float bestDist2 = FLT_MAX;

        for (size_t i = 0; i < corners.size(); ++i) {
            const float3 d = corners[i] - point;
            const float dist2 = dot(d, d);
            if (dist2 < bestDist2) {
                bestDist2 = dist2;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    static bool host_build_displacement_surface(
        const BSP::BSP& bsp,
        size_t faceIndex,
        HostDisplacementSurface& out
    )
    {
        out = HostDisplacementSurface{};
        out.faceIndex = faceIndex;

        const BSP::DFace& face = bsp.get_dfaces()[faceIndex];
        if (face.dispInfo < 0
            || static_cast<size_t>(face.dispInfo) >= bsp.get_dispinfos().size()) {
            return false;
        }

        const BSP::DispInfo& disp = bsp.get_dispinfos()[face.dispInfo];
        const std::vector<BSP::DispVert>& dispVerts = bsp.get_dispverts();

        out.dispInfoIndex = static_cast<size_t>(face.dispInfo);
        out.gridSize = (1 << disp.power) + 1;
        out.lightmapWidth = face.lightmapTextureSizeInLuxels[0] + 1;
        out.lightmapHeight = face.lightmapTextureSizeInLuxels[1] + 1;

        if (out.gridSize < 2
            || out.lightmapWidth <= 0
            || out.lightmapHeight <= 0) {
            return false;
        }

        const size_t vertexCount =
            static_cast<size_t>(out.gridSize) * static_cast<size_t>(out.gridSize);
        if (disp.dispVertStart < 0
            || static_cast<size_t>(disp.dispVertStart) + vertexCount > dispVerts.size()) {
            return false;
        }

        const float3 modelOrg =
            VradFaceGeometry::face_model_origin(bsp, faceIndex);
        const std::vector<float3> winding =
            VradFaceGeometry::face_world_winding(bsp, faceIndex, modelOrg);
        if (winding.size() != 4) {
            return false;
        }

        const float3 startPos = make_float3(disp.startPosition);
        const size_t startCorner = host_closest_corner_index(winding, startPos);
        const float3 p00 = winding[startCorner];
        const float3 p10 = winding[(startCorner + 1) % 4];
        const float3 p11 = winding[(startCorner + 2) % 4];
        const float3 p01 = winding[(startCorner + 3) % 4];

        out.vertices.resize(vertexCount);
        out.normals.resize(vertexCount, make_float3(0.0f, 0.0f, 1.0f));

        for (int y = 0; y < out.gridSize; ++y) {
            const float v = static_cast<float>(y) / static_cast<float>(out.gridSize - 1);
            for (int x = 0; x < out.gridSize; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(out.gridSize - 1);
                const size_t index =
                    static_cast<size_t>(y) * static_cast<size_t>(out.gridSize)
                    + static_cast<size_t>(x);
                const BSP::DispVert& dispVert =
                    dispVerts[static_cast<size_t>(disp.dispVertStart) + index];

                const float3 basePos = host_bilerp3(p00, p10, p11, p01, u, v);
                const float3 offset =
                    make_float3(dispVert.vector) * dispVert.dist;
                out.vertices[index] = basePos + offset;
            }
        }

        std::vector<float3> normalSums(vertexCount, make_float3(0.0f, 0.0f, 0.0f));

        for (int y = 0; y + 1 < out.gridSize; ++y) {
            for (int x = 0; x + 1 < out.gridSize; ++x) {
                const int nextX = x + 1;
                const int nextY = y + 1;
                const bool odd = (((y * out.gridSize) + x) & 1) != 0;

                const size_t quad[4] = {
                    host_disp_index(out, x, y),
                    host_disp_index(out, x, nextY),
                    host_disp_index(out, nextX, nextY),
                    host_disp_index(out, nextX, y),
                };

                int tris[2][3];
                if (odd) {
                    tris[0][0] = 0; tris[0][1] = 1; tris[0][2] = 3;
                    tris[1][0] = 1; tris[1][1] = 2; tris[1][2] = 3;
                }
                else {
                    tris[0][0] = 0; tris[0][1] = 1; tris[0][2] = 2;
                    tris[1][0] = 0; tris[1][1] = 2; tris[1][2] = 3;
                }

                for (int triIndex = 0; triIndex < 2; ++triIndex) {
                    const size_t ia = quad[tris[triIndex][0]];
                    const size_t ib = quad[tris[triIndex][1]];
                    const size_t ic = quad[tris[triIndex][2]];
                    const float3 a = out.vertices[ia];
                    const float3 b = out.vertices[ib];
                    const float3 c = out.vertices[ic];
                    const float3 triNormal = cross(b - a, c - a);

                    if (len(triNormal) <= 1e-6f) {
                        continue;
                    }

                    normalSums[ia] += triNormal;
                    normalSums[ib] += triNormal;
                    normalSums[ic] += triNormal;
                }
            }
        }

        for (size_t index = 0; index < vertexCount; ++index) {
            float3 normal = host_normalized(normalSums[index]);
            if (len(normal) <= 1e-6f) {
                normal = make_float3(0.0f, 0.0f, 1.0f);
            }
            out.normals[index] = normal;
        }

        out.valid = true;
        return true;
    }

    static bool host_disp_uv_to_surf_point(
        const HostDisplacementSurface& disp,
        float u,
        float v,
        float pushEps,
        float3& outPos
    )
    {
        if (!disp.valid || disp.gridSize < 2 || disp.vertices.empty()) {
            return false;
        }

        if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
            return false;
        }

        const float flU = u * static_cast<float>(disp.gridSize - 1.000001f);
        const float flV = v * static_cast<float>(disp.gridSize - 1.000001f);
        const int snapU = static_cast<int>(flU);
        const int snapV = static_cast<int>(flV);
        const int nextU = std::min(snapU + 1, disp.gridSize - 1);
        const int nextV = std::min(snapV + 1, disp.gridSize - 1);
        const float fracU = flU - static_cast<float>(snapU);
        const float fracV = flV - static_cast<float>(snapV);
        const bool odd = ((((snapV * disp.gridSize) + snapU) & 1) != 0);

        float3 point;
        float3 normal = make_float3(0.0f, 0.0f, 1.0f);

        if (odd) {
            if ((fracU + fracV) >= 1.0f) {
                const size_t i0 = host_disp_index(disp, snapU, nextV);
                const size_t i1 = host_disp_index(disp, nextU, nextV);
                const size_t i2 = host_disp_index(disp, nextU, snapV);
                const float3 edgeU = disp.vertices[i0] - disp.vertices[i1];
                const float3 edgeV = disp.vertices[i2] - disp.vertices[i1];
                point = disp.vertices[i1]
                    + edgeU * (1.0f - fracU)
                    + edgeV * (1.0f - fracV);
                normal = host_normalized(cross(edgeU, edgeV));
            }
            else {
                const size_t i0 = host_disp_index(disp, snapU, snapV);
                const size_t i1 = host_disp_index(disp, snapU, nextV);
                const size_t i2 = host_disp_index(disp, nextU, snapV);
                const float3 edgeU = disp.vertices[i2] - disp.vertices[i0];
                const float3 edgeV = disp.vertices[i1] - disp.vertices[i0];
                point = disp.vertices[i0] + edgeU * fracU + edgeV * fracV;
                normal = host_normalized(cross(edgeU, edgeV));
            }
        }
        else {
            if (fracU < fracV) {
                const size_t i0 = host_disp_index(disp, snapU, snapV);
                const size_t i1 = host_disp_index(disp, snapU, nextV);
                const size_t i2 = host_disp_index(disp, nextU, nextV);
                const float3 edgeU = disp.vertices[i2] - disp.vertices[i1];
                const float3 edgeV = disp.vertices[i0] - disp.vertices[i1];
                point = disp.vertices[i1]
                    + edgeU * fracU
                    + edgeV * (1.0f - fracV);
                normal = host_normalized(cross(edgeV, edgeU));
            }
            else {
                const size_t i0 = host_disp_index(disp, snapU, snapV);
                const size_t i1 = host_disp_index(disp, nextU, nextV);
                const size_t i2 = host_disp_index(disp, nextU, snapV);
                const float3 edgeU = disp.vertices[i0] - disp.vertices[i2];
                const float3 edgeV = disp.vertices[i1] - disp.vertices[i2];
                point = disp.vertices[i2]
                    + edgeU * (1.0f - fracU)
                    + edgeV * fracV;
                normal = host_normalized(cross(edgeV, edgeU));
            }
        }

        if (len(normal) <= 1e-6f) {
            return false;
        }

        outPos = point + normal * pushEps;
        return true;
    }

    static bool host_eval_displacement_surface(
        const HostDisplacementSurface& disp,
        float u,
        float v,
        float3& outPos,
        float3& outNormal
    )
    {
        if (!disp.valid || disp.gridSize < 2 || disp.vertices.empty()) {
            return false;
        }

        u = std::max(0.0f, std::min(u, 1.0f));
        v = std::max(0.0f, std::min(v, 1.0f));

        if (!host_disp_uv_to_surf_point(disp, u, v, 0.0f, outPos)) {
            return false;
        }

        const float fx = u * static_cast<float>(disp.gridSize - 1.000001f);
        const float fy = v * static_cast<float>(disp.gridSize - 1.000001f);

        const int x0 = std::min(disp.gridSize - 2, std::max(0, static_cast<int>(fx)));
        const int y0 = std::min(disp.gridSize - 2, std::max(0, static_cast<int>(fy)));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;

        const float tx = fx - static_cast<float>(x0);
        const float ty = fy - static_cast<float>(y0);

        const size_t i00 = static_cast<size_t>(y0) * disp.gridSize + x0;
        const size_t i10 = static_cast<size_t>(y0) * disp.gridSize + x1;
        const size_t i11 = static_cast<size_t>(y1) * disp.gridSize + x1;
        const size_t i01 = static_cast<size_t>(y1) * disp.gridSize + x0;

        float3 normal = host_bilerp3(
            disp.normals[i00],
            disp.normals[i10],
            disp.normals[i11],
            disp.normals[i01],
            tx,
            ty
        );
        normal = host_normalized(normal);
        if (len(normal) <= 1e-6f) {
            return false;
        }

        outNormal = normal;
        return true;
    }

    static std::vector<float3> host_make_plane_winding(const BSP::DPlane& plane)
    {
        const float3 n = host_normalized(make_float3(plane.normal));
        const float3 center = n * plane.dist;
        const float3 up = fabsf(n.z) < 0.999f
            ? make_float3(0.0f, 0.0f, 1.0f)
            : make_float3(1.0f, 0.0f, 0.0f);
        const float3 tangent = host_normalized(cross(up, n));
        const float3 bitangent = host_normalized(cross(n, tangent));
        const float size = HOST_COORD_EXTENT * 2.0f;

        return {
            center - tangent * size - bitangent * size,
            center + tangent * size - bitangent * size,
            center + tangent * size + bitangent * size,
            center - tangent * size + bitangent * size,
        };
    }

    static std::vector<float3> host_clip_winding_to_plane(
        const std::vector<float3>& in,
        const BSP::DPlane& plane
    )
    {
        static constexpr float CLIP_EPSILON = 0.01f;

        std::vector<float3> out;
        if (in.empty()) {
            return out;
        }

        const float3 n = make_float3(plane.normal);

        for (size_t i = 0; i < in.size(); ++i) {
            const float3 a = in[i];
            const float3 b = in[(i + 1) % in.size()];
            const float da = dot(a, n) - plane.dist;
            const float db = dot(b, n) - plane.dist;
            const bool aInside = da <= CLIP_EPSILON;
            const bool bInside = db <= CLIP_EPSILON;

            if (aInside && bInside) {
                out.push_back(b);
            }
            else if (aInside && !bInside) {
                const float denom = da - db;
                if (fabsf(denom) > 1e-6f) {
                    const float t = da / denom;
                    out.push_back(a + (b - a) * t);
                }
            }
            else if (!aInside && bInside) {
                const float denom = da - db;
                if (fabsf(denom) > 1e-6f) {
                    const float t = da / denom;
                    out.push_back(a + (b - a) * t);
                }
                out.push_back(b);
            }
        }

        return out;
    }

    static void append_opaque_brush_triangles(
        BSP::BSP& bsp,
        std::vector<OptixRT::Triangle>& triangles
    )
    {
        const std::vector<BSP::DBrush>& brushes = bsp.get_brushes();
        const std::vector<BSP::DBrushSide>& sides = bsp.get_brushsides();
        const std::vector<BSP::DPlane>& planes = bsp.get_planes();
        const std::vector<uint8_t> worldBrushes =
            host_collect_world_model_brushes(bsp);
        size_t added = 0;

        for (size_t brushIndex = 0; brushIndex < brushes.size(); ++brushIndex) {
            if (brushIndex >= worldBrushes.size() || !worldBrushes[brushIndex]) {
                continue;
            }

            const BSP::DBrush& brush = brushes[brushIndex];

            if (!host_brush_blocks_light(brush)) {
                continue;
            }

            if (brush.firstSide < 0 || brush.numSides <= 0) {
                continue;
            }

            const size_t firstSide = static_cast<size_t>(brush.firstSide);
            const size_t numSides = static_cast<size_t>(brush.numSides);
            if (firstSide + numSides > sides.size()) {
                continue;
            }

            for (size_t localSide = 0; localSide < numSides; ++localSide) {
                const size_t sideIndex = firstSide + localSide;
                const BSP::DBrushSide& side = sides[sideIndex];

                if (side.bevel || !host_brush_side_blocks_light(bsp, brush, side)) {
                    continue;
                }

                if (side.planeNum >= planes.size()) {
                    continue;
                }

                std::vector<float3> winding =
                    host_make_plane_winding(planes[side.planeNum]);

                for (size_t clipSide = 0; clipSide < numSides && winding.size() >= 3; ++clipSide) {
                    const size_t otherSideIndex = firstSide + clipSide;
                    if (otherSideIndex == sideIndex) {
                        continue;
                    }

                    const BSP::DBrushSide& otherSide = sides[otherSideIndex];
                    if (otherSide.planeNum >= planes.size()) {
                        winding.clear();
                        break;
                    }

                    winding = host_clip_winding_to_plane(
                        winding,
                        planes[otherSide.planeNum]
                    );
                }

                if (winding.size() < 3) {
                    continue;
                }

                const float3 a = winding[0];

                for (size_t i = 1; i + 1 < winding.size(); ++i) {
                    const float3 b = winding[i];
                    const float3 c = winding[i + 1];

                    if (len(cross(b - a, c - a)) <= 1e-4f) {
                        continue;
                    }

                    const float3 center = (a + b + c) / 3.0f;

                    OptixRT::Triangle tri{};
                    tri.v0 = a + host_normalized(a - center) * HOST_OCCLUDER_WIDEN_EPSILON;
                    tri.v1 = b + host_normalized(b - center) * HOST_OCCLUDER_WIDEN_EPSILON;
                    tri.v2 = c + host_normalized(c - center) * HOST_OCCLUDER_WIDEN_EPSILON;
                    tri.sourceId = 0xffffffffu;
                    tri.role = RTX_ROLE_OCCLUDER;
                    tri.visibilityMask = 0xff;
                    triangles.push_back(tri);
                    ++added;
                }
            }
        }

        std::cout << "    Added " << added
            << " opaque brush occluder triangles." << std::endl;
    }

    static bool host_model_name_contains(
        const BSP::StaticPropData& staticProps,
        const BSP::StaticPropLumpV5& prop,
        const char* token
    )
    {
        if (prop.propType >= staticProps.dict.size()) {
            return false;
        }

        std::string name(staticProps.dict[prop.propType].name);
        name = host_lower_material_name(name);
        return name.find(token) != std::string::npos;
    }

    struct HostStaticPropTriangle {
        float3 v0;
        float3 v1;
        float3 v2;
    };

    struct HostStaticPropMesh {
        bool attempted = false;
        bool loaded = false;
        std::vector<HostStaticPropTriangle> triangles;
    };

#pragma pack(push, 1)
    struct HostVtxVertex {
        uint8_t boneWeightIndex[3];
        uint8_t numBones;
        int16_t origMeshVertID;
        int8_t boneID[3];
    };

    struct HostVtxStripHeader {
        int32_t numIndices;
        int32_t indexOffset;
        int32_t numVerts;
        int32_t vertOffset;
        int16_t numBones;
        uint8_t flags;
        int32_t numBoneStateChanges;
        int32_t boneStateChangeOffset;
    };

    struct HostVtxStripGroupHeader {
        int32_t numVerts;
        int32_t vertOffset;
        int32_t numIndices;
        int32_t indexOffset;
        int32_t numStrips;
        int32_t stripOffset;
        uint8_t flags;
    };

    struct HostVtxMeshHeader {
        int32_t numStripGroups;
        int32_t stripGroupHeaderOffset;
        uint8_t flags;
    };

    struct HostVtxModelLODHeader {
        int32_t numMeshes;
        int32_t meshOffset;
        float switchPoint;
    };

    struct HostVtxModelHeader {
        int32_t numLODs;
        int32_t lodOffset;
    };

    struct HostVtxBodyPartHeader {
        int32_t numModels;
        int32_t modelOffset;
    };

    struct HostVtxFileHeader {
        int32_t version;
        int32_t vertCacheSize;
        uint16_t maxBonesPerStrip;
        uint16_t maxBonesPerTri;
        int32_t maxBonesPerVert;
        int32_t checkSum;
        int32_t numLODs;
        int32_t materialReplacementListOffset;
        int32_t numBodyParts;
        int32_t bodyPartOffset;
    };
#pragma pack(pop)

    struct HostStudioHeaderPrefix {
        int32_t id;
        int32_t version;
        int32_t checksum;
        char name[64];
        int32_t length;
        float eyeposition[3];
        float illumposition[3];
        float hullMin[3];
        float hullMax[3];
        float viewBbMin[3];
        float viewBbMax[3];
        int32_t flags;
        int32_t numBones;
        int32_t boneIndex;
        int32_t numBoneControllers;
        int32_t boneControllerIndex;
        int32_t numHitboxSets;
        int32_t hitboxSetIndex;
        int32_t numLocalAnim;
        int32_t localAnimIndex;
        int32_t numLocalSeq;
        int32_t localSeqIndex;
        int32_t activityListVersion;
        int32_t eventsIndexed;
        int32_t numTextures;
        int32_t textureIndex;
        int32_t numCdTextures;
        int32_t cdTextureIndex;
        int32_t numSkinRef;
        int32_t numSkinFamilies;
        int32_t skinIndex;
        int32_t numBodyParts;
        int32_t bodyPartIndex;
    };

    struct HostStudioBodyPart {
        int32_t szNameIndex;
        int32_t numModels;
        int32_t base;
        int32_t modelIndex;
    };

    struct HostStudioModel {
        char name[64];
        int32_t type;
        float boundingRadius;
        int32_t numMeshes;
        int32_t meshIndex;
        int32_t numVertices;
        int32_t vertexIndex;
        int32_t tangentsIndex;
        int32_t numAttachments;
        int32_t attachmentIndex;
        int32_t numEyeballs;
        int32_t eyeballIndex;
        uint32_t vertexDataPtr;
        uint32_t tangentDataPtr;
        int32_t unused[8];
    };

    struct HostStudioMesh {
        int32_t material;
        int32_t modelIndex;
        int32_t numVertices;
        int32_t vertexOffset;
        int32_t numFlexes;
        int32_t flexIndex;
        int32_t materialType;
        int32_t materialParam;
        int32_t meshId;
        float center[3];
        uint32_t modelVertexDataPtr;
        int32_t numLodVertexes[8];
        int32_t unused[8];
    };

    struct HostVvdHeader {
        int32_t id;
        int32_t version;
        int32_t checksum;
        int32_t numLods;
        int32_t numLodVertexes[8];
        int32_t numFixups;
        int32_t fixupTableStart;
        int32_t vertexDataStart;
        int32_t tangentDataStart;
    };

    struct HostStudioVertex {
        float boneWeight[3];
        uint8_t bone[3];
        uint8_t numBones;
        float position[3];
        float normal[3];
        float texCoord[2];
    };

    static_assert(sizeof(HostVtxVertex) == 9, "Unexpected VTX vertex layout.");
    static_assert(sizeof(HostVtxStripHeader) == 27, "Unexpected VTX strip layout.");
    static_assert(sizeof(HostVtxStripGroupHeader) == 25, "Unexpected VTX strip group layout.");
    static_assert(sizeof(HostVtxMeshHeader) == 9, "Unexpected VTX mesh layout.");
    static_assert(sizeof(HostStudioHeaderPrefix) == 240, "Unexpected MDL header prefix layout.");
    static_assert(sizeof(HostStudioModel) == 148, "Unexpected MDL model layout.");
    static_assert(sizeof(HostStudioMesh) == 116, "Unexpected MDL mesh layout.");
    static_assert(sizeof(HostStudioVertex) == 48, "Unexpected VVD vertex layout.");

    template <typename T>
    static bool host_read_struct(
        const std::vector<uint8_t>& data,
        size_t offset,
        T& out
    ) {
        if (offset > data.size() || sizeof(T) > data.size() - offset) {
            return false;
        }

        std::memcpy(&out, data.data() + offset, sizeof(T));
        return true;
    }

    static bool host_valid_range(
        const std::vector<uint8_t>& data,
        size_t offset,
        size_t bytes
    ) {
        return offset <= data.size() && bytes <= data.size() - offset;
    }

    static bool host_read_file(
        const std::string& path,
        std::vector<uint8_t>& out
    ) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }

        f.seekg(0, std::ios::end);
        std::streamoff size = f.tellg();
        if (size <= 0) {
            return false;
        }

        f.seekg(0, std::ios::beg);
        out.resize(static_cast<size_t>(size));
        f.read(reinterpret_cast<char*>(out.data()), size);
        return f.good();
    }

    static std::string host_normalize_path(std::string path) {
        for (char& c : path) {
            if (c == '/') {
                c = '\\';
            }
        }

        return path;
    }

    static std::string host_normalized_model_path(const std::string& modelName) {
        std::string normalized = host_normalize_path(modelName);

        if (normalized.size() < 4
            || host_lower_material_name(normalized.substr(normalized.size() - 4)) != ".mdl") {
            normalized += ".mdl";
        }

        return normalized;
    }

    static std::string host_join_asset_path(
        const std::string& rootPath,
        const std::string& modelName
    ) {
        std::string normalized = host_normalized_model_path(modelName);
        std::string root = host_normalize_path(rootPath);

        while (!root.empty() && (root.back() == '\\' || root.back() == '/')) {
            root.pop_back();
        }

        if (!root.empty() && normalized.size() > 1 && normalized[1] == ':') {
            return normalized;
        }

        return root.empty() ? normalized : (root + "\\" + normalized);
    }

    static std::vector<std::string> host_asset_roots(void) {
        std::vector<std::string> roots;

        for (const std::string& root : g_assetSearchRoots) {
            add_unique_search_root(roots, root);
        }

        if (roots.empty()) {
            add_unique_search_root(roots, g_assetRoot);
        }

        return roots;
    }

    static std::string host_replace_extension(
        const std::string& path,
        const std::string& extension
    ) {
        size_t dot = path.find_last_of('.');
        if (dot == std::string::npos) {
            return path + extension;
        }

        return path.substr(0, dot) + extension;
    }

    static bool host_load_first_existing_file(
        const std::vector<std::string>& paths,
        std::string& loadedPath,
        std::vector<uint8_t>& bytes
    ) {
        for (const std::string& path : paths) {
            if (host_read_file(path, bytes)) {
                loadedPath = path;
                return true;
            }
        }

        return false;
    }

    static float3 host_transform_static_prop_point(
        float3 point,
        const BSP::StaticPropLumpV5& prop
    ) {
        const float pitch = prop.angles.x * 0.01745329251994329577f;
        const float yaw = prop.angles.y * 0.01745329251994329577f;
        const float roll = prop.angles.z * 0.01745329251994329577f;

        const float sp = sinf(pitch);
        const float cp = cosf(pitch);
        const float sy = sinf(yaw);
        const float cy = cosf(yaw);
        const float sr = sinf(roll);
        const float cr = cosf(roll);
        const float3 origin = make_float3(prop.origin);

        return make_float3(
            origin.x + point.x * (cp * cy)
                + point.y * (sr * sp * cy + cr * -sy)
                + point.z * (cr * sp * cy + -sr * -sy),
            origin.y + point.x * (cp * sy)
                + point.y * (sr * sp * sy + cr * cy)
                + point.z * (cr * sp * sy + -sr * cy),
            origin.z + point.x * (-sp)
                + point.y * (sr * cp)
                + point.z * (cr * cp)
        );
    }

    static bool host_static_prop_model_name(
        const BSP::StaticPropData& staticProps,
        const BSP::StaticPropLumpV5& prop,
        std::string& out
    ) {
        if (prop.propType >= staticProps.dict.size()) {
            return false;
        }

        out.assign(staticProps.dict[prop.propType].name);
        size_t nul = out.find('\0');
        if (nul != std::string::npos) {
            out.resize(nul);
        }

        return !out.empty();
    }

    static const float3* host_get_vvd_vertex_position(
        const std::vector<uint8_t>& vvd,
        const HostVvdHeader& vvdHeader,
        int32_t vertexIndex
    ) {
        if (vertexIndex < 0) {
            return nullptr;
        }

        const size_t offset =
            static_cast<size_t>(vvdHeader.vertexDataStart)
            + static_cast<size_t>(vertexIndex) * sizeof(HostStudioVertex)
            + offsetof(HostStudioVertex, position);

        if (!host_valid_range(vvd, offset, sizeof(float) * 3)) {
            return nullptr;
        }

        return reinterpret_cast<const float3*>(vvd.data() + offset);
    }

    static bool host_append_static_prop_mesh_triangles(
        const std::string& modelName,
        HostStaticPropMesh& mesh
    ) {
        std::vector<uint8_t> mdl;
        std::vector<uint8_t> vvd;
        std::vector<uint8_t> vtx;
        std::string mdlPath;
        std::string vvdPath;
        std::string vtxPath;

        for (const std::string& root : host_asset_roots()) {
            const std::string candidateMdlPath =
                host_join_asset_path(root, modelName);
            const std::string candidateVvdPath =
                host_replace_extension(candidateMdlPath, ".vvd");
            const std::vector<std::string> candidateVtxPaths = {
                host_replace_extension(candidateMdlPath, ".dx90.vtx"),
                host_replace_extension(candidateMdlPath, ".dx80.vtx"),
                host_replace_extension(candidateMdlPath, ".sw.vtx")
            };

            if (host_read_file(candidateMdlPath, mdl)
                && host_read_file(candidateVvdPath, vvd)
                && host_load_first_existing_file(candidateVtxPaths, vtxPath, vtx)) {
                mdlPath = candidateMdlPath;
                vvdPath = candidateVvdPath;
                break;
            }

            mdl.clear();
            vvd.clear();
            vtx.clear();
        }

        if (mdl.empty() || vvd.empty() || vtx.empty()) {
            return false;
        }

        HostStudioHeaderPrefix studioHeader{};
        HostVvdHeader vvdHeader{};
        HostVtxFileHeader vtxHeader{};
        if (!host_read_struct(mdl, 0, studioHeader)
            || !host_read_struct(vvd, 0, vvdHeader)
            || !host_read_struct(vtx, 0, vtxHeader)) {
            return false;
        }

        if (studioHeader.numBodyParts <= 0
            || vtxHeader.numBodyParts <= 0
            || studioHeader.bodyPartIndex <= 0
            || vtxHeader.bodyPartOffset <= 0
            || vvdHeader.vertexDataStart <= 0) {
            return false;
        }

        const int bodyPartCount = std::min(
            studioHeader.numBodyParts,
            vtxHeader.numBodyParts
        );

        for (int bodyId = 0; bodyId < bodyPartCount; ++bodyId) {
            HostStudioBodyPart studioBodyPart{};
            HostVtxBodyPartHeader vtxBodyPart{};
            const size_t studioBodyOffset =
                static_cast<size_t>(studioHeader.bodyPartIndex)
                + static_cast<size_t>(bodyId) * sizeof(HostStudioBodyPart);
            const size_t vtxBodyOffset =
                static_cast<size_t>(vtxHeader.bodyPartOffset)
                + static_cast<size_t>(bodyId) * sizeof(HostVtxBodyPartHeader);

            if (!host_read_struct(mdl, studioBodyOffset, studioBodyPart)
                || !host_read_struct(vtx, vtxBodyOffset, vtxBodyPart)) {
                continue;
            }

            const int modelCount = std::min(
                studioBodyPart.numModels,
                vtxBodyPart.numModels
            );

            for (int modelId = 0; modelId < modelCount; ++modelId) {
                HostStudioModel studioModel{};
                HostVtxModelHeader vtxModel{};
                const size_t studioModelOffset =
                    studioBodyOffset
                    + static_cast<size_t>(studioBodyPart.modelIndex)
                    + static_cast<size_t>(modelId) * sizeof(HostStudioModel);
                const size_t vtxModelOffset =
                    vtxBodyOffset
                    + static_cast<size_t>(vtxBodyPart.modelOffset)
                    + static_cast<size_t>(modelId) * sizeof(HostVtxModelHeader);

                if (!host_read_struct(mdl, studioModelOffset, studioModel)
                    || !host_read_struct(vtx, vtxModelOffset, vtxModel)
                    || studioModel.numMeshes <= 0
                    || vtxModel.numLODs <= 0) {
                    continue;
                }

                HostVtxModelLODHeader vtxLod{};
                const size_t vtxLodOffset =
                    vtxModelOffset + static_cast<size_t>(vtxModel.lodOffset);
                if (!host_read_struct(vtx, vtxLodOffset, vtxLod)) {
                    continue;
                }

                const int meshCount = std::min(
                    studioModel.numMeshes,
                    vtxLod.numMeshes
                );

                for (int meshId = 0; meshId < meshCount; ++meshId) {
                    HostStudioMesh studioMesh{};
                    HostVtxMeshHeader vtxMesh{};
                    const size_t studioMeshOffset =
                        studioModelOffset
                        + static_cast<size_t>(studioModel.meshIndex)
                        + static_cast<size_t>(meshId) * sizeof(HostStudioMesh);
                    const size_t vtxMeshOffset =
                        vtxLodOffset
                        + static_cast<size_t>(vtxLod.meshOffset)
                        + static_cast<size_t>(meshId) * sizeof(HostVtxMeshHeader);

                    if (!host_read_struct(mdl, studioMeshOffset, studioMesh)
                        || !host_read_struct(vtx, vtxMeshOffset, vtxMesh)) {
                        continue;
                    }

                    for (int groupId = 0; groupId < vtxMesh.numStripGroups; ++groupId) {
                        HostVtxStripGroupHeader stripGroup{};
                        const size_t stripGroupOffset =
                            vtxMeshOffset
                            + static_cast<size_t>(vtxMesh.stripGroupHeaderOffset)
                            + static_cast<size_t>(groupId) * sizeof(HostVtxStripGroupHeader);

                        if (!host_read_struct(vtx, stripGroupOffset, stripGroup)) {
                            continue;
                        }

                        for (int stripId = 0; stripId < stripGroup.numStrips; ++stripId) {
                            HostVtxStripHeader strip{};
                            const size_t stripOffset =
                                stripGroupOffset
                                + static_cast<size_t>(stripGroup.stripOffset)
                                + static_cast<size_t>(stripId) * sizeof(HostVtxStripHeader);

                            if (!host_read_struct(vtx, stripOffset, strip)) {
                                continue;
                            }

                            if ((strip.flags & 0x01) == 0) {
                                continue;
                            }

                            for (int i = 0; i + 2 < strip.numIndices; i += 3) {
                                const size_t indexOffset =
                                    stripGroupOffset
                                    + static_cast<size_t>(stripGroup.indexOffset)
                                    + static_cast<size_t>(strip.indexOffset + i)
                                        * sizeof(uint16_t);

                                uint16_t vertexIndices[3]{};
                                if (!host_valid_range(vtx, indexOffset, sizeof(vertexIndices))) {
                                    continue;
                                }
                                std::memcpy(vertexIndices, vtx.data() + indexOffset, sizeof(vertexIndices));

                                int32_t originalVertexIds[3]{};
                                bool valid = true;
                                for (int corner = 0; corner < 3; ++corner) {
                                    const size_t vertexOffset =
                                        stripGroupOffset
                                        + static_cast<size_t>(stripGroup.vertOffset)
                                        + static_cast<size_t>(vertexIndices[corner])
                                            * sizeof(HostVtxVertex);
                                    HostVtxVertex vertex{};
                                    if (!host_read_struct(vtx, vertexOffset, vertex)) {
                                        valid = false;
                                        break;
                                    }

                                    originalVertexIds[corner] =
                                        studioModel.vertexIndex
                                        + studioMesh.vertexOffset
                                        + vertex.origMeshVertID;
                                }

                                if (!valid) {
                                    continue;
                                }

                                const float3* p0 = host_get_vvd_vertex_position(
                                    vvd, vvdHeader, originalVertexIds[0]);
                                const float3* p1 = host_get_vvd_vertex_position(
                                    vvd, vvdHeader, originalVertexIds[1]);
                                const float3* p2 = host_get_vvd_vertex_position(
                                    vvd, vvdHeader, originalVertexIds[2]);

                                if (!p0 || !p1 || !p2) {
                                    continue;
                                }

                                if (len(cross(*p1 - *p0, *p2 - *p0)) <= 1e-5f) {
                                    continue;
                                }

                                mesh.triangles.push_back({ *p0, *p1, *p2 });
                            }
                        }
                    }
                }
            }
        }

        return !mesh.triangles.empty();
    }

    static const HostStaticPropMesh& host_get_static_prop_mesh(
        const std::string& modelName
    ) {
        static std::unordered_map<std::string, HostStaticPropMesh> meshCache;

        const std::string cacheKey = host_lower_material_name(host_normalize_path(modelName));
        HostStaticPropMesh& mesh = meshCache[cacheKey];
        if (!mesh.attempted) {
            mesh.attempted = true;
            mesh.loaded = host_append_static_prop_mesh_triangles(modelName, mesh);
        }

        return mesh;
    }

    static bool host_try_read_mdl_hull_aabb(
        const std::string& modelName,
        float3& outMins,
        float3& outMaxs
    )
    {
        std::vector<uint8_t> mdl;

        for (const std::string& root : host_asset_roots()) {
            const std::string path =
                host_join_asset_path(root, modelName);

            if (!host_read_file(path, mdl)) {
                continue;
            }

            if (mdl.size() < sizeof(HostStudioHeaderPrefix)) {
                continue;
            }

            HostStudioHeaderPrefix h{};
            if (!host_read_struct(mdl, 0, h)) {
                continue;
            }

            outMins = make_float3(h.hullMin[0], h.hullMin[1], h.hullMin[2]);
            outMaxs = make_float3(h.hullMax[0], h.hullMax[1], h.hullMax[2]);

            const float dx = outMaxs.x - outMins.x;
            const float dy = outMaxs.y - outMins.y;
            const float dz = outMaxs.z - outMins.z;

            if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
                continue;
            }

            if (dx <= 1e-3f || dy <= 1e-3f || dz <= 1e-3f) {
                continue;
            }

            if (dx > 65536.0f || dy > 65536.0f || dz > 65536.0f) {
                continue;
            }

            return true;
        }

        return false;
    }

    static void host_static_prop_bounds(
        const BSP::StaticPropData& staticProps,
        const BSP::StaticPropLumpV5& prop,
        float3& mins,
        float3& maxs
    )
    {
        mins = make_float3(-32.0f, -32.0f, 0.0f);
        maxs = make_float3(32.0f, 32.0f, 96.0f);

        if (host_model_name_contains(staticProps, prop, "tree")) {
            mins = make_float3(-48.0f, -48.0f, 0.0f);
            maxs = make_float3(48.0f, 48.0f, 320.0f);
        }
        else if (host_model_name_contains(staticProps, prop, "truck")
            || host_model_name_contains(staticProps, prop, "car")
            || host_model_name_contains(staticProps, prop, "vehicle")) {
            mins = make_float3(-96.0f, -48.0f, 0.0f);
            maxs = make_float3(96.0f, 48.0f, 96.0f);
        }
        else if (host_model_name_contains(staticProps, prop, "crate")
            || host_model_name_contains(staticProps, prop, "box")) {
            mins = make_float3(-32.0f, -32.0f, 0.0f);
            maxs = make_float3(32.0f, 32.0f, 64.0f);
        }
        else if (host_model_name_contains(staticProps, prop, "bench")
            || host_model_name_contains(staticProps, prop, "table")) {
            mins = make_float3(-56.0f, -24.0f, 0.0f);
            maxs = make_float3(56.0f, 24.0f, 48.0f);
        }
        else if (host_model_name_contains(staticProps, prop, "fence")
            || host_model_name_contains(staticProps, prop, "railing")) {
            mins = make_float3(-64.0f, -8.0f, 0.0f);
            maxs = make_float3(64.0f, 8.0f, 64.0f);
        }
        else if (host_model_name_contains(staticProps, prop, "plant")
            || host_model_name_contains(staticProps, prop, "bush")
            || host_model_name_contains(staticProps, prop, "foliage")) {
            mins = make_float3(-24.0f, -24.0f, 0.0f);
            maxs = make_float3(24.0f, 24.0f, 64.0f);
        }
    }

    static float3 host_rotate_static_prop_point(
        float3 point,
        const BSP::StaticPropLumpV5& prop
    )
    {
        return host_transform_static_prop_point(point, prop);
    }

    static void append_static_prop_box_face(
        std::vector<OptixRT::Triangle>& triangles,
        const float3& a,
        const float3& b,
        const float3& c,
        const float3& d
    )
    {
        const float3 faceCenter = (a + b + c + d) * 0.25f;
        const float expand = HOST_OCCLUDER_WIDEN_EPSILON;

        OptixRT::Triangle tri0{};
        tri0.v0 = a + host_normalized(a - faceCenter) * expand;
        tri0.v1 = b + host_normalized(b - faceCenter) * expand;
        tri0.v2 = c + host_normalized(c - faceCenter) * expand;
        tri0.sourceId = 0xffffffffu;
        tri0.role = RTX_ROLE_OCCLUDER | RTX_ROLE_STATIC_PROP;
        tri0.visibilityMask = 0xff;
        triangles.push_back(tri0);

        OptixRT::Triangle tri1{};
        tri1.v0 = a + host_normalized(a - faceCenter) * expand;
        tri1.v1 = c + host_normalized(c - faceCenter) * expand;
        tri1.v2 = d + host_normalized(d - faceCenter) * expand;
        tri1.sourceId = 0xffffffffu;
        tri1.role = RTX_ROLE_OCCLUDER | RTX_ROLE_STATIC_PROP;
        tri1.visibilityMask = 0xff;
        triangles.push_back(tri1);
    }

    static void append_static_prop_occluders(
        BSP::BSP& bsp,
        std::vector<OptixRT::Triangle>& triangles
    )
    {
        const BSP::StaticPropData& staticProps = bsp.get_static_props();
        size_t added = 0;
        size_t meshProps = 0;
        size_t skippedProps = 0;

        for (const BSP::StaticPropLumpV5& prop : staticProps.props) {
            if (prop.flags & BSP::STATIC_PROP_NO_SHADOW) {
                continue;
            }

            std::string modelName;
            if (host_static_prop_model_name(staticProps, prop, modelName)) {
                const HostStaticPropMesh& mesh = host_get_static_prop_mesh(modelName);
                if (mesh.loaded) {
                    for (const HostStaticPropTriangle& localTri : mesh.triangles) {
                        float3 tv0 =
                            host_transform_static_prop_point(localTri.v0, prop);
                        float3 tv1 =
                            host_transform_static_prop_point(localTri.v1, prop);
                        float3 tv2 =
                            host_transform_static_prop_point(localTri.v2, prop);

                        if (len(cross(tv1 - tv0, tv2 - tv0)) <= 1e-4f) {
                            continue;
                        }

                        OptixRT::Triangle tri{};
                        tri.v0 = tv0;
                        tri.v1 = tv1;
                        tri.v2 = tv2;
                        tri.sourceId = 0xffffffffu;
                        tri.role = RTX_ROLE_OCCLUDER | RTX_ROLE_STATIC_PROP;
                        tri.visibilityMask = 0xff;
                        triangles.push_back(tri);
                        ++added;
                    }

                    ++meshProps;
                    continue;
                }
            }
            ++skippedProps;
        }

        std::cout << "    Added " << added
            << " static prop occluder triangles ("
            << meshProps << " mesh props, "
            << skippedProps << " skipped without mesh)." << std::endl;
    }

    static void append_blocking_triangles(
        BSP::BSP& bsp,
        std::vector<OptixRT::Triangle>& triangles
    ) {
        for (const BSP::Face& face : bsp.get_faces()) {
            int32_t flags = face.get_texinfo().flags;
            const bool isDisplacement = face.get_data().dispInfo >= 0;
            const std::string materialName =
                bsp.get_texture_name(face.get_texinfo().texData);
            const uint32_t role =
                host_surface_role_from_flags(flags, isDisplacement, materialName);

            if (!(role & (RTX_ROLE_OCCLUDER | RTX_ROLE_SKY))) {
                continue;
            }

            if (isDisplacement
                && static_cast<size_t>(face.id) < g_hostDispSurfaces.size()) {
                const HostDisplacementSurface& disp =
                    g_hostDispSurfaces[static_cast<size_t>(face.id)];

                if (disp.valid && disp.gridSize >= 2) {
                    for (int y = 0; y + 1 < disp.gridSize; ++y) {
                        for (int x = 0; x + 1 < disp.gridSize; ++x) {
                            const size_t i00 = static_cast<size_t>(y) * disp.gridSize + x;
                            const size_t i10 = static_cast<size_t>(y) * disp.gridSize + (x + 1);
                            const size_t i11 = static_cast<size_t>(y + 1) * disp.gridSize + (x + 1);
                            const size_t i01 = static_cast<size_t>(y + 1) * disp.gridSize + x;

                            const float3 quad[4] = {
                                disp.vertices[i00],
                                disp.vertices[i10],
                                disp.vertices[i11],
                                disp.vertices[i01],
                            };
                            const int tris[2][3] = {
                                { 0, 1, 2 },
                                { 0, 2, 3 },
                            };

                            for (int triIndex = 0; triIndex < 2; ++triIndex) {
                                float3 a = quad[tris[triIndex][0]];
                                float3 b = quad[tris[triIndex][1]];
                                float3 c = quad[tris[triIndex][2]];

                                if (len(cross(b - a, c - a)) <= 1e-4f) {
                                    continue;
                                }

                                const float3 center = (a + b + c) / 3.0f;

                                OptixRT::Triangle tri{};
                                tri.v0 = a + host_normalized(a - center) * HOST_OCCLUDER_WIDEN_EPSILON;
                                tri.v1 = b + host_normalized(b - center) * HOST_OCCLUDER_WIDEN_EPSILON;
                                tri.v2 = c + host_normalized(c - center) * HOST_OCCLUDER_WIDEN_EPSILON;
                                tri.sourceId = static_cast<uint32_t>(face.id);
                                tri.role = role;
                                tri.visibilityMask = 0xff;
                                triangles.push_back(tri);
                            }
                        }
                    }

                    continue;
                }
            }

            const std::vector<BSP::Edge>& edges = face.get_edges();

            if (edges.size() < 3)
                continue;

            std::vector<BSP::Vec3<float>> verts;
            verts.reserve(edges.size());

            verts.push_back(edges[0].vertex1);

            BSP::Vec3<float> current = edges[0].vertex2;
            verts.push_back(current);

            for (size_t ei = 1; ei < edges.size(); ++ei) {
                const BSP::Edge& edge = edges[ei];
                BSP::Vec3<float> next;

                if (vec_equal(edge.vertex1, current)) {
                    next = edge.vertex2;
                }
                else if (vec_equal(edge.vertex2, current)) {
                    next = edge.vertex1;
                }
                else {
                    next = edge.vertex1;
                }

                if (ei + 1 < edges.size()) {
                    verts.push_back(next);
                }

                current = next;
            }

            if (verts.size() < 3)
                continue;

            const float3 a = make_float3(verts[0]);

            for (size_t i = 1; i + 1 < verts.size(); ++i) {
                float3 b = make_float3(verts[i]);
                float3 c = make_float3(verts[i + 1]);

                float3 e1 = b - a;
                float3 e2 = c - a;

                if (len(cross(e1, e2)) <= 1e-4f)
                    continue;

                float3 center = (a + b + c) / 3.0f;

                OptixRT::Triangle tri{};
                tri.v0 = a + host_normalized(a - center) * HOST_OCCLUDER_WIDEN_EPSILON;
                tri.v1 = b + host_normalized(b - center) * HOST_OCCLUDER_WIDEN_EPSILON;
                tri.v2 = c + host_normalized(c - center) * HOST_OCCLUDER_WIDEN_EPSILON;
                tri.sourceId = static_cast<uint32_t>(face.id);
                tri.role = role;
                tri.visibilityMask = 0xff;
                triangles.push_back(tri);
            }
        }

        append_opaque_brush_triangles(bsp, triangles);
        append_static_prop_occluders(bsp, triangles);
    }

    static bool host_cluster_in_pvs(
        const BSP::BSP& bsp,
        int16_t sampleCluster,
        int16_t lightCluster
    ) {
        const BSP::BSP::VisMatrix& vis = bsp.get_visibility();

        if (sampleCluster < 0 || lightCluster < 0) {
            return true;
        }

        if (static_cast<size_t>(sampleCluster) >= vis.size()) {
            return true;
        }

        const std::vector<uint8_t>& pvs = vis[sampleCluster];
        size_t byteIndex = static_cast<size_t>(lightCluster) / 8;
        size_t bitIndex = static_cast<size_t>(lightCluster) % 8;

        if (byteIndex >= pvs.size()) {
            return true;
        }

        return ((pvs[byteIndex] >> bitIndex) & 0x1) != 0x0;
    }

    struct HostFaceInfo {
        BSP::DFace face;
        BSP::DPlane plane;
        BSP::TexInfo texInfo;
        CUDAMatrix::CUDAMatrix<double, 3, 3> Ainv;
        float3 faceNorm;
        size_t faceIndex;
        size_t lightmapWidth;
        size_t lightmapHeight;
        size_t lightmapSize;
        size_t lightmapStartIndex;

        HostFaceInfo(BSP::BSP& bsp, size_t faceIndex)
            : face(bsp.get_dfaces()[faceIndex]),
              plane(bsp.get_planes()[face.planeNum]),
              texInfo(bsp.get_texinfos()[face.texInfo]),
              faceIndex(faceIndex),
              lightmapWidth(face.lightmapTextureSizeInLuxels[0] + 1),
              lightmapHeight(face.lightmapTextureSizeInLuxels[1] + 1),
              lightmapSize(lightmapWidth * lightmapHeight),
              lightmapStartIndex(face.lightOffset / sizeof(BSP::RGBExp32))
        {
            const gmtl::Matrix<double, 3, 3>& xyzMatrix =
                bsp.get_faces()[faceIndex].get_st_xyz_matrix();

            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    Ainv[row][col] = xyzMatrix[row][col];
                }
            }

            faceNorm = make_float3(plane.normal);
        }

        float3 xyz_from_st(float s, float t) const
        {
            float sOffset = texInfo.lightmapVecs[0][3];
            float tOffset = texInfo.lightmapVecs[1][3];
            float sMin = static_cast<float>(face.lightmapTextureMinsInLuxels[0]);
            float tMin = static_cast<float>(face.lightmapTextureMinsInLuxels[1]);

            CUDAMatrix::CUDAMatrix<double, 3, 1> B;

            B[0][0] = s - sOffset + sMin;
            B[1][0] = t - tOffset + tMin;
            B[2][0] = plane.dist;

            CUDAMatrix::CUDAMatrix<double, 3, 1> result = Ainv * B;

            return make_float3(
                static_cast<float>(result[0][0]),
                static_cast<float>(result[1][0]),
                static_cast<float>(result[2][0])
            );
        }
    };

    static bool host_is_finite(float v)
    {
        return std::isfinite(v);
    }

    static bool host_is_finite(float3 v)
    {
        return host_is_finite(v.x)
            && host_is_finite(v.y)
            && host_is_finite(v.z);
    }

    static bool host_is_valid_ray(const OptixRT::SunRay& ray)
    {
        return host_is_finite(ray.origin)
            && host_is_finite(ray.direction)
            && host_is_finite(ray.tmin)
            && host_is_finite(ray.tmax)
            && ray.tmax > ray.tmin
            && len(ray.direction) > 1e-6f;
    }

    static float3 host_normalized(float3 v)
    {
        float length = len(v);
        if (!std::isfinite(length) || length <= 1e-6f) {
            return make_float3();
        }

        return v / length;
    }

    static float3 host_find_sky_ambient(const BSP::BSP& bsp)
    {
        for (const BSP::DWorldLight& light : bsp.get_worldlights()) {
            if (light.type == BSP::EMIT_SKYAMBIENT && light.style == 0) {
                return make_float3(light.intensity) * 255.0f;
            }
        }

        return make_float3(24.0f, 24.0f, 24.0f);
    }

    static bool host_face_receives_light(
        const BSP::BSP& bsp,
        const HostFaceInfo& faceInfo
    )
    {
        const std::string materialName =
            bsp.get_texture_name(faceInfo.texInfo.texData);
        uint32_t role = host_surface_role_from_flags(
            faceInfo.texInfo.flags,
            faceInfo.face.dispInfo >= 0,
            materialName
        );
        return (role & RTX_ROLE_RECEIVER) != 0;
    }

    static float3 host_make_tangent(float3 n)
    {
        float3 up = fabsf(n.z) < 0.999f
            ? make_float3(0.0f, 0.0f, 1.0f)
            : make_float3(1.0f, 0.0f, 0.0f);

        return host_normalized(cross(up, n));
    }

    static float3 host_make_bitangent(float3 n, float3 tangent)
    {
        return host_normalized(cross(n, tangent));
    }

    static float3 host_ambient_direction(float3 n, size_t sampleIndex)
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

        float3 tangent = host_make_tangent(n);
        float3 bitangent = host_make_bitangent(n, tangent);

        float x = kDisk[sampleIndex].x;
        float y = kDisk[sampleIndex].y;
        float z = sqrtf(std::max(0.0f, 1.0f - x * x - y * y));

        return host_normalized(tangent * x + bitangent * y + n * z);
    }

    static float2 host_luxel_subsample_offset(size_t sampleIndex)
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

    static float3 host_soft_sun_direction(float3 sunDir, size_t sampleIndex)
    {
        float2 aperture = host_luxel_subsample_offset(sampleIndex);
        float3 tangent = host_make_tangent(sunDir);
        float3 bitangent = host_make_bitangent(sunDir, tangent);

        return host_normalized(
            sunDir
            + tangent * (aperture.x * HOST_SUN_ANGULAR_RADIUS)
            + bitangent * (aperture.y * HOST_SUN_ANGULAR_RADIUS)
        );
    }

    static uint32_t host_hit_role(const OptixRT::RayHit& hit)
    {
        if (!hit.hit || hit.primitiveIndex >= g_optixTriangles.size()) {
            return 0;
        }

        return g_optixTriangles[hit.primitiveIndex].role;
    }

    static float host_luxel_coord(
        size_t coord,
        size_t size,
        float offset
    )
    {
        if (size <= 1) {
            return 0.5f;
        }

        const float lower = 0.125f;
        const float upper = static_cast<float>(size) - 1.125f;
        const float value = static_cast<float>(coord) + 0.5f + offset;

        return std::max(lower, std::min(value, upper));
    }

    static float3 host_surface_reflectivity(
        const BSP::BSP& bsp,
        const HostFaceInfo& faceInfo
    ) {
        if (faceInfo.texInfo.texData < 0
            || static_cast<size_t>(faceInfo.texInfo.texData) >= bsp.get_texdatas().size()) {
            return make_float3(0.5f, 0.5f, 0.5f);
        }

        const BSP::DTexData& texData = bsp.get_texdatas()[faceInfo.texInfo.texData];
        return make_float3(texData.reflectivity);
    }

    static float3 host_gi_direction(float3 n, size_t sampleIndex)
    {
        return host_ambient_direction(n, sampleIndex % HOST_SKY_AMBIENT_SAMPLES);
    }

    static bool host_leaf_receives_ambient(const BSP::DLeaf& leaf)
    {
        return (leaf.contents & BSP::CONTENTS_SOLID) == 0;
    }

    static BSP::RGBExp32 host_rgbexp32_from_float3(float3 color)
    {
        color.x = std::max(color.x, 0.0f);
        color.y = std::max(color.y, 0.0f);
        color.z = std::max(color.z, 0.0f);

        float maxColor = std::max(color.x, std::max(color.y, color.z));
        int exponent = 0;

        if (maxColor > 0.0f) {
            float normalized = maxColor;

            while (normalized > 255.0f && exponent < 127) {
                ++exponent;
                normalized *= 0.5f;
            }

            while (normalized < 127.0f && exponent > -128) {
                --exponent;
                normalized *= 2.0f;
            }
        }

        float scalar = std::ldexp(1.0f, -exponent);

        return BSP::RGBExp32{
            static_cast<uint8_t>(std::min(color.x * scalar, 255.0f)),
            static_cast<uint8_t>(std::min(color.y * scalar, 255.0f)),
            static_cast<uint8_t>(std::min(color.z * scalar, 255.0f)),
            static_cast<int8_t>(exponent)
        };
    }

    static float3 host_tonemap_leaf_ambient(float3 color)
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

    static void host_filter_displacement_direct_lighting(
        const BSP::BSP& bsp,
        const std::vector<HostDisplacementSurface>& dispSurfaces,
        std::vector<float3>& lightSamples
    )
    {
        std::vector<float3> filtered = lightSamples;

        static const float kWeights[3][3] = {
            { 1.0f, 2.0f, 1.0f },
            { 2.0f, 4.0f, 2.0f },
            { 1.0f, 2.0f, 1.0f },
        };

        for (size_t faceIndex = 0; faceIndex < bsp.get_dfaces().size(); ++faceIndex) {
            const BSP::DFace& face = bsp.get_dfaces()[faceIndex];
            if (face.lightOffset < 0
                || face.dispInfo < 0
                || faceIndex >= dispSurfaces.size()
                || !dispSurfaces[faceIndex].valid) {
                continue;
            }

            const size_t width =
                static_cast<size_t>(face.lightmapTextureSizeInLuxels[0] + 1);
            const size_t height =
                static_cast<size_t>(face.lightmapTextureSizeInLuxels[1] + 1);
            const size_t start =
                static_cast<size_t>(face.lightOffset) / sizeof(BSP::RGBExp32);

            for (size_t t = 0; t < height; ++t) {
                for (size_t s = 0; s < width; ++s) {
                    float3 accum = make_float3(0.0f, 0.0f, 0.0f);
                    float totalWeight = 0.0f;

                    for (int dy = -1; dy <= 1; ++dy) {
                        const int nt = static_cast<int>(t) + dy;
                        if (nt < 0 || nt >= static_cast<int>(height)) {
                            continue;
                        }

                        for (int dx = -1; dx <= 1; ++dx) {
                            const int ns = static_cast<int>(s) + dx;
                            if (ns < 0 || ns >= static_cast<int>(width)) {
                                continue;
                            }

                            const float weight = kWeights[dy + 1][dx + 1];
                            const size_t sampleIndex =
                                start
                                + static_cast<size_t>(nt) * width
                                + static_cast<size_t>(ns);
                            accum += lightSamples[sampleIndex] * weight;
                            totalWeight += weight;
                        }
                    }

                    if (totalWeight > 0.0f) {
                        filtered[start + t * width + s] =
                            accum / totalWeight;
                    }
                }
            }
        }

        lightSamples.swap(filtered);
    }

    void init(BSP::BSP& bsp) {
        std::cout << "Setting up OptiX ray-trace acceleration structure... "
            << std::flush;

        using Clock = std::chrono::high_resolution_clock;

        auto start = Clock::now();

        g_hostFaceGeometry.clear();
        g_hostFaceGeometry.resize(bsp.get_faces().size());
        g_hostDispSurfaces.clear();
        g_hostDispSurfaces.resize(bsp.get_faces().size());
        for (size_t faceIndex = 0; faceIndex < g_hostFaceGeometry.size(); ++faceIndex) {
            VradFaceGeometry::build_face_geometry(
                bsp,
                faceIndex,
                g_hostFaceGeometry[faceIndex]
            );
            host_build_displacement_surface(
                bsp,
                faceIndex,
                g_hostDispSurfaces[faceIndex]
            );
        }
        g_optixTriangles.clear();
        append_blocking_triangles(bsp, g_optixTriangles);

        std::vector<char> ptx = load_optix_ptx();

        g_pOptixTracer.reset(new OptixRT::OptixSunLosTracer());
        g_pOptixTracer->init(ptx.data(), ptx.size());
        g_pOptixTracer->build_world_gas(g_optixTriangles);

        bsp.init_ambient_samples();

        auto end = Clock::now();
        std::chrono::milliseconds ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start
            );

        std::cout << "Done! (" << ms.count() << " ms, "
            << g_optixTriangles.size() << " RTX triangles, "
            << bsp.get_brushes().size() << " brushes, "
            << bsp.get_dispinfos().size() << " displacements, "
            << bsp.get_static_props().props.size() << " static props)"
            << std::endl;
    }

    void cleanup(void) {
        g_pOptixTracer = nullptr;
        g_pRayTracer = nullptr;
        g_optixTriangles.clear();
        g_hostFaceGeometry.clear();
        g_hostDispSurfaces.clear();
    }

    void compute_direct_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
        if (!g_pOptixTracer) {
            throw std::runtime_error("CUDARAD::init must create the OptiX tracer before direct lighting");
        }

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

        auto attenuate_host = [](const BSP::DWorldLight& light, float dist) {
            return light.constantAtten
                + light.linearAtten * dist
                + light.quadraticAtten * dist * dist;
        };

        std::vector<float3> lightSamples(
            bsp.get_lightsamples().size(),
            make_float3()
        );
        std::vector<BSP::DFace> dFaces = bsp.get_dfaces();
        std::vector<OptixRT::SunRay> rays;
        std::vector<RayContribution> contributions;
        float3 skyAmbient = host_find_sky_ambient(bsp);

        size_t numFaces = bsp.get_faces().size();
        size_t processedFaces = 0;

        std::cout << "Building OptiX direct-lighting ray batch for "
            << numFaces << " faces at " << HOST_DIRECT_SUBSAMPLES
            << " spp/luxel..." << std::endl;

        auto startTime = Clock::now();

        for (size_t faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
            HostFaceInfo faceInfo(bsp, faceIndex);
            const VradFaceGeometry::FaceGeometry* pFaceGeometry =
                faceIndex < g_hostFaceGeometry.size()
                    ? &g_hostFaceGeometry[faceIndex]
                    : nullptr;
            const HostDisplacementSurface* pDispSurface =
                faceIndex < g_hostDispSurfaces.size()
                    ? &g_hostDispSurfaces[faceIndex]
                    : nullptr;

            if (faceInfo.face.lightOffset < 0 || !host_face_receives_light(bsp, faceInfo)) {
                ++processedFaces;
                continue;
            }

            const bool useCanonicalSamples =
                pFaceGeometry
                && pFaceGeometry->valid
                && !pFaceGeometry->isDisplacement
                && !pFaceGeometry->samples.empty();

            std::vector<const VradFaceGeometry::Sample*> canonicalCellSamples;
            if (useCanonicalSamples) {
                canonicalCellSamples.assign(faceInfo.lightmapSize, nullptr);
                for (const VradFaceGeometry::Sample& sample : pFaceGeometry->samples) {
                    const size_t sampleIndex =
                        static_cast<size_t>(sample.t) * faceInfo.lightmapWidth
                        + static_cast<size_t>(sample.s);
                    if (sampleIndex < canonicalCellSamples.size()) {
                        canonicalCellSamples[sampleIndex] = &sample;
                    }
                }
            }

            for (size_t t = 0; t < faceInfo.lightmapHeight; ++t) {
                for (size_t s = 0; s < faceInfo.lightmapWidth; ++s) {
                    const size_t sampleIndex = t * faceInfo.lightmapWidth + s;
                    const size_t lightmapIndex =
                        faceInfo.lightmapStartIndex + sampleIndex;
                    const float subSampleWeight =
                        1.0f / static_cast<float>(HOST_DIRECT_SUBSAMPLES);
                    const VradFaceGeometry::Sample* pCanonicalSample =
                        useCanonicalSamples
                        && sampleIndex < canonicalCellSamples.size()
                            ? canonicalCellSamples[sampleIndex]
                            : nullptr;

                    for (size_t directSample = 0;
                        directSample < HOST_DIRECT_SUBSAMPLES;
                        ++directSample) {
                        float3 n = host_normalized(faceInfo.faceNorm);
                        float3 samplePos;
                        float3 sampleNormal = n;

                        if (pDispSurface && pDispSurface->valid) {
                            const float2 sampleOffset =
                                host_luxel_subsample_offset(directSample);
                            const float u = clampf_host(
                                (static_cast<float>(s) + 0.5f + sampleOffset.x)
                                    / static_cast<float>(faceInfo.lightmapWidth),
                                0.0f,
                                1.0f
                            );
                            const float v = clampf_host(
                                (static_cast<float>(t) + 0.5f + sampleOffset.y)
                                    / static_cast<float>(faceInfo.lightmapHeight),
                                0.0f,
                                1.0f
                            );

                            if (!host_eval_displacement_surface(
                                *pDispSurface,
                                u,
                                v,
                                samplePos,
                                sampleNormal
                            )) {
                                continue;
                            }

                            n = sampleNormal;
                            samplePos = samplePos + sampleNormal * HOST_RAY_BIAS;
                        }
                        else if (pCanonicalSample) {
                            samplePos = pCanonicalSample->pos + sampleNormal * HOST_RAY_BIAS;
                        }
                        else {
                            float2 sampleOffset =
                                host_luxel_subsample_offset(directSample);
                            float ss = host_luxel_coord(
                                s,
                                faceInfo.lightmapWidth,
                                sampleOffset.x
                            );
                            float tt = host_luxel_coord(
                                t,
                                faceInfo.lightmapHeight,
                                sampleOffset.y
                            );

                            if (pFaceGeometry && pFaceGeometry->valid
                                    && !pFaceGeometry->isDisplacement) {
                                samplePos = VradFaceGeometry::luxel_space_to_world(
                                    *pFaceGeometry,
                                    ss,
                                    tt,
                                    faceInfo.face
                                );
                                samplePos.x += n.x * HOST_RAY_BIAS;
                                samplePos.y += n.y * HOST_RAY_BIAS;
                                samplePos.z += n.z * HOST_RAY_BIAS;
                            }
                            else {
                                samplePos = faceInfo.xyz_from_st(ss, tt)
                                    + n * HOST_RAY_BIAS;
                            }
                        }

                        if (!host_is_finite(n) || !host_is_finite(samplePos)) {
                            continue;
                        }

                        if (!host_is_finite(sampleNormal)
                            || len(sampleNormal) <= 1e-6f) {
                            continue;
                        }

                        BSP::Vec3<float> sampleVec{
                            samplePos.x,
                            samplePos.y,
                            samplePos.z,
                        };
                        int16_t sampleCluster = bsp.cluster_for_pos(sampleVec);

                        for (size_t ai = 0; ai < HOST_SKY_AMBIENT_SAMPLES; ++ai) {
                            float3 ambientDir =
                                host_ambient_direction(sampleNormal, ai);
                            float ndotl = dot(sampleNormal, ambientDir);

                            if (ndotl <= 0.0f) {
                                continue;
                            }

                            OptixRT::SunRay ray = {};
                            ray.origin = samplePos
                                + ambientDir * HOST_RAY_TMIN;
                            ray.direction = ambientDir;
                            ray.tmin = HOST_RAY_TMIN;
                            ray.tmax = HOST_COORD_EXTENT;
                            ray.visibilityMask = 0xff;

                            if (!host_is_valid_ray(ray)) {
                                continue;
                            }

                            rays.push_back(ray);
                            contributions.push_back(RayContribution{
                                lightmapIndex,
                                skyAmbient * (
                                    ndotl * subSampleWeight
                                    / static_cast<float>(HOST_SKY_AMBIENT_SAMPLES)
                                ),
                                DirectVisibilityMode::Unblocked,
                            });
                        }

                        for (const BSP::DWorldLight& light : bsp.get_worldlights()) {
                            if (light.type == BSP::EMIT_SKYAMBIENT) {
                                continue;
                            }

                            if (light.type == BSP::EMIT_SKYLIGHT) {
                                float3 lightNormal = make_float3(light.normal);
                                float3 sunDir = host_soft_sun_direction(
                                    host_normalized(lightNormal) * -1.0f,
                                    directSample
                                );
                                float3 sunNormal = sampleNormal;

                                if (!host_is_finite(sunDir)
                                    || !host_is_finite(sunNormal)) {
                                    continue;
                                }

                                float ndotl = dot(sunNormal, sunDir);

                                if (ndotl <= 0.0f) {
                                    continue;
                                }

                                float3 origin =
                                    samplePos + sunDir * HOST_RAY_TMIN;

                                OptixRT::SunRay ray = {};
                                ray.origin = origin;
                                ray.direction = sunDir;
                                ray.tmin = HOST_RAY_TMIN;
                                ray.tmax = HOST_COORD_EXTENT;
                                ray.visibilityMask = 0xff;

                                if (!host_is_valid_ray(ray)) {
                                    continue;
                                }

                                rays.push_back(ray);
                                contributions.push_back(RayContribution{
                                    lightmapIndex,
                                    make_float3(light.intensity)
                                        * ndotl * 255.0f * subSampleWeight,
                                    DirectVisibilityMode::SkyFirst,
                                });

                                continue;
                            }

                            if (!host_cluster_in_pvs(
                                bsp,
                                sampleCluster,
                                static_cast<int16_t>(light.cluster)
                            )) {
                                continue;
                            }

                            float3 lightPos = make_float3(light.origin);
                            if (!host_is_finite(lightPos)) {
                                continue;
                            }

                            float3 diff = samplePos - lightPos;

                            if (len(n) > 0.0f && dot(diff, n) >= 0.0f) {
                                continue;
                            }

                            float distToLight = len(diff);
                            if (distToLight <= 1e-6f) {
                                continue;
                            }

                            float3 dir = diff / distToLight;
                            float penumbraScale = 1.0f;

                            if (!host_is_finite(dir)) {
                                continue;
                            }

                            if (light.type == BSP::EMIT_SPOTLIGHT) {
                                float3 lightNorm = make_float3(light.normal);
                                if (!host_is_finite(lightNorm)) {
                                    continue;
                                }

                                float lightDot = dot(dir, lightNorm);

                                if (lightDot < light.stopdot2) {
                                    continue;
                                }
                                else if (lightDot < light.stopdot) {
                                    penumbraScale =
                                        (lightDot - light.stopdot2)
                                        / (light.stopdot - light.stopdot2);
                                }
                            }

                            const float SHADOW_EPSILON = HOST_RAY_BIAS;
                            float3 shadowStart =
                                samplePos + n * SHADOW_EPSILON;
                            float3 shadowDelta = lightPos - shadowStart;
                            float shadowDist = len(shadowDelta);

                            if (shadowDist <= 1e-6f) {
                                continue;
                            }

                            float attenuation =
                                attenuate_host(light, distToLight);

                            if (!std::isfinite(attenuation)
                                || attenuation <= 1e-6f) {
                                continue;
                            }

                            float3 lightContribution =
                                make_float3(light.intensity);
                            lightContribution *=
                                penumbraScale * 255.0f
                                * subSampleWeight / attenuation;

                            OptixRT::SunRay ray = {};
                            ray.origin = shadowStart;
                            ray.direction = shadowDelta / shadowDist;
                            ray.tmin = HOST_RAY_TMIN;
                            ray.tmax = shadowDist - HOST_RAY_TMIN;
                            ray.visibilityMask = 0xff;

                            if (!host_is_valid_ray(ray)) {
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
            << " direct-lighting rays with OptiX..." << std::endl;

        std::vector<OptixRT::RayHit> directHits;
        g_pOptixTracer->trace_batch(rays, directHits);

        for (size_t i = 0; i < contributions.size(); ++i) {
            const uint32_t hitRole = host_hit_role(directHits[i]);
            const RayContribution& contribution = contributions[i];
            const bool visible =
                contribution.visibilityMode == DirectVisibilityMode::SkyFirst
                    ? ((hitRole & RTX_ROLE_SKY) != 0)
                    : (!directHits[i].hit || (hitRole & RTX_ROLE_SKY) != 0);

            if (visible) {
                lightSamples[contributions[i].lightmapIndex] +=
                    contribution.color;
            }
        }

        host_filter_displacement_direct_lighting(
            bsp,
            g_hostDispSurfaces,
            lightSamples
        );

        std::vector<float3> faceAverages(numFaces, make_float3());

        for (size_t faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
            HostFaceInfo faceInfo(bsp, faceIndex);

            if (faceInfo.face.lightOffset < 0 || !host_face_receives_light(bsp, faceInfo)) {
                continue;
            }

            float3 totalLight = make_float3();
            const size_t totalSamples = faceInfo.lightmapSize;

            for (size_t sampleIndex = 0; sampleIndex < faceInfo.lightmapSize; ++sampleIndex) {
                totalLight += lightSamples[faceInfo.lightmapStartIndex + sampleIndex];
            }

            if (totalSamples == 0) {
                continue;
            }

            faceAverages[faceIndex] =
                totalLight / static_cast<float>(totalSamples);
        }

        std::vector<OptixRT::SunRay> giRays;
        std::vector<GiRayContribution> giContributions;

        std::cout << "Building OptiX indirect-lighting ray batch..."
            << std::endl;

        for (size_t faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
            HostFaceInfo faceInfo(bsp, faceIndex);

            if (faceInfo.face.lightOffset < 0 || !host_face_receives_light(bsp, faceInfo)) {
                continue;
            }

            const VradFaceGeometry::FaceGeometry* pFaceGeometry =
                faceIndex < g_hostFaceGeometry.size()
                    ? &g_hostFaceGeometry[faceIndex]
                    : nullptr;
            const HostDisplacementSurface* pDispSurface =
                faceIndex < g_hostDispSurfaces.size()
                    ? &g_hostDispSurfaces[faceIndex]
                    : nullptr;

            const float3 reflectivity = host_surface_reflectivity(bsp, faceInfo);

            for (size_t t = 0; t < faceInfo.lightmapHeight; ++t) {
                for (size_t s = 0; s < faceInfo.lightmapWidth; ++s) {
                    float3 samplePos;
                    float3 sampleNormal = host_normalized(faceInfo.faceNorm);
                    if (pDispSurface && pDispSurface->valid) {
                        const float u = clampf_host(
                            (static_cast<float>(s) + 0.5f)
                                / static_cast<float>(faceInfo.lightmapWidth),
                            0.0f,
                            1.0f
                        );
                        const float v = clampf_host(
                            (static_cast<float>(t) + 0.5f)
                                / static_cast<float>(faceInfo.lightmapHeight),
                            0.0f,
                            1.0f
                        );

                        if (!host_eval_displacement_surface(
                            *pDispSurface,
                            u,
                            v,
                            samplePos,
                            sampleNormal
                        )) {
                            continue;
                        }

                        samplePos = samplePos + sampleNormal * HOST_RAY_BIAS;
                    }
                    else {
                        float ss = clampf_host(
                            static_cast<float>(s) + 0.5f,
                            0.5f,
                            static_cast<float>(faceInfo.lightmapWidth) - 1.5f
                        );
                        float tt = clampf_host(
                            static_cast<float>(t) + 0.5f,
                            0.5f,
                            static_cast<float>(faceInfo.lightmapHeight) - 1.5f
                        );

                        if (pFaceGeometry && pFaceGeometry->valid
                            && !pFaceGeometry->isDisplacement) {
                            samplePos = VradFaceGeometry::luxel_space_to_world(
                                *pFaceGeometry,
                                ss,
                                tt,
                                faceInfo.face
                            );
                            samplePos = samplePos + sampleNormal * HOST_RAY_BIAS;
                        }
                        else {
                            samplePos = faceInfo.xyz_from_st(ss, tt)
                                + sampleNormal * HOST_RAY_BIAS;
                        }
                    }

                    if (!host_is_finite(samplePos) || !host_is_finite(sampleNormal)) {
                        continue;
                    }

                    size_t sampleIndex = t * faceInfo.lightmapWidth + s;
                    size_t lightmapIndex =
                        faceInfo.lightmapStartIndex + sampleIndex;

                    for (size_t gi = 0; gi < HOST_GI_SAMPLES; ++gi) {
                        float3 giDir = host_gi_direction(sampleNormal, gi);
                        float ndotl = dot(sampleNormal, giDir);

                        if (ndotl <= 0.0f) {
                            continue;
                        }

                        OptixRT::SunRay ray = {};
                        ray.origin = samplePos + giDir * HOST_RAY_TMIN;
                        ray.direction = giDir;
                        ray.tmin = HOST_RAY_TMIN;
                        ray.tmax = HOST_COORD_EXTENT;
                        ray.visibilityMask = 0xff;

                        if (!host_is_valid_ray(ray)) {
                            continue;
                        }

                        giRays.push_back(ray);
                        giContributions.push_back(GiRayContribution{
                            lightmapIndex,
                            reflectivity,
                            ndotl / static_cast<float>(HOST_GI_SAMPLES),
                        });
                    }
                }
            }
        }

        std::cout << "Tracing " << giRays.size()
            << " indirect-lighting rays with OptiX..." << std::endl;

        std::vector<OptixRT::RayHit> giHits;
        g_pOptixTracer->trace_batch(giRays, giHits);

        for (size_t i = 0; i < giHits.size(); ++i) {
            const OptixRT::RayHit& hit = giHits[i];
            const GiRayContribution& c = giContributions[i];

            if (!hit.hit || hit.primitiveIndex >= g_optixTriangles.size()) {
                float3 bouncedSky = make_float3(
                    skyAmbient.x * c.reflectivity.x,
                    skyAmbient.y * c.reflectivity.y,
                    skyAmbient.z * c.reflectivity.z
                );
                lightSamples[c.lightmapIndex] +=
                    bouncedSky * (c.weight * HOST_SKY_GI_SCALE);
                continue;
            }

            const OptixRT::Triangle& hitTriangle =
                g_optixTriangles[hit.primitiveIndex];

            if (hitTriangle.role & RTX_ROLE_SKY) {
                float3 bouncedSky = make_float3(
                    skyAmbient.x * c.reflectivity.x,
                    skyAmbient.y * c.reflectivity.y,
                    skyAmbient.z * c.reflectivity.z
                );
                lightSamples[c.lightmapIndex] +=
                    bouncedSky * (c.weight * HOST_SKY_GI_SCALE);
                continue;
            }

            uint32_t sourceFace = hitTriangle.sourceId;

            if (sourceFace >= faceAverages.size()) {
                continue;
            }

            float3 bounced = make_float3(
                faceAverages[sourceFace].x * c.reflectivity.x,
                faceAverages[sourceFace].y * c.reflectivity.y,
                faceAverages[sourceFace].z * c.reflectivity.z
            );

            lightSamples[c.lightmapIndex] +=
                bounced * (c.weight * HOST_INDIRECT_SCALE);
        }

        for (size_t faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
            HostFaceInfo faceInfo(bsp, faceIndex);

            if (faceInfo.face.lightOffset < 0 || !host_face_receives_light(bsp, faceInfo)) {
                continue;
            }

            const float3 reflectivity = host_surface_reflectivity(bsp, faceInfo);
            const float3 ambientFloor = make_float3(
                skyAmbient.x * reflectivity.x,
                skyAmbient.y * reflectivity.y,
                skyAmbient.z * reflectivity.z
            ) * HOST_AMBIENT_FLOOR_SCALE;

            float3 totalLight = make_float3();

            for (size_t t = 0; t < faceInfo.lightmapHeight; ++t) {
                for (size_t s = 0; s < faceInfo.lightmapWidth; ++s) {
                    size_t sampleIndex = t * faceInfo.lightmapWidth + s;
                    const size_t lightmapIndex =
                        faceInfo.lightmapStartIndex + sampleIndex;

                    lightSamples[lightmapIndex] += ambientFloor;
                    totalLight += lightSamples[lightmapIndex];
                }
            }

            float3 avgLight =
                totalLight / static_cast<float>(faceInfo.lightmapSize);
            faceAverages[faceIndex] = avgLight;

            if (faceInfo.lightmapStartIndex > 0) {
                lightSamples[faceInfo.lightmapStartIndex - 1] = avgLight;
            }

            dFaces[faceIndex].styles[0] = 0x00;
            dFaces[faceIndex].styles[1] = 0xFF;
            dFaces[faceIndex].styles[2] = 0xFF;
            dFaces[faceIndex].styles[3] = 0xFF;
        }

        std::vector<BSP::CompressedLightCube> leafAmbient(
            bsp.get_leaves().size()
        );

        std::vector<OptixRT::SunRay> leafRays;
        struct LeafRayContribution {
            size_t leafIndex;
            size_t side;
        };
        std::vector<LeafRayContribution> leafContributions;

        static const float3 kCubeDirs[6] = {
            { 1.0f, 0.0f, 0.0f },
            { -1.0f, 0.0f, 0.0f },
            { 0.0f, 1.0f, 0.0f },
            { 0.0f, -1.0f, 0.0f },
            { 0.0f, 0.0f, 1.0f },
            { 0.0f, 0.0f, -1.0f },
        };

        for (size_t leafIndex = 0; leafIndex < bsp.get_leaves().size(); ++leafIndex) {
            const BSP::DLeaf& leaf = bsp.get_leaves()[leafIndex];

            for (size_t side = 0; side < 6; ++side) {
                leafAmbient[leafIndex].color[side] =
                    host_rgbexp32_from_float3(make_float3());
            }

            if (!host_leaf_receives_ambient(leaf)) {
                continue;
            }

            float3 center = make_float3(
                (static_cast<float>(leaf.mins[0]) + static_cast<float>(leaf.maxs[0])) * 0.5f,
                (static_cast<float>(leaf.mins[1]) + static_cast<float>(leaf.maxs[1])) * 0.5f,
                (static_cast<float>(leaf.mins[2]) + static_cast<float>(leaf.maxs[2])) * 0.5f
            );

            for (size_t side = 0; side < 6; ++side) {
                OptixRT::SunRay ray = {};
                ray.origin = center;
                ray.direction = kCubeDirs[side];
                ray.tmin = 0.01f;
                ray.tmax = HOST_COORD_EXTENT;
                ray.visibilityMask = 0xff;

                if (!host_is_valid_ray(ray)) {
                    continue;
                }

                leafRays.push_back(ray);
                leafContributions.push_back(LeafRayContribution{ leafIndex, side });
            }
        }

        std::cout << "Tracing " << leafRays.size()
            << " leaf ambient rays with OptiX..." << std::endl;

        std::vector<OptixRT::RayHit> leafHits;
        g_pOptixTracer->trace_batch(leafRays, leafHits);

        for (size_t i = 0; i < leafHits.size(); ++i) {
            const LeafRayContribution& c = leafContributions[i];
            const OptixRT::RayHit& hit = leafHits[i];

            float3 color = skyAmbient * HOST_LEAF_SKY_AMBIENT_SCALE;

            if (hit.hit && hit.primitiveIndex < g_optixTriangles.size()) {
                const OptixRT::Triangle& hitTriangle =
                    g_optixTriangles[hit.primitiveIndex];
                uint32_t sourceFace = hitTriangle.sourceId;
                if (!(hitTriangle.role & RTX_ROLE_SKY)
                    && sourceFace < faceAverages.size()) {
                    color = faceAverages[sourceFace]
                        * HOST_LEAF_SURFACE_AMBIENT_SCALE;
                }
            }

            leafAmbient[c.leafIndex].color[c.side] =
                host_rgbexp32_from_float3(host_tonemap_leaf_ambient(color));
        }

        CUDABSP::CUDABSP cudaBSP;
        CUDA_CHECK_ERROR(cudaMemcpy(
            &cudaBSP,
            pCudaBSP,
            sizeof(CUDABSP::CUDABSP),
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
                sizeof(BSP::DFace) * dFaces.size(),
                cudaMemcpyHostToDevice
            ));
        }

        if (!leafAmbient.empty()) {
            CUDA_CHECK_ERROR(cudaMemcpy(
                cudaBSP.ambientLightSamples,
                leafAmbient.data(),
                sizeof(BSP::CompressedLightCube) * leafAmbient.size(),
                cudaMemcpyHostToDevice
            ));
        }

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        auto endTime = Clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime
        );

        std::cout << "Done! (" << ms.count() << " ms)" << std::endl;
    }

    void antialias_direct_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
        cudaEvent_t startEvent;
        cudaEvent_t stopEvent;

        CUDA_CHECK_ERROR(cudaEventCreate(&startEvent));
        CUDA_CHECK_ERROR(cudaEventCreate(&stopEvent));

        CUDA_CHECK_ERROR(cudaEventRecord(startEvent));

        size_t numFaces = bsp.get_faces().size();

        dim3 blockDim(
            AA::MAP_FACES_AA_BLOCK_WIDTH,
            AA::MAP_FACES_AA_BLOCK_HEIGHT
        );

        uint32_t* aaTargetsGlobal = nullptr;
        uint32_t* scannedAATargetsGlobal = nullptr;
        uint32_t* aaTargetIndicesGlobal = nullptr;
        float3* finalSamplesGlobal = nullptr;

        size_t luxelBufferSize =
            numFaces * CUDABSP::MAX_LUXELS_PER_FACE;

        CUDA_CHECK_ERROR(cudaMalloc(
            &aaTargetsGlobal,
            sizeof(uint32_t) * luxelBufferSize
        ));

        CUDA_CHECK_ERROR(cudaMalloc(
            &scannedAATargetsGlobal,
            sizeof(uint32_t) * luxelBufferSize
        ));

        CUDA_CHECK_ERROR(cudaMalloc(
            &aaTargetIndicesGlobal,
            sizeof(uint32_t) * luxelBufferSize
        ));

        CUDA_CHECK_ERROR(cudaMalloc(
            &finalSamplesGlobal,
            sizeof(float3) * luxelBufferSize
        ));

        KERNEL_LAUNCH(
            AA::map_faces_AA,
            numFaces, blockDim,
            pCudaBSP,
            aaTargetsGlobal,
            scannedAATargetsGlobal,
            aaTargetIndicesGlobal,
            finalSamplesGlobal
        );

        CUDA_CHECK_ERROR(cudaEventRecord(stopEvent));

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        CUDA_CHECK_ERROR(cudaFree(aaTargetsGlobal));

        float time;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&time, startEvent, stopEvent));

        CUDA_CHECK_ERROR(cudaEventDestroy(startEvent));
        CUDA_CHECK_ERROR(cudaEventDestroy(stopEvent));

        std::cout << "Done! (" << time << " ms)" << std::endl;
    }

    void compute_ambient_lighting(CUDABSP::CUDABSP* pCudaBSP) {
        using Clock = std::chrono::high_resolution_clock;

        auto start = Clock::now();

        const size_t BLOCK_WIDTH = 32;

        size_t numLeaves;

        CUDA_CHECK_ERROR(
            cudaMemcpy(
                &numLeaves, &pCudaBSP->numLeaves, sizeof(size_t),
                cudaMemcpyDeviceToHost
            )
        );

        KERNEL_LAUNCH(
            AmbientLighting::map_leaves,
            numLeaves, BLOCK_WIDTH,
            pCudaBSP
        );

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        auto end = Clock::now();
        std::chrono::milliseconds ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start
                );

        std::cout << "Done! (" << ms.count() << " ms)" << std::endl;
    }

    void compute_leaf_ambient(CUDABSP::CUDABSP* pCudaBSP)
    {
        LeafAmbient::run(pCudaBSP);
    }

    void bounce_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP)
    {
        volatile size_t* pFacesCompleted;

        CUDA_CHECK_ERROR(
            cudaHostAlloc(
                &pFacesCompleted,
                sizeof(size_t),
                cudaHostAllocMapped
            )
        );

        *pFacesCompleted = 0;

        volatile size_t* pDeviceFacesCompleted;

        CUDA_CHECK_ERROR(
            cudaHostGetDevicePointer(
                const_cast<size_t**>(&pDeviceFacesCompleted),
                const_cast<size_t*>(pFacesCompleted),
                0
            )
        );

        const size_t BLOCK_WIDTH = 8;
        const size_t BLOCK_HEIGHT = 8;

        size_t numFaces = bsp.get_faces().size();

        dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);

        std::cout
            << "Launching path traced bounce lighting: "
            << numFaces * BLOCK_WIDTH * BLOCK_HEIGHT
            << " threads ("
            << numFaces
            << " faces, "
            << BouncedLighting::PT_SPP
            << " spp, depth "
            << BouncedLighting::PT_DEPTH
            << ")..."
            << std::endl;

        cudaEvent_t startEvent;
        cudaEvent_t stopEvent;

        CUDA_CHECK_ERROR(cudaEventCreate(&startEvent));
        CUDA_CHECK_ERROR(cudaEventCreate(&stopEvent));
        CUDA_CHECK_ERROR(cudaEventRecord(startEvent));

        KERNEL_LAUNCH(
            BouncedLighting::map_faces,
            numFaces,
            blockDim,
            pCudaBSP,
            const_cast<size_t*>(pDeviceFacesCompleted)
        );

        flush_wddm_queue();

        size_t lastFacesCompleted = 0;
        size_t facesCompleted;

        do
        {
            CUDA_CHECK_ERROR(cudaPeekAtLastError());

            facesCompleted = *pFacesCompleted;

            if (facesCompleted > lastFacesCompleted)
            {
                std::cout
                    << "  "
                    << facesCompleted
                    << "/"
                    << numFaces
                    << " faces bounced..."
                    << std::endl;
            }

            lastFacesCompleted = facesCompleted;

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } while (facesCompleted < numFaces);

        CUDA_CHECK_ERROR(cudaEventRecord(stopEvent));
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        float time;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&time, startEvent, stopEvent));

        CUDA_CHECK_ERROR(cudaEventDestroy(startEvent));
        CUDA_CHECK_ERROR(cudaEventDestroy(stopEvent));

        cudaFreeHost(const_cast<size_t*>(pFacesCompleted));

        std::cout << "Done! (" << time << " ms)" << std::endl;
    }
}
