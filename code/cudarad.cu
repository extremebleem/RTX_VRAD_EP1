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

#include "core/state.h"
#include "core/lighting/ambient_lighting.h"


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
    static std::unique_ptr<SilkRAD::Core::RuntimeState> g_coreState;
    static std::vector<OptixRT::Triangle> g_optixTriangles;
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

    static std::vector<std::string> host_asset_roots(void)
    {
        std::vector<std::string> roots;

        for (const std::string& root : g_assetSearchRoots) {
            add_unique_search_root(roots, root);
        }

        if (roots.empty()) {
            add_unique_search_root(roots, g_assetRoot);
        }

        return roots;
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

    static std::vector<char> load_optix_ptx();

    static void host_rebuild_optix_world_from_core(void)
    {
        if (!g_coreState) {
            return;
        }

        SilkRAD::Core::Lighting::build_runtime_world(
            *g_coreState,
            g_optixTriangles
        );

        std::vector<char> ptx = load_optix_ptx();
        if (!g_pOptixTracer) {
            g_pOptixTracer.reset(new OptixRT::OptixSunLosTracer());
            g_pOptixTracer->init(ptx.data(), ptx.size());
        }
        g_pOptixTracer->build_world_gas(g_optixTriangles);
    }

    void init(BSP::BSP& bsp) {
        SilkRAD::Core::BSP::SourceMap sourceMap(bsp);
        SilkRAD::Core::BuildOptions options;
        options.assetSearchRoots = host_asset_roots();
        g_coreState.reset(new SilkRAD::Core::RuntimeState(
            SilkRAD::Core::build_runtime_state(sourceMap, options)
        ));
        host_rebuild_optix_world_from_core();

        const SilkRAD::Core::Lighting::DirectLightingResult& direct =
            g_coreState->directLightingInputs;
        std::cout << "Initialized core runtime state ("
            << direct.ordinaryFaceCount << " ordinary faces, "
            << direct.displacementFaceCount << " displacement faces, "
            << direct.worldTriangleCount << " world face triangles, "
            << direct.worldBrushTriangleCount << " world brush triangles, "
            << direct.displacementTriangleCount << " displacement triangles, "
            << direct.staticPropTriangleCount << " static prop triangles)"
            << std::endl;
    }

    void cleanup(void) {
        g_pOptixTracer = nullptr;
        g_pRayTracer = nullptr;
        g_coreState.reset();
        g_optixTriangles.clear();
    }

    void compute_direct_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
        if (!g_coreState) {
            init(bsp);
        }

        const SilkRAD::Core::Lighting::DirectLightingResult& direct =
            g_coreState->directLightingInputs;
        std::cout << "Core direct-lighting runtime using canonical geometry ("
            << direct.ordinaryFaceCount << " ordinary, "
            << direct.displacementFaceCount << " displacement, "
            << direct.worldTriangleCount << " world-face tris, "
            << direct.worldBrushTriangleCount << " world-brush tris, "
            << direct.displacementTriangleCount << " disp tris, "
            << direct.staticPropTriangleCount << " static-prop tris)"
            << std::endl;

        if (!g_pOptixTracer) {
            throw std::runtime_error("CUDARAD::init must create the OptiX tracer before direct lighting");
        }

        SilkRAD::Core::Lighting::compute_direct_lighting_runtime(
            bsp,
            pCudaBSP,
            *g_coreState,
            g_optixTriangles,
            *g_pOptixTracer
        );
    }

    void compute_leaf_ambient(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
        if (!g_coreState) {
            init(bsp);
        }

        if (!g_pOptixTracer) {
            throw std::runtime_error("CUDARAD::init must create the OptiX tracer before leaf ambient");
        }

        SilkRAD::Core::Lighting::compute_leaf_ambient_from_cuda_runtime(
            bsp,
            pCudaBSP,
            *g_coreState,
            *g_pOptixTracer
        );
    }

}
