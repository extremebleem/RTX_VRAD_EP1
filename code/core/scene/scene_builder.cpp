#include "scene_builder.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../geometry_rules.h"

namespace SilkRAD::Core::Scene {
    namespace {
        struct StaticPropTriangle {
            Common::Vec3f v0;
            Common::Vec3f v1;
            Common::Vec3f v2;
        };

        struct StaticPropLightingMeshData {
            size_t lod = 0;
            std::vector<Common::StaticPropLightingVertex> vertices;
        };

        struct StaticPropMesh {
            bool attempted = false;
            bool loaded = false;
            uint32_t checksum = 0;
            std::vector<StaticPropTriangle> triangles;
            std::vector<StaticPropLightingMeshData> lightingMeshes;
        };

#pragma pack(push, 1)
        struct VtxVertex {
            uint8_t boneWeightIndex[3];
            uint8_t numBones;
            int16_t origMeshVertID;
            int8_t boneID[3];
        };

        struct VtxStripHeader {
            int32_t numIndices;
            int32_t indexOffset;
            int32_t numVerts;
            int32_t vertOffset;
            int16_t numBones;
            uint8_t flags;
            int32_t numBoneStateChanges;
            int32_t boneStateChangeOffset;
        };

        struct VtxStripGroupHeader {
            int32_t numVerts;
            int32_t vertOffset;
            int32_t numIndices;
            int32_t indexOffset;
            int32_t numStrips;
            int32_t stripOffset;
            uint8_t flags;
        };

        struct VtxMeshHeader {
            int32_t numStripGroups;
            int32_t stripGroupHeaderOffset;
            uint8_t flags;
        };

        struct VtxModelLODHeader {
            int32_t numMeshes;
            int32_t meshOffset;
            float switchPoint;
        };

        struct VtxModelHeader {
            int32_t numLODs;
            int32_t lodOffset;
        };

        struct VtxBodyPartHeader {
            int32_t numModels;
            int32_t modelOffset;
        };

        struct VtxFileHeader {
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

        struct StudioHeaderPrefix {
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

        struct StudioBodyPart {
            int32_t szNameIndex;
            int32_t numModels;
            int32_t base;
            int32_t modelIndex;
        };

        struct StudioModel {
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

        struct StudioMesh {
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

        struct VvdHeader {
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

        struct StudioVertex {
            float boneWeight[3];
            uint8_t bone[3];
            uint8_t numBones;
            float position[3];
            float normal[3];
            float texCoord[2];
        };

        static_assert(sizeof(VtxVertex) == 9, "Unexpected VTX vertex layout.");
        static_assert(sizeof(VtxStripHeader) == 27, "Unexpected VTX strip layout.");
        static_assert(sizeof(VtxStripGroupHeader) == 25, "Unexpected VTX strip group layout.");
        static_assert(sizeof(VtxMeshHeader) == 9, "Unexpected VTX mesh layout.");
        static_assert(sizeof(StudioHeaderPrefix) == 240, "Unexpected MDL header prefix layout.");
        static_assert(sizeof(StudioModel) == 148, "Unexpected MDL model layout.");
        static_assert(sizeof(StudioMesh) == 116, "Unexpected MDL mesh layout.");
        static_assert(sizeof(StudioVertex) == 48, "Unexpected VVD vertex layout.");

        Common::Vec3f from_bsp_vec3(const ::BSP::Vec3<float>& v)
        {
            return Common::make_vec3(v.x, v.y, v.z);
        }

        Common::OccluderTriangle make_triangle(
            Common::Vec3f a,
            Common::Vec3f b,
            Common::Vec3f c,
            uint32_t sourceId,
            uint32_t role,
            Common::OccluderTriangle::SourceKind sourceKind,
            int32_t surfaceFlags,
            int32_t contents
        )
        {
            Common::OccluderTriangle tri;
            tri.v0 = a;
            tri.v1 = b;
            tri.v2 = c;
            tri.sourceId = sourceId;
            tri.role = role;
            tri.sourceKind = sourceKind;
            tri.surfaceFlags = surfaceFlags;
            tri.contents = contents;
            return tri;
        }

        template <typename T>
        bool read_struct(const std::vector<uint8_t>& data, size_t offset, T& out)
        {
            if (offset > data.size() || sizeof(T) > data.size() - offset) {
                return false;
            }

            std::memcpy(&out, data.data() + offset, sizeof(T));
            return true;
        }

        bool valid_range(
            const std::vector<uint8_t>& data,
            size_t offset,
            size_t bytes
        )
        {
            return offset <= data.size() && bytes <= data.size() - offset;
        }

        bool read_file(const std::string& path, std::vector<uint8_t>& out)
        {
            std::ifstream stream(path, std::ios::binary);
            if (!stream) {
                return false;
            }

            stream.seekg(0, std::ios::end);
            const std::streamoff size = stream.tellg();
            if (size <= 0) {
                return false;
            }

            stream.seekg(0, std::ios::beg);
            out.resize(static_cast<size_t>(size));
            stream.read(reinterpret_cast<char*>(out.data()), size);
            return stream.good();
        }

        std::string normalize_path(std::string path)
        {
            for (char& c : path) {
                if (c == '/') {
                    c = '\\';
                }
            }
            return path;
        }

        std::string lower_ascii(std::string text)
        {
            std::transform(
                text.begin(),
                text.end(),
                text.begin(),
                [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                }
            );
            return text;
        }

        uint32_t surface_role_from_flags(
            int32_t flags,
            bool displacement,
            const std::string& materialName
        )
        {
            uint32_t role = 0;

            if (displacement) {
                role |= GeometryRules::RTX_ROLE_DISPLACEMENT;
            }

            if (flags & ::BSP::SURF_SKY) {
                return role | GeometryRules::RTX_ROLE_SKY;
            }

            if (flags & (::BSP::SURF_TRIGGER | ::BSP::SURF_SKIP | ::BSP::SURF_HINT)) {
                return role | GeometryRules::RTX_ROLE_TOOL;
            }

            if (!displacement
                && ((flags & ::BSP::SURF_NODRAW)
                    || (flags & ::BSP::SURF_NOLIGHT)
                    || GeometryRules::is_service_material(materialName))) {
                return role | GeometryRules::RTX_ROLE_TOOL;
            }

            if (flags & ::BSP::SURF_TRANS) {
                return role | GeometryRules::RTX_ROLE_TRANSLUCENT;
            }

            if (displacement || !(flags & ::BSP::SURF_NOLIGHT)) {
                role |= GeometryRules::RTX_ROLE_RECEIVER;
            }

            if (displacement || !(flags & ::BSP::SURF_NOSHADOWS)) {
                role |= GeometryRules::RTX_ROLE_OCCLUDER;
            }

            return role;
        }

        std::string replace_extension(
            const std::string& path,
            const std::string& extension
        )
        {
            const size_t dot = path.find_last_of('.');
            if (dot == std::string::npos) {
                return path + extension;
            }

            return path.substr(0, dot) + extension;
        }

        std::string normalized_model_path(const std::string& modelName)
        {
            std::string normalized = normalize_path(modelName);
            if (normalized.size() < 4
                || lower_ascii(normalized.substr(normalized.size() - 4)) != ".mdl") {
                normalized += ".mdl";
            }
            return normalized;
        }

        std::string join_asset_path(
            const std::string& rootPath,
            const std::string& modelName
        )
        {
            std::string root = normalize_path(rootPath);
            while (!root.empty() && (root.back() == '\\' || root.back() == '/')) {
                root.pop_back();
            }

            const std::string normalized = normalized_model_path(modelName);
            if (root.empty() || (normalized.size() > 1 && normalized[1] == ':')) {
                return normalized;
            }

            return root + "\\" + normalized;
        }

        bool load_first_existing_file(
            const std::vector<std::string>& paths,
            std::string& loadedPath,
            std::vector<uint8_t>& bytes
        )
        {
            for (const std::string& path : paths) {
                if (read_file(path, bytes)) {
                    loadedPath = path;
                    return true;
                }
            }

            return false;
        }

        Common::Vec3f transform_static_prop_point(
            Common::Vec3f point,
            const ::BSP::StaticPropLumpV5& prop
        )
        {
            const float pitch = prop.angles.x * 0.01745329251994329577f;
            const float yaw = prop.angles.y * 0.01745329251994329577f;
            const float roll = prop.angles.z * 0.01745329251994329577f;

            const float sp = std::sin(pitch);
            const float cp = std::cos(pitch);
            const float sy = std::sin(yaw);
            const float cy = std::cos(yaw);
            const float sr = std::sin(roll);
            const float cr = std::cos(roll);

            return Common::make_vec3(
                prop.origin.x + point.x * (cp * cy)
                    + point.y * (sr * sp * cy + cr * -sy)
                    + point.z * (cr * sp * cy + -sr * -sy),
                prop.origin.y + point.x * (cp * sy)
                    + point.y * (sr * sp * sy + cr * cy)
                    + point.z * (cr * sp * sy + -sr * cy),
                prop.origin.z + point.x * (-sp)
                    + point.y * (sr * cp)
                    + point.z * (cr * cp)
            );
        }

        Common::Vec3f transform_static_prop_normal(
            Common::Vec3f normal,
            const ::BSP::StaticPropLumpV5& prop
        )
        {
            const float pitch = prop.angles.x * 0.01745329251994329577f;
            const float yaw = prop.angles.y * 0.01745329251994329577f;
            const float roll = prop.angles.z * 0.01745329251994329577f;

            const float sp = std::sin(pitch);
            const float cp = std::cos(pitch);
            const float sy = std::sin(yaw);
            const float cy = std::cos(yaw);
            const float sr = std::sin(roll);
            const float cr = std::cos(roll);

            return Common::normalized(Common::make_vec3(
                normal.x * (cp * cy)
                    + normal.y * (sr * sp * cy + cr * -sy)
                    + normal.z * (cr * sp * cy + -sr * -sy),
                normal.x * (cp * sy)
                    + normal.y * (sr * sp * sy + cr * cy)
                    + normal.z * (cr * sp * sy + -sr * cy),
                normal.x * (-sp)
                    + normal.y * (sr * cp)
                    + normal.z * (cr * cp)
            ));
        }

        bool static_prop_model_name(
            const ::BSP::StaticPropData& staticProps,
            const ::BSP::StaticPropLumpV5& prop,
            std::string& out
        )
        {
            if (prop.propType >= staticProps.dict.size()) {
                return false;
            }

            out.assign(staticProps.dict[prop.propType].name);
            const size_t nul = out.find('\0');
            if (nul != std::string::npos) {
                out.resize(nul);
            }

            return !out.empty();
        }

        const Common::Vec3f* get_vvd_vertex_position(
            const std::vector<uint8_t>& vvd,
            const VvdHeader& vvdHeader,
            int32_t vertexIndex
        )
        {
            if (vertexIndex < 0) {
                return nullptr;
            }

            const size_t offset =
                static_cast<size_t>(vvdHeader.vertexDataStart)
                + static_cast<size_t>(vertexIndex) * sizeof(StudioVertex)
                + offsetof(StudioVertex, position);

            if (!valid_range(vvd, offset, sizeof(float) * 3)) {
                return nullptr;
            }

            return reinterpret_cast<const Common::Vec3f*>(vvd.data() + offset);
        }

        const Common::Vec3f* get_vvd_vertex_normal(
            const std::vector<uint8_t>& vvd,
            const VvdHeader& vvdHeader,
            int32_t vertexIndex
        )
        {
            if (vertexIndex < 0) {
                return nullptr;
            }

            const size_t offset =
                static_cast<size_t>(vvdHeader.vertexDataStart)
                + static_cast<size_t>(vertexIndex) * sizeof(StudioVertex)
                + offsetof(StudioVertex, normal);

            if (!valid_range(vvd, offset, sizeof(float) * 3)) {
                return nullptr;
            }

            return reinterpret_cast<const Common::Vec3f*>(vvd.data() + offset);
        }

        bool append_static_prop_mesh_triangles(
            const std::vector<std::string>& assetRoots,
            const std::string& modelName,
            StaticPropMesh& mesh
        )
        {
            std::vector<uint8_t> mdl;
            std::vector<uint8_t> vvd;
            std::vector<uint8_t> vtx;

            for (const std::string& root : assetRoots) {
                const std::string mdlPath = join_asset_path(root, modelName);
                const std::string vvdPath = replace_extension(mdlPath, ".vvd");
                const std::vector<std::string> candidateVtxPaths = {
                    replace_extension(mdlPath, ".dx90.vtx"),
                    replace_extension(mdlPath, ".dx80.vtx"),
                    replace_extension(mdlPath, ".sw.vtx")
                };
                std::string vtxPath;

                if (read_file(mdlPath, mdl)
                    && read_file(vvdPath, vvd)
                    && load_first_existing_file(candidateVtxPaths, vtxPath, vtx)) {
                    break;
                }

                mdl.clear();
                vvd.clear();
                vtx.clear();
            }

            if (mdl.empty() || vvd.empty() || vtx.empty()) {
                return false;
            }

            StudioHeaderPrefix studioHeader{};
            VvdHeader vvdHeader{};
            VtxFileHeader vtxHeader{};
            if (!read_struct(mdl, 0, studioHeader)
                || !read_struct(vvd, 0, vvdHeader)
                || !read_struct(vtx, 0, vtxHeader)) {
                return false;
            }

            mesh.checksum = static_cast<uint32_t>(studioHeader.checksum);

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
                StudioBodyPart studioBodyPart{};
                VtxBodyPartHeader vtxBodyPart{};
                const size_t studioBodyOffset =
                    static_cast<size_t>(studioHeader.bodyPartIndex)
                    + static_cast<size_t>(bodyId) * sizeof(StudioBodyPart);
                const size_t vtxBodyOffset =
                    static_cast<size_t>(vtxHeader.bodyPartOffset)
                    + static_cast<size_t>(bodyId) * sizeof(VtxBodyPartHeader);

                if (!read_struct(mdl, studioBodyOffset, studioBodyPart)
                    || !read_struct(vtx, vtxBodyOffset, vtxBodyPart)) {
                    continue;
                }

                const int modelCount = std::min(
                    studioBodyPart.numModels,
                    vtxBodyPart.numModels
                );

                for (int modelId = 0; modelId < modelCount; ++modelId) {
                    StudioModel studioModel{};
                    VtxModelHeader vtxModel{};
                    const size_t studioModelOffset =
                        studioBodyOffset
                        + static_cast<size_t>(studioBodyPart.modelIndex)
                        + static_cast<size_t>(modelId) * sizeof(StudioModel);
                    const size_t vtxModelOffset =
                        vtxBodyOffset
                        + static_cast<size_t>(vtxBodyPart.modelOffset)
                        + static_cast<size_t>(modelId) * sizeof(VtxModelHeader);

                    if (!read_struct(mdl, studioModelOffset, studioModel)
                        || !read_struct(vtx, vtxModelOffset, vtxModel)
                        || studioModel.numMeshes <= 0
                        || vtxModel.numLODs <= 0) {
                        continue;
                    }

                    VtxModelLODHeader vtxLod{};
                    const size_t vtxLodOffset =
                        vtxModelOffset + static_cast<size_t>(vtxModel.lodOffset);
                    if (!read_struct(vtx, vtxLodOffset, vtxLod)) {
                        continue;
                    }

                    const int meshCount = std::min(studioModel.numMeshes, vtxLod.numMeshes);
                    for (int meshId = 0; meshId < meshCount; ++meshId) {
                        StudioMesh studioMesh{};
                        VtxMeshHeader vtxMesh{};
                        const size_t studioMeshOffset =
                            studioModelOffset
                            + static_cast<size_t>(studioModel.meshIndex)
                            + static_cast<size_t>(meshId) * sizeof(StudioMesh);
                        const size_t vtxMeshOffset =
                            vtxLodOffset
                            + static_cast<size_t>(vtxLod.meshOffset)
                            + static_cast<size_t>(meshId) * sizeof(VtxMeshHeader);

                        if (!read_struct(mdl, studioMeshOffset, studioMesh)
                            || !read_struct(vtx, vtxMeshOffset, vtxMesh)) {
                            continue;
                        }

                        for (int groupId = 0; groupId < vtxMesh.numStripGroups; ++groupId) {
                            VtxStripGroupHeader stripGroup{};
                            const size_t stripGroupOffset =
                                vtxMeshOffset
                                + static_cast<size_t>(vtxMesh.stripGroupHeaderOffset)
                                + static_cast<size_t>(groupId) * sizeof(VtxStripGroupHeader);

                            if (!read_struct(vtx, stripGroupOffset, stripGroup)) {
                                continue;
                            }

                            StaticPropLightingMeshData lightingMesh;
                            lightingMesh.lod = 0;
                            lightingMesh.vertices.reserve(
                                static_cast<size_t>(std::max(stripGroup.numVerts, 0))
                            );

                            for (int vertexId = 0; vertexId < stripGroup.numVerts; ++vertexId) {
                                const size_t vertexOffset =
                                    stripGroupOffset
                                    + static_cast<size_t>(stripGroup.vertOffset)
                                    + static_cast<size_t>(vertexId) * sizeof(VtxVertex);
                                VtxVertex vertex{};
                                if (!read_struct(vtx, vertexOffset, vertex)) {
                                    lightingMesh.vertices.clear();
                                    break;
                                }

                                const int32_t originalVertexId =
                                    studioModel.vertexIndex
                                    + studioMesh.vertexOffset
                                    + vertex.origMeshVertID;
                                const Common::Vec3f* position =
                                    get_vvd_vertex_position(vvd, vvdHeader, originalVertexId);
                                const Common::Vec3f* normal =
                                    get_vvd_vertex_normal(vvd, vvdHeader, originalVertexId);
                                if (!position || !normal) {
                                    lightingMesh.vertices.clear();
                                    break;
                                }

                                lightingMesh.vertices.push_back(
                                    Common::StaticPropLightingVertex{
                                        *position,
                                        *normal
                                    }
                                );
                            }

                            for (int stripId = 0; stripId < stripGroup.numStrips; ++stripId) {
                                VtxStripHeader strip{};
                                const size_t stripOffset =
                                    stripGroupOffset
                                    + static_cast<size_t>(stripGroup.stripOffset)
                                    + static_cast<size_t>(stripId) * sizeof(VtxStripHeader);

                                if (!read_struct(vtx, stripOffset, strip)
                                    || (strip.flags & 0x01) == 0) {
                                    continue;
                                }

                                for (int i = 0; i + 2 < strip.numIndices; i += 3) {
                                    const size_t indexOffset =
                                        stripGroupOffset
                                        + static_cast<size_t>(stripGroup.indexOffset)
                                        + static_cast<size_t>(strip.indexOffset + i)
                                            * sizeof(uint16_t);

                                    uint16_t vertexIndices[3]{};
                                    if (!valid_range(vtx, indexOffset, sizeof(vertexIndices))) {
                                        continue;
                                    }
                                    std::memcpy(
                                        vertexIndices,
                                        vtx.data() + indexOffset,
                                        sizeof(vertexIndices)
                                    );

                                    int32_t originalVertexIds[3]{};
                                    bool valid = true;
                                    for (int corner = 0; corner < 3; ++corner) {
                                        const size_t vertexOffset =
                                            stripGroupOffset
                                            + static_cast<size_t>(stripGroup.vertOffset)
                                            + static_cast<size_t>(vertexIndices[corner])
                                                * sizeof(VtxVertex);
                                        VtxVertex vertex{};
                                        if (!read_struct(vtx, vertexOffset, vertex)) {
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

                                    const Common::Vec3f* p0 =
                                        get_vvd_vertex_position(vvd, vvdHeader, originalVertexIds[0]);
                                    const Common::Vec3f* p1 =
                                        get_vvd_vertex_position(vvd, vvdHeader, originalVertexIds[1]);
                                    const Common::Vec3f* p2 =
                                        get_vvd_vertex_position(vvd, vvdHeader, originalVertexIds[2]);
                                    if (!p0 || !p1 || !p2) {
                                        continue;
                                    }

                                    const Common::Vec3f crossVec =
                                        Common::cross(Common::sub(*p1, *p0), Common::sub(*p2, *p0));
                                    if (Common::length(crossVec) <= 1e-5f) {
                                        continue;
                                    }

                                    mesh.triangles.push_back({ *p0, *p1, *p2 });
                                }
                            }

                            if (!lightingMesh.vertices.empty()) {
                                mesh.lightingMeshes.push_back(std::move(lightingMesh));
                            }
                        }
                    }
                }
            }

            return !mesh.triangles.empty();
        }

        const StaticPropMesh& get_static_prop_mesh(
            const std::vector<std::string>& assetRoots,
            const std::string& modelName
        )
        {
            static std::unordered_map<std::string, StaticPropMesh> cache;

            std::string cacheKey;
            for (const std::string& root : assetRoots) {
                cacheKey += lower_ascii(normalize_path(root));
                cacheKey.push_back('|');
            }
            cacheKey += lower_ascii(normalize_path(modelName));

            StaticPropMesh& mesh = cache[cacheKey];
            if (!mesh.attempted) {
                mesh.attempted = true;
                mesh.loaded = append_static_prop_mesh_triangles(assetRoots, modelName, mesh);
            }
            return mesh;
        }

        std::vector<Common::Vec3f> make_plane_winding(const ::BSP::DPlane& plane)
        {
            const Common::Vec3f normal = Common::normalized(from_bsp_vec3(plane.normal));
            const Common::Vec3f center = Common::scale(normal, plane.dist);
            Common::Vec3f up = Common::make_vec3(0.0f, 0.0f, 0.0f);
            if (std::fabs(normal.x) >= std::fabs(normal.y)
                && std::fabs(normal.x) >= std::fabs(normal.z)) {
                up.z = 1.0f;
            }
            else if (std::fabs(normal.y) >= std::fabs(normal.z)) {
                up.z = 1.0f;
            }
            else {
                up.x = 1.0f;
            }

            up = Common::normalized(Common::sub(
                up,
                Common::scale(normal, Common::dot(up, normal))
            ));
            const Common::Vec3f right = Common::normalized(Common::cross(up, normal));
            const float size = 16384.0f * 2.0f;
            const Common::Vec3f scaledUp = Common::scale(up, size);
            const Common::Vec3f scaledRight = Common::scale(right, size);

            return {
                Common::add(Common::sub(center, scaledRight), scaledUp),
                Common::add(Common::add(center, scaledRight), scaledUp),
                Common::sub(Common::add(center, scaledRight), scaledUp),
                Common::sub(Common::sub(center, scaledRight), scaledUp),
            };
        }

        ::BSP::DPlane flipped_plane(const std::vector<::BSP::DPlane>& planes, size_t planeIndex)
        {
            if ((planeIndex ^ 1u) < planes.size()) {
                return planes[planeIndex ^ 1u];
            }

            ::BSP::DPlane plane = planes[planeIndex];
            plane.normal.x = -plane.normal.x;
            plane.normal.y = -plane.normal.y;
            plane.normal.z = -plane.normal.z;
            plane.dist = -plane.dist;
            return plane;
        }

        std::vector<Common::Vec3f> clip_winding_to_plane(
            const std::vector<Common::Vec3f>& input,
            const ::BSP::DPlane& plane
        )
        {
            static constexpr float CLIP_EPSILON = 0.01f;

            std::vector<Common::Vec3f> output;
            if (input.empty()) {
                return output;
            }

            const Common::Vec3f normal = from_bsp_vec3(plane.normal);
            for (size_t i = 0; i < input.size(); ++i) {
                const Common::Vec3f a = input[i];
                const Common::Vec3f b = input[(i + 1) % input.size()];
                const float da = Common::dot(a, normal) - plane.dist;
                const float db = Common::dot(b, normal) - plane.dist;
                const bool aInside = da >= -CLIP_EPSILON;
                const bool bInside = db >= -CLIP_EPSILON;

                if (aInside && bInside) {
                    output.push_back(b);
                }
                else if (aInside && !bInside) {
                    const float denom = da - db;
                    if (std::fabs(denom) > 1e-6f) {
                        const float t = da / denom;
                        output.push_back(Common::add(a, Common::scale(Common::sub(b, a), t)));
                    }
                }
                else if (!aInside && bInside) {
                    const float denom = da - db;
                    if (std::fabs(denom) > 1e-6f) {
                        const float t = da / denom;
                        output.push_back(Common::add(a, Common::scale(Common::sub(b, a), t)));
                    }
                    output.push_back(b);
                }
            }

            return output;
        }

        void append_world_brush_triangles(
            const BSP::SourceMap& sourceMap,
            SceneBuildResult& result
        )
        {
            const ::BSP::BSP& bsp = sourceMap.raw_bsp();
            const std::vector<uint8_t> worldBrushMask = sourceMap.world_brush_mask();
            const std::vector<::BSP::DBrush>& brushes = bsp.get_brushes();
            const std::vector<::BSP::DBrushSide>& sides = bsp.get_brushsides();
            const std::vector<::BSP::DPlane>& planes = bsp.get_planes();

            for (size_t brushIndex = 0; brushIndex < brushes.size(); ++brushIndex) {
                if (brushIndex >= worldBrushMask.size() || !worldBrushMask[brushIndex]) {
                    continue;
                }

                const ::BSP::DBrush& brush = brushes[brushIndex];
                if (brush.firstSide < 0
                    || brush.numSides <= 0) {
                    continue;
                }

                const size_t firstSide = static_cast<size_t>(brush.firstSide);
                const size_t numSides = static_cast<size_t>(brush.numSides);
                if (firstSide + numSides > sides.size()) {
                    continue;
                }

                for (size_t localSide = 0; localSide < numSides; ++localSide) {
                    const size_t sideIndex = firstSide + localSide;
                    const ::BSP::DBrushSide& side = sides[sideIndex];
                    if (side.bevel || side.dispInfo >= 0 || side.planeNum >= planes.size()) {
                        continue;
                    }

                    int32_t surfaceFlags = 0;
                    std::string materialName;
                    if (side.texInfo >= 0
                        && static_cast<size_t>(side.texInfo) < bsp.get_texinfos().size()) {
                        const ::BSP::TexInfo& texInfo = bsp.get_texinfos()[side.texInfo];
                        surfaceFlags = texInfo.flags;
                        materialName = bsp.get_texture_name(texInfo.texData);
                    }

                    if (!GeometryRules::world_brush_side_blocks_light(
                            brush.contents,
                            surfaceFlags,
                            materialName)) {
                        continue;
                    }

                    std::vector<Common::Vec3f> winding = make_plane_winding(planes[side.planeNum]);
                    for (size_t clipSide = 0; clipSide < numSides && winding.size() >= 3; ++clipSide) {
                        const size_t otherSideIndex = firstSide + clipSide;
                        if (otherSideIndex == sideIndex) {
                            continue;
                        }

                        const ::BSP::DBrushSide& otherSide = sides[otherSideIndex];
                        if (otherSide.bevel) {
                            continue;
                        }
                        if (otherSide.planeNum >= planes.size()) {
                            winding.clear();
                            break;
                        }

                        winding = clip_winding_to_plane(
                            winding,
                            flipped_plane(planes, otherSide.planeNum)
                        );
                    }

                    if (winding.size() < 3) {
                        continue;
                    }

                    const Common::Vec3f a = winding[0];
                    for (size_t i = 1; i + 1 < winding.size(); ++i) {
                        const Common::Vec3f b = winding[i];
                        const Common::Vec3f c = winding[i + 1];
                        const Common::Vec3f crossVec =
                            Common::cross(Common::sub(b, a), Common::sub(c, a));
                        if (Common::length(crossVec) <= 1e-4f) {
                            continue;
                        }

                        result.worldBrushTriangles.push_back(make_triangle(
                            a,
                            b,
                            c,
                            static_cast<uint32_t>(brushIndex),
                            GeometryRules::RTX_ROLE_OCCLUDER,
                            Common::OccluderTriangle::SourceKind::Brush,
                            surfaceFlags,
                            brush.contents
                        ));
                    }
                }
            }
        }

        void append_static_prop_triangles(
            const BSP::SourceMap& sourceMap,
            const SceneBuildOptions& options,
            SceneBuildResult& result
        )
        {
            const ::BSP::StaticPropData& staticProps = sourceMap.static_props();
            if (options.assetSearchRoots.empty()) {
                return;
            }

            for (const ::BSP::StaticPropLumpV5& prop : staticProps.props) {
                if (prop.flags & ::BSP::STATIC_PROP_NO_SHADOW) {
                    continue;
                }

                std::string modelName;
                if (!static_prop_model_name(staticProps, prop, modelName)) {
                    continue;
                }

                const StaticPropMesh& mesh =
                    get_static_prop_mesh(options.assetSearchRoots, modelName);
                if (!mesh.loaded) {
                    continue;
                }

                for (const StaticPropTriangle& localTri : mesh.triangles) {
                    const Common::Vec3f tv0 = transform_static_prop_point(localTri.v0, prop);
                    const Common::Vec3f tv1 = transform_static_prop_point(localTri.v1, prop);
                    const Common::Vec3f tv2 = transform_static_prop_point(localTri.v2, prop);

                    const Common::Vec3f crossVec =
                        Common::cross(Common::sub(tv1, tv0), Common::sub(tv2, tv0));
                    if (Common::length(crossVec) <= 1e-4f) {
                        continue;
                    }

                    result.staticPropTriangles.push_back(make_triangle(
                        tv0,
                        tv1,
                        tv2,
                        static_cast<uint32_t>(&prop - staticProps.props.data()),
                        GeometryRules::RTX_ROLE_OCCLUDER | GeometryRules::RTX_ROLE_STATIC_PROP,
                        Common::OccluderTriangle::SourceKind::StaticProp,
                        0,
                        ::BSP::CONTENTS_SOLID | ::BSP::CONTENTS_MOVEABLE | ::BSP::CONTENTS_OPAQUE
                    ));
                }

                Common::StaticPropLightingProp lightingProp;
                lightingProp.propIndex =
                    static_cast<size_t>(&prop - staticProps.props.data());
                lightingProp.modelChecksum = mesh.checksum;
                lightingProp.flags = prop.flags;
                lightingProp.hasLightingOrigin =
                    (prop.flags & ::BSP::STATIC_PROP_USE_LIGHTING_ORIGIN) != 0;
                lightingProp.origin = Common::make_vec3(
                    prop.origin.x,
                    prop.origin.y,
                    prop.origin.z
                );
                lightingProp.lightingOrigin = Common::make_vec3(
                    prop.lightingOrigin.x,
                    prop.lightingOrigin.y,
                    prop.lightingOrigin.z
                );

                for (const StaticPropLightingMeshData& meshData : mesh.lightingMeshes) {
                    if (meshData.vertices.empty()) {
                        continue;
                    }

                    Common::StaticPropLightingMesh lightingMesh;
                    lightingMesh.lod = meshData.lod;
                    lightingMesh.vertices.reserve(meshData.vertices.size());
                    lightingMesh.colors.resize(
                        meshData.vertices.size(),
                        Common::make_vec3(0.0f, 0.0f, 0.0f)
                    );

                    for (const Common::StaticPropLightingVertex& vertex : meshData.vertices) {
                        lightingMesh.vertices.push_back(Common::StaticPropLightingVertex{
                            transform_static_prop_point(vertex.pos, prop),
                            transform_static_prop_normal(vertex.normal, prop)
                        });
                    }

                    lightingProp.meshes.push_back(std::move(lightingMesh));
                }

                if (!lightingProp.meshes.empty()) {
                    result.staticPropLightingProps.push_back(std::move(lightingProp));
                }
            }
        }
    }

    SceneBuildResult build_scene(
        const BSP::SourceMap& sourceMap,
        const std::vector<Geometry::FaceGeometry>& faceGeometry,
        const std::vector<Geometry::DispGeometry>& dispGeometry,
        const SceneBuildOptions& options
    )
    {
        SceneBuildResult result;
        const ::BSP::BSP& bsp = sourceMap.raw_bsp();

        const std::vector<size_t> worldFaces = sourceMap.world_face_indices();
        for (size_t faceIndex : worldFaces) {
            if (faceIndex >= faceGeometry.size()) {
                continue;
            }

            const Geometry::FaceGeometry& geometry = faceGeometry[faceIndex];
            if (geometry.isDisplacement) {
                continue;
            }

            const ::BSP::Face& face = sourceMap.face(faceIndex);
            const int32_t flags = face.get_texinfo().flags;
            const std::string materialName =
                bsp.get_texture_name(face.get_texinfo().texData);
            const uint32_t role = surface_role_from_flags(flags, false, materialName);
            if (!(role & (GeometryRules::RTX_ROLE_OCCLUDER | GeometryRules::RTX_ROLE_SKY))) {
                continue;
            }

            if (!geometry.valid && !(role & GeometryRules::RTX_ROLE_SKY)) {
                continue;
            }

            const Common::Vec3f modelOrigin = geometry.valid
                ? geometry.modelOrigin
                : Geometry::face_model_origin(sourceMap, faceIndex);

            const std::vector<Common::Vec3f> winding =
                Geometry::face_world_winding(sourceMap, faceIndex, modelOrigin);
            if (winding.size() < 3) {
                continue;
            }

            for (size_t i = 1; i + 1 < winding.size(); ++i) {
                result.worldFaceTriangles.push_back(make_triangle(
                    winding[0],
                    winding[i],
                    winding[i + 1],
                    static_cast<uint32_t>(faceIndex),
                    role,
                    Common::OccluderTriangle::SourceKind::Face,
                    flags,
                    0
                ));
            }
        }

        if (options.includeWorldBrushSideOccluders) {
            append_world_brush_triangles(sourceMap, result);
        }

        for (size_t faceIndex = 0; faceIndex < dispGeometry.size(); ++faceIndex) {
            const Geometry::DispGeometry& geometry = dispGeometry[faceIndex];
            if (!geometry.valid || geometry.gridSize < 2) {
                continue;
            }

            const ::BSP::Face& face = sourceMap.face(faceIndex);
            const int32_t flags = face.get_texinfo().flags;
            const std::string materialName =
                bsp.get_texture_name(face.get_texinfo().texData);
            const uint32_t role = surface_role_from_flags(flags, true, materialName);
            if (!(role & (GeometryRules::RTX_ROLE_OCCLUDER | GeometryRules::RTX_ROLE_SKY))) {
                continue;
            }

            for (size_t y = 0; y + 1 < geometry.gridSize; ++y) {
                for (size_t x = 0; x + 1 < geometry.gridSize; ++x) {
                    const size_t i00 = y * geometry.gridSize + x;
                    const size_t i10 = y * geometry.gridSize + (x + 1);
                    const size_t i11 = (y + 1) * geometry.gridSize + (x + 1);
                    const size_t i01 = (y + 1) * geometry.gridSize + x;
                    const bool odd = (((y * geometry.gridSize) + x) & 1u) != 0;

                    if (odd) {
                        result.displacementTriangles.push_back(make_triangle(
                            geometry.surfaceVertices[i00].pos,
                            geometry.surfaceVertices[i01].pos,
                            geometry.surfaceVertices[i10].pos,
                            static_cast<uint32_t>(faceIndex),
                            role,
                            Common::OccluderTriangle::SourceKind::Displacement,
                            flags,
                            0
                        ));
                        result.displacementTriangles.push_back(make_triangle(
                            geometry.surfaceVertices[i01].pos,
                            geometry.surfaceVertices[i11].pos,
                            geometry.surfaceVertices[i10].pos,
                            static_cast<uint32_t>(faceIndex),
                            role,
                            Common::OccluderTriangle::SourceKind::Displacement,
                            flags,
                            0
                        ));
                    }
                    else {
                        result.displacementTriangles.push_back(make_triangle(
                            geometry.surfaceVertices[i00].pos,
                            geometry.surfaceVertices[i01].pos,
                            geometry.surfaceVertices[i11].pos,
                            static_cast<uint32_t>(faceIndex),
                            role,
                            Common::OccluderTriangle::SourceKind::Displacement,
                            flags,
                            0
                        ));
                        result.displacementTriangles.push_back(make_triangle(
                            geometry.surfaceVertices[i00].pos,
                            geometry.surfaceVertices[i11].pos,
                            geometry.surfaceVertices[i10].pos,
                            static_cast<uint32_t>(faceIndex),
                            role,
                            Common::OccluderTriangle::SourceKind::Displacement,
                            flags,
                            0
                        ));
                    }
                }
            }
        }

        append_static_prop_triangles(sourceMap, options, result);
        return result;
    }
}
