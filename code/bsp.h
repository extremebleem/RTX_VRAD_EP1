#ifndef __BSP_H_
#define __BSP_H_

#include <cstdint>
#include <cstdlib>

#include <string>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <memory>

#include <fstream>

#include "include/gmtl/Matrix.h"


namespace BSP {
    inline constexpr uint32_t make_id(
            uint32_t a, uint32_t b, uint32_t c, uint32_t d
            ) {
        return static_cast<uint32_t>(a | (b << 8) | (c << 16) | (d << 24));
    }

    const uint32_t IDBSPHEADER = make_id('V', 'B', 'S', 'P');
    const size_t HEADER_LUMPS = 64;
    const size_t MAX_MAP_PLANES = 65536;
    const size_t MAX_MAP_EDGES = 256000;
    const size_t MAX_MAP_SURFEDGES = 512000;
    const size_t MAX_MAP_TEXINFO = 12288;
    const size_t MAX_MAP_TEXDATA = 2048;
    const size_t MAX_MAP_TEXDATA_STRING_DATA = 256000;
    const size_t TEXTURE_NAME_LENGTH = 128;
    const size_t MAX_LUXELS_PER_FACE = 32 * 32;

    /* Standard Gamma-correction constants */
    const double GAMMA = 2.2;
    const double INV_GAMMA = 1.0 / GAMMA;

    /* Arbitrary "small" value */
    const double EPSILON = 1e-3;

    struct PendingLump {
        std::vector<uint8_t> data;
        int version = 0;
        int alignment = 4;
        bool clearFourCC = true;
    };

    static std::unordered_map<int, PendingLump> m_pendingLumps;

    enum LumpType
    {
        LUMP_ENTITIES = 0,	// *
        LUMP_PLANES = 1,	// *
        LUMP_TEXDATA = 2,	// JAY: This is texdata now, previously LUMP_TEXTURES
        LUMP_VERTEXES = 3,	// *
        LUMP_VISIBILITY = 4,	// *
        LUMP_NODES = 5,	// *
        LUMP_TEXINFO = 6,	// *
        LUMP_FACES = 7,	// *
        LUMP_LIGHTING = 8,	// *
        LUMP_OCCLUSION = 9,
        LUMP_LEAFS = 10,	// *
        // UNUSED
        LUMP_EDGES = 12,	// *
        LUMP_SURFEDGES = 13,	// *
        LUMP_MODELS = 14,	// *
        LUMP_WORLDLIGHTS = 15,	// 
        LUMP_LEAFFACES = 16,	// *
        LUMP_LEAFBRUSHES = 17,	// *
        LUMP_BRUSHES = 18,	// *
        LUMP_BRUSHSIDES = 19,	// *
        LUMP_AREAS = 20,	// *
        LUMP_AREAPORTALS = 21,	// *
        LUMP_PORTALS = 22,
        LUMP_CLUSTERS = 23,
        LUMP_PORTALVERTS = 24,
        LUMP_CLUSTERPORTALS = 25,
        LUMP_DISPINFO = 26,
        LUMP_ORIGINALFACES = 27,
        // UNUSED
        LUMP_PHYSCOLLIDE = 29,
        LUMP_VERTNORMALS = 30,
        LUMP_VERTNORMALINDICES = 31,
        LUMP_DISP_LIGHTMAP_ALPHAS = 32,
        LUMP_DISP_VERTS = 33,		// CDispVerts
        LUMP_DISP_LIGHTMAP_SAMPLE_POSITIONS = 34,	// For each displacement
        //     For each lightmap sample
        //         byte for index
        //         if 255, then index = next byte + 255
        //         3 bytes for barycentric coordinates
        // The game lump is a method of adding game-specific lumps
        // FIXME: Eventually, all lumps could use the game lump system
        LUMP_GAME_LUMP = 35,
        LUMP_LEAFWATERDATA = 36,
        LUMP_PRIMITIVES = 37,
        LUMP_PRIMVERTS = 38,
        LUMP_PRIMINDICES = 39,
        // A pak file can be embedded in a .bsp now, and the file system will search the pak
        //  file first for any referenced names, before deferring to the game directory 
        //  file system/pak files and finally the base directory file system/pak files.
        LUMP_PAKFILE = 40,
        LUMP_CLIPPORTALVERTS = 41,
        // A map can have a number of cubemap entities in it which cause cubemap renders
        // to be taken after running vrad.
        LUMP_CUBEMAPS = 42,
        LUMP_TEXDATA_STRING_DATA = 43,
        LUMP_TEXDATA_STRING_TABLE = 44,
        LUMP_OVERLAYS = 45,
        LUMP_LEAFMINDISTTOWATER = 46,
        LUMP_FACE_MACRO_TEXTURE_INFO = 47,
        LUMP_DISP_TRIS = 48,
        LUMP_PHYSCOLLIDESURFACE = 49,	// deprecated.  We no longer use win32-speicifc havok compression on terrain
        LUMP_WATEROVERLAYS = 50,
        LUMP_LIGHTMAPPAGES = 51,	// xbox: alternate lightdata implementation
        LUMP_LIGHTMAPPAGEINFOS = 52,	// xbox: indexed by faces for alternate lightdata

        // optional lumps for HDR
        LUMP_LIGHTING_HDR = 53,
        LUMP_WORLDLIGHTS_HDR = 54,
        LUMP_LEAF_AMBIENT_LIGHTING_HDR = 55,	// NOTE: this data overrides part of the data stored in LUMP_LEAFS.
        LUMP_LEAF_AMBIENT_LIGHTING = 56,	// NOTE: this data overrides part of the data stored in LUMP_LEAFS.

        LUMP_XZIPPAKFILE = 57,   // xbox: maps may now have xzp's instead of zips stored. 
        LUMP_FACES_HDR = 58,	// HDR maps may have different face data.
        LUMP_MAP_FLAGS = 59,   // extended level-wide flags. not present in all levels

    };

    static_assert(LUMP_ENTITIES == 0, "Lump miscount!");

    struct Lump {
        int32_t fileOffset;
        int32_t fileLen;
        int32_t version;
        uint8_t fourCC[4];
    };

    struct Header {
        int32_t ident;
        int32_t version;
        Lump lumps[HEADER_LUMPS];
        int32_t mapRevision;
    };

    template<typename T>
    struct Vec3 {
        T x;
        T y;
        T z;
    };

    struct DVis {
        int32_t numClusters;
        int32_t bitofs[8][2];
    };

    struct DPlane {
        Vec3<float> normal;
        float dist;
        int32_t type;
    };

    struct DEdge {
        uint16_t vertex1;
        uint16_t vertex2;
    };

    struct DFace {
        uint16_t planeNum;
        uint8_t side;
        uint8_t onNode;
        int32_t firstEdge;
        int16_t numEdges;
        int16_t texInfo;
        int16_t dispInfo;
        int16_t surfaceFogVolumeID;
        uint8_t styles[4];
        int32_t lightOffset;
        float area;
        int32_t lightmapTextureMinsInLuxels[2];
        int32_t lightmapTextureSizeInLuxels[2];
        int32_t origFace;
        uint16_t numPrims;
        uint16_t firstPrimID;
        uint32_t smoothingGroups;
    };

    enum TexInfoFlag {
        SURF_LIGHT = 0x1,
        SURF_SKY2D = 0x2,
        SURF_SKY = 0x4,
        SURF_WARP = 0x8,
        SURF_TRANS = 0x10,
        SURF_NOPORTAL = 0x20,
        SURF_TRIGGER = 0x40,
        SURF_NODRAW = 0x80,
        SURF_HINT = 0x100,
        SURF_SKIP = 0x200,
        SURF_NOLIGHT = 0x400,
        SURF_BUMPLIGHT = 0x800,
        SURF_NOSHADOWS = 0x1000,
        SURF_NODECALS = 0x2000,
        SURF_NOCHOP = 0x4000,
        SURF_HITBOX = 0x8000,
    };

    struct TexInfo {
        float textureVecs[2][4];
        float lightmapVecs[2][4];
        int32_t flags;
        int32_t texData;
    };

    struct DTexData {
        Vec3<float> reflectivity;
        int32_t nameStringTableID;
        int32_t width;
        int32_t height;
        int32_t view_width;
        int32_t view_height;
    };

    struct DModel {
        Vec3<float> mins;
        Vec3<float> maxs;
        Vec3<float> origin;
        int32_t headNode;
        int32_t firstFace;
        int32_t numFaces;
    };

    enum LeafContents {
        CONTENTS_SOLID = 0x1,
        CONTENTS_WINDOW = 0x2,
        CONTENTS_AUX = 0x4,
        CONTENTS_GRATE = 0x8,
        CONTENTS_SLIME = 0x10,
        CONTENTS_WATER = 0x20,
        CONTENTS_BLOCKLOS = 0x40,
        CONTENTS_OPAQUE = 0x80,
        LAST_VISIBLE_CONTENTS = 0x80,
        ALL_VISIBLE_CONTENTS
            = (LAST_VISIBLE_CONTENTS | (LAST_VISIBLE_CONTENTS - 1)),
        CONTENTS_TESTFOGVOLUME = 0x100,
        CONTENTS_UNUSED = 0x200,
        CONTENTS_UNUSED6 = 0x400,
        CONTENTS_TEAM1 = 0x800,
        CONTENTS_TEAM2 = 0x1000,
        CONTENTS_IGNORE_NODRAW_OPAQUE = 0x2000,
        CONTENTS_MOVEABLE = 0x4000,
        CONTENTS_AREAPORTAL = 0x8000,
        CONTENTS_PLAYERCLIP = 0x10000,
        CONTENTS_MONSTERCLIP = 0x20000,
        CONTENTS_CURRENT_0 = 0x40000,
        CONTENTS_CURRENT_90 = 0x80000,
        CONTENTS_CURRENT_180 = 0x100000,
        CONTENTS_CURRENT_270 = 0x200000,
        CONTENTS_CURRENT_UP = 0x400000,
        CONTENTS_CURRENT_DOWN = 0x800000,
        CONTENTS_ORIGIN = 0x1000000,
        CONTENTS_MONSTER = 0x2000000,
        CONTENTS_DEBRIS = 0x4000000,
        CONTENTS_DETAIL = 0x8000000,
        CONTENTS_TRANSLUCENT = 0x10000000,
        CONTENTS_LADDER = 0x20000000,
        CONTENTS_HITBOX = 0x40000000,
    };

    struct DNode {
        int32_t planeNum;
        int32_t children[2];
        int16_t mins[3];
        int16_t maxs[3];
        uint16_t firstFace;
        uint16_t numFaces;
        int16_t area;
    };

    struct DLeaf {
        uint32_t contents;
        int16_t cluster;
        int16_t area:9;
        int16_t flags:7;
        int16_t mins[3];
        int16_t maxs[3];
        uint16_t firstLeafFace;
        uint16_t numLeafFaces;
        uint16_t firstLeafBrush;
        uint16_t numLeafBrushes;
        int16_t leafWaterDataID;
    };

    enum EmitType {
        EMIT_SURFACE,
        EMIT_POINT,
        EMIT_SPOTLIGHT,
        EMIT_SKYLIGHT,
        EMIT_QUAKELIGHT,
        EMIT_SKYAMBIENT,
    };

    static_assert(EMIT_SURFACE == 0, "EmitType miscount!");
    static_assert(EMIT_SKYAMBIENT == 5, "EmitType miscount!");

    struct DWorldLight {
        Vec3<float> origin;
        Vec3<float> intensity;
        Vec3<float> normal;
        int32_t cluster;
        EmitType type;
        int32_t style;
        float stopdot;
        float stopdot2;
        float exponent;
        float radius;
        float constantAtten;
        float linearAtten;
        float quadraticAtten;
        int32_t flags;
        int32_t texinfo;
        int32_t owner;
    };

    struct RGBExp32 {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        int8_t exp;
    };

    struct CompressedLightCube {
        RGBExp32 color[6];
    };

    enum GameLumpID {
        GAMELUMP_DETAIL_PROPS = 'dprp',
        GAMELUMP_DETAIL_PROP_LIGHTING = 'dplt',
        GAMELUMP_STATIC_PROPS = 'sprp',
        GAMELUMP_DETAIL_PROP_LIGHTING_HDR = 'dplh',
    };

    struct GameLump {
        int32_t id;
        uint16_t flags;
        uint16_t version;
        int32_t fileOffset;
        int32_t fileLen;
    };

    struct GameLumpHeader {
        int32_t lumpCount;
    };

    struct Edge {
        Vec3<float> vertex1;
        Vec3<float> vertex2;
    };

    class BSP;

    class Entity {
        private:
            std::unordered_map<std::string, std::string> m_data;

        public:
            Entity();

            const std::string& get(const std::string& key) const;
            const std::string& get(
                const std::string& key,
                const std::string& defaultVal
            ) const;

            bool has_key(const std::string& key) const;

            void set(const std::string& key, const std::string& value);

            size_t size(void) const;
    };

    class Light {
        private:
            Vec3<float> m_coords;
            int16_t m_cluster;

        public:
            double r;
            double g;
            double b;

            double c;
            double l;
            double q;

            float stopdot;
            float stopdot2;
            float exponent;
            float radius;
            int32_t style;

            double innerCone;
            double outerCone;
            Vec3<double> direction;

            EmitType emitType;

            Light(const BSP& bsp, const Entity& entity);

            bool parse_light_value_raw_optional(const std::string& value);

            inline double attenuate(double distance) const {
                return c + l * distance + q * distance * distance;
            };
             

            void parse_light_value_raw(const std::string& value);

            const Vec3<float>& get_coords(void) const;
            void parse_color(const std::string& value);
            DWorldLight to_worldlight(void) const;

        private:
            
            void parse_direction(const Entity& entity);
   
            void parse_falloff(const Entity& entity);

            void parse_generic(const BSP& bsp, const Entity& entity);
            void parse_point(const BSP& bsp, const Entity& entity);
            void parse_spot(const BSP& bsp, const Entity& entity);
            void parse_environment(const BSP& bsp, const Entity& entity);
    };

    class FaceLightSampleProxy {
        public:
            using Iter = std::vector<RGBExp32>::iterator;
            using ConstIter = std::vector<RGBExp32>::const_iterator;

        private:
            Iter m_begin;
            Iter m_end;

        public:
            FaceLightSampleProxy(
                Iter& begin,
                Iter& end
            );

            RGBExp32& operator[](size_t i);
            const RGBExp32& operator[](size_t i) const;

            Iter begin(void);
            Iter end(void);

    };

    class Face {
        private:
            static size_t s_faceCount;

            BSP& m_bsp;

            DFace& m_faceData;
            DPlane& m_planeData;
            TexInfo& m_texInfo;
            DTexData& m_texData;

            gmtl::Matrix<double, 3, 3> m_Ainv;

            std::vector<Edge> m_edges;

            //LightSample& m_avgLightSample;

            void load_edges(
                const DFace& faceData,
                const std::vector<Vec3<float>>& vertices,
                const std::vector<DEdge>& dEdges,
                const std::vector<int32_t>& surfEdges
            );

        public:
            const size_t id;

            Face(BSP& bsp, DFace& faceData);

            void make_st_xyz_matrix(gmtl::Matrix<double, 3, 3>& Ainv) const;

            const DFace& get_data(void) const;
            const DPlane& get_planedata(void) const;
            const TexInfo& get_texinfo(void) const;
            const DTexData& get_texdata(void) const;
            const gmtl::Matrix<double, 3, 3>& get_st_xyz_matrix(void) const;

            void set_texinfo_index(int16_t index);
            void set_texdata_index(int32_t index);

            const std::vector<Edge>& get_edges(void) const;

            const std::vector<uint8_t> get_styles(void) const;
            void set_styles(const std::vector<uint8_t>& styles);

            int32_t get_lightmap_width(void) const;
            int32_t get_lightmap_height(void) const;
            size_t get_lightmap_size(void) const;
            size_t get_lightmap_offset(void) const;
            void set_lightmap_offset(size_t offset);

            FaceLightSampleProxy get_lightsamples(void);

            RGBExp32 get_average_lighting(void) const;
            void set_average_lighting(const RGBExp32& sample);

            Vec3<float> xyz_from_lightmap_st(float s, float t) const;
    };

    class BSP {
        friend Face::Face(BSP&, DFace&);

        public:
            using VisMatrix = std::vector<std::vector<uint8_t>>;

        private:
            Header m_header;

            std::vector<DModel> m_models;
            std::vector<DPlane> m_planes;
            std::vector<Vec3<float>> m_vertices;
            std::vector<DEdge> m_edges;
            std::vector<int32_t> m_surfEdges;
            std::vector<DFace> m_dFaces;
            std::vector<RGBExp32> m_lightSamples;
            std::vector<TexInfo> m_texInfos;
            std::vector<DTexData> m_texDatas;
            std::vector<Face> m_faces;
            std::vector<DNode> m_nodes;
            std::vector<DLeaf> m_leaves;
            std::vector<CompressedLightCube> m_ambientLightSamples;
            std::vector<DWorldLight> m_worldLights;

            std::string m_entData;
            std::vector<Light> m_lights;

            // Original raw visibility lump data, because I'm too lazy to
            // actually bother writing the visibility lump compression
            // algorithm myself.
            std::vector<uint8_t> m_visLumpData;

            // Decompressed visibility matrix
            VisMatrix m_visibility;

            std::unordered_map<int, std::vector<uint8_t>> m_extraLumps;
            std::unordered_map<int32_t, std::vector<uint8_t>> m_extraGameLumps;

            std::unordered_set<int> m_loadedLumps;

            std::map<int32_t, GameLump> m_gameLumps;

            bool m_fullbright;

            bool m_hasHDR;

            void init(std::ifstream& file);

            template<typename Container>
            void load_lump(
                std::ifstream& file,
                const LumpType lumpID,
                Container& dest
            );

            void load_lights(const std::string& entData);
            void load_visibility(std::ifstream& file);
            void load_extras(std::ifstream& file);

            void load_gamelumps(std::ifstream& file);

            template<typename Container>
            void load_single_gamelump(
                std::ifstream& file,
                const GameLump& gameLump,
                Container& dest
            );

            template<typename Container>
            void save_lump(
                std::ofstream& file,
                const LumpType lumpID,
                const Container& src,
                bool isExtraLump = false
            );

            void write_pending_lumps(std::ofstream& file);

            //void save_faces(
            //    std::ofstream& file,
            //    std::unordered_map<int, std::ofstream::off_type>& offsets,
            //    std::unordered_map<int, size_t>& sizes
            //);

            void save_lights(std::ofstream& file);

            void save_visibility(std::ofstream& file);

            void save_extras(std::ofstream& file);

            void save_gamelumps(std::ofstream& file);

            template<typename Container>
            void save_single_gamelump(
                std::ofstream& file,
                int32_t gameLumpID,
                const Container& src
            );

        public:
            BSP();
            BSP(const std::string& filename);
            BSP(std::ifstream& file);

            // Disallow copy/move semantics for now, because that completely
            // *wrecks* all of the Face helper objects' internal references,
            // and I don't want to have to deal with that at the moment.
            BSP(const BSP& other) = delete;
            BSP& operator=(const BSP& other) = delete;

            int get_format_version(void) const;

            const Header& get_header(void) const;

            const std::vector<DModel>& get_models(void) const;
            const std::vector<DPlane>& get_planes(void) const;
            const std::vector<Vec3<float>>& get_vertices(void) const;
            const std::vector<DEdge>& get_edges(void) const;
            const std::vector<int32_t>& get_surfedges(void) const;
            std::vector<DFace>& get_dfaces(void);
            const std::vector<DFace>& get_dfaces(void) const;
            std::vector<RGBExp32>& get_lightsamples(void);
            const std::vector<RGBExp32>& get_lightsamples(void) const;
            const std::vector<TexInfo>& get_texinfos(void) const;
            const std::vector<DTexData>& get_texdatas(void) const;

            std::vector<Face>& get_faces(void);
            const std::vector<Face>& get_faces(void) const;

            const std::vector<DNode>& get_nodes(void) const;
            const std::vector<DLeaf>& get_leaves(void) const;

            std::vector<CompressedLightCube>& get_ambient_samples(void);
            const std::vector<CompressedLightCube>& get_ambient_samples(void) const;

            const std::vector<DWorldLight>& get_worldlights(void) const;
            const std::vector<Light>& get_lights(void) const;

            const std::string& get_entdata(void);

            const VisMatrix& get_visibility(void) const;
            int16_t cluster_for_pos(const Vec3<float>& pos) const;

            const std::unordered_map<int, std::vector<uint8_t>>&
                get_extras(void) const;

            void build_worldlights(void);
            void init_ambient_samples(void);

            bool is_fullbright(void) const;
            void set_fullbright(bool fullbright);

            bool is_hdr(void) const;
            void set_hdr(bool hdr);

            bool has_visibility_data(void) const;

            void dump_lumps() const;

            void write(const std::string& filename);
            void write(std::ofstream& file);
    };

    class InvalidBSP : public std::runtime_error {
        public:
            InvalidBSP(const std::string& what) : std::runtime_error(what) {}
    };

    class EntityParser {
        private:
            int m_index;
            std::string m_entData;

        public:
            EntityParser(const std::string& entData);

            Entity next_ent(void);
    };
}

#endif
