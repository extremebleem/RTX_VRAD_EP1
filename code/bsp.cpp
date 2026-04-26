#include <iostream>
#include <sstream>
#include <fstream>

#include <string>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <limits>
#include <algorithm>

#include <cstring>
#include <cassert>
#include <cmath>

#include "include/gmtl/Matrix.h"
#include "include/gmtl/MatrixOps.h"

#include "bsp.h"
#include "bsp_shared.h"


namespace BSP {

    /*************
     * BSP Class *
     *************/

    BSP::BSP() : m_fullbright(true) {}

    BSP::BSP(const std::string& filename) : BSP() {
        std::ifstream file(filename, std::ios::binary);

        if (file.fail()) {
            throw InvalidBSP("Could not open " + filename + "!");
        }

        init(file);
    }

    BSP::BSP(std::ifstream& file) : BSP() {
        init(file);
    }

    void BSP::init(std::ifstream& file) {
        file.read(reinterpret_cast<char*>(&m_header), sizeof(Header));

        if (m_header.ident != IDBSPHEADER) {
            throw InvalidBSP("Bad BSP! :(");
        }

        int version = get_format_version();

        if (version < 19 || version > 21) {
            std::stringstream s;

            s << "Unsupported BSP version: " << version << " (22 > supported > 18)";

            throw InvalidBSP(s.str());
        }

        const bool isHDR = m_header.lumps[LUMP_LIGHTING_HDR].fileLen != 0;

        set_hdr(isHDR);

        load_lump(file, LUMP_MODELS, m_models);
        load_lump(file, LUMP_PLANES, m_planes);
        load_lump(file, LUMP_VERTEXES, m_vertices);
        load_lump(file, LUMP_EDGES, m_edges);
        load_lump(file, LUMP_SURFEDGES, m_surfEdges);

        if (isHDR) {
            load_lump(file, LUMP_FACES_HDR, m_dFaces);
        }
        if (m_dFaces.empty()) {
            load_lump(file, LUMP_FACES, m_dFaces);
        }

        if (isHDR) {
            load_lump(file, LUMP_LIGHTING_HDR, m_lightSamples);
        }
        if (m_lightSamples.empty()) {
            load_lump(file, LUMP_LIGHTING, m_lightSamples);
        }

        set_fullbright(m_lightSamples.empty());

        if (!is_fullbright()) {
            if (isHDR) {
                load_lump(
                    file,
                    LUMP_LEAF_AMBIENT_LIGHTING_HDR,
                    m_ambientLightSamples
                );
            }

            if (m_ambientLightSamples.empty()) {
                load_lump(
                    file,
                    LUMP_LEAF_AMBIENT_LIGHTING,
                    m_ambientLightSamples
                );
            }
        }

        load_lump(file, LUMP_TEXINFO, m_texInfos);
        load_lump(file, LUMP_TEXDATA, m_texDatas);

        for (DFace& faceData : m_dFaces) {
            //std::cout << faceData.texInfo << std::endl;

            m_faces.push_back(Face(*this, faceData));

            if (is_fullbright()) {
                Face& face = m_faces.back();

                // Average lighting entry
                m_lightSamples.push_back(RGBExp32 {0, 0, 0, 0});

                // The lightmap offset always points to the entry *after* the
                // average lighting entry.
                face.set_lightmap_offset(m_lightSamples.size());

                // The rest of the lightmap
                for (size_t i=0; i<face.get_lightmap_size(); i++) {
                    m_lightSamples.push_back(RGBExp32 {0, 0, 0, 0});
                }
            }
        }

        load_lump(file, LUMP_NODES, m_nodes);
        load_lump(file, LUMP_LEAFS, m_leaves);

        std::vector<char> entData;
        load_lump(file, LUMP_ENTITIES, entData);

        m_entData.assign(entData.begin(), entData.end());

        load_lights(m_entData);

        if (!is_fullbright()) {
            if (isHDR) {
                load_lump(file, LUMP_WORLDLIGHTS_HDR, m_worldLights);
            }
            
            if (m_worldLights.empty()) {
                load_lump(file, LUMP_WORLDLIGHTS, m_worldLights);
            }
        }

        load_visibility(file);
        load_gamelumps(file);

        load_extras(file);
    }

    template<typename Container>
    void BSP::load_lump(
            std::ifstream& file,
            const LumpType lumpID,
            Container& dest
            ) {

        Lump& lump = m_header.lumps[lumpID];
        
        size_t lumpSize = lump.fileLen;

        if (lumpSize == 0)
            return;

        std::ifstream::off_type offset = lump.fileOffset;
        size_t numElems = lumpSize / sizeof(typename Container::value_type);

        if (lumpSize % sizeof(typename Container::value_type))
        {
            printf("ValidateLump: odd size for lump %d", lumpID);
            assert(0 && "odd size for lump");
        }

        dest.resize(numElems);

        file.seekg(offset);
        file.read(reinterpret_cast<char*>(dest.data()), lumpSize);

        m_loadedLumps.insert(lumpID);
    }

    void BSP::load_lights(const std::string& entData) {
        EntityParser entParser(entData);

        Entity entity;

        while ((entity = entParser.next_ent()).size() > 0) {
            assert(entity.has_key("classname"));

            const std::string& classname = entity.get("classname");

            if (classname == "light" ||
                classname == "light_spot" ||
                classname == "light_environment") {
                m_lights.push_back(Light(*this, entity));

                if (classname == "light_environment") {
                    Light ambient(*this, entity);
                    ambient.emitType = EMIT_SKYAMBIENT;

                    if (entity.has_key("_ambient")) {
                        ambient.parse_color(entity.get("_ambient"));
                    }
                    else {
                        ambient.r *= 0.5;
                        ambient.g *= 0.5;
                        ambient.b *= 0.5;
                    }

                    m_lights.push_back(ambient);
                }
            }
        }
    }

    void BSP::load_visibility(std::ifstream& file) {
        load_lump(file, LUMP_VISIBILITY, m_visLumpData);

        if (m_visLumpData.size() < sizeof(DVis)) {
            // There is no visibility data available. Welp.
            return;
        }

        DVis& visLump = *reinterpret_cast<DVis*>(m_visLumpData.data());
        int32_t numClusters = visLump.numClusters;

        int32_t (*byteOffsetPairs)[2] = visLump.bitofs;

        m_visibility.resize(numClusters);

        /* Decompress the visibility matrix */
        for (int cluster1=0; cluster1<numClusters; cluster1++) {
            int32_t pvsOffset = byteOffsetPairs[cluster1][0];
            std::vector<uint8_t>& clusterVis = m_visibility[cluster1];

            int cluster2 = 0;
            while (cluster2 < numClusters) {
                uint8_t pvsByte = m_visLumpData[pvsOffset++];

                if (pvsByte == 0) {
                    uint8_t skipCount = m_visLumpData[pvsOffset++];
                    for (uint8_t i=0; i<skipCount; i++) {
                        clusterVis.push_back(0);
                        cluster2 += 8;
                    }
                }
                else {
                    clusterVis.push_back(pvsByte);
                    cluster2 += 8;
                }
            }
        }
    }

    void BSP::load_extras(std::ifstream& file) {
        for (unsigned int lumpID=0; lumpID<HEADER_LUMPS; lumpID++) {
            if (m_loadedLumps.find(lumpID) != m_loadedLumps.end()) {
                // Lump was previously loaded; ignore
                continue;
            }

            Lump& lump = m_header.lumps[lumpID];

            if (lump.fileOffset == 0 || lump.fileLen == 0) {
                // Unused lump; ignore
                continue;
            }

            std::vector<uint8_t> extraLumpData;

            load_lump(file, static_cast<LumpType>(lumpID), extraLumpData);

            m_extraLumps[lumpID] = std::move(extraLumpData);
        }
    }

    template<typename Container>
    void BSP::load_single_gamelump(
            std::ifstream& file,
            const GameLump& gameLump,
            Container& dest
            ) {

        std::ifstream::off_type offset = gameLump.fileOffset;
        size_t lumpSize = gameLump.fileLen;
        size_t numElems = lumpSize / sizeof(typename Container::value_type);

        dest.resize(numElems);

        file.seekg(offset);
        file.read(reinterpret_cast<char*>(dest.data()), lumpSize);
    }

    void BSP::load_gamelumps(std::ifstream& file) {
        std::vector<uint8_t> gameLumpData;
        load_lump(file, LUMP_GAME_LUMP, gameLumpData);

        GameLumpHeader& gameLumpHeader = *reinterpret_cast<GameLumpHeader*>(
            gameLumpData.data()
        );

        int32_t lumpCount = gameLumpHeader.lumpCount;

        GameLump* pGameLumps = (GameLump*)((&gameLumpHeader.lumpCount) + 1);

        for (int i=0; i<lumpCount; i++) {
            GameLump& gameLump = pGameLumps[i];

            switch (gameLump.id) {
                case GAMELUMP_STATIC_PROPS:
                // case GAMELUMP_STATIC_PROPS: {
                    // std::vector<>

                    // break;
                // }
                default: {
                    std::vector<uint8_t> extraGameLumpData;

                    load_single_gamelump(
                        file,
                        gameLump,
                        extraGameLumpData
                    );

                    m_extraGameLumps[gameLump.id]
                        = std::move(extraGameLumpData);
                }
            }

            m_gameLumps[gameLump.id] = gameLump;
        }
    }

    int BSP::get_format_version(void) const {
        return m_header.version;
    }

    const Header& BSP::get_header(void) const {
        return m_header;
    }

    const std::vector<DModel>& BSP::get_models(void) const {
        return m_models;
    }

    const std::vector<DPlane>& BSP::get_planes(void) const {
        return m_planes;
    }

    const std::vector<Vec3<float>>& BSP::get_vertices(void) const {
        return m_vertices;
    }

    const std::vector<DEdge>& BSP::get_edges(void) const {
        return m_edges;
    }

    const std::vector<int32_t>& BSP::get_surfedges(void) const {
        return m_surfEdges;
    }

    std::vector<DFace>& BSP::get_dfaces(void) {
        return m_dFaces;
    }

    const std::vector<DFace>& BSP::get_dfaces(void) const {
        return m_dFaces;
    }

    std::vector<RGBExp32>& BSP::get_lightsamples(void) {
        return m_lightSamples;
    }

    const std::vector<RGBExp32>& BSP::get_lightsamples(void) const {
        return m_lightSamples;
    }

    const std::vector<TexInfo>& BSP::get_texinfos(void) const {
        return m_texInfos;
    }

    const std::vector<DTexData>& BSP::get_texdatas(void) const {
        return m_texDatas;
    }

    std::vector<Face>& BSP::get_faces(void) {
        return m_faces;
    }

    const std::vector<Face>& BSP::get_faces(void) const {
        return m_faces;
    }

    const std::vector<DNode>& BSP::get_nodes(void) const {
        return m_nodes;
    }

    const std::vector<DLeaf>& BSP::get_leaves(void) const {
        return m_leaves;
    }

    std::vector<CompressedLightCube>& BSP::get_ambient_samples(void) {
        return m_ambientLightSamples;
    }

    const std::vector<CompressedLightCube>&
    BSP::get_ambient_samples(void) const {
        return m_ambientLightSamples;
    }

    const std::vector<DWorldLight>& BSP::get_worldlights(void) const {
        return m_worldLights;
    }

    const std::vector<Light>& BSP::get_lights(void) const {
        return m_lights;
    }

    const std::string& BSP::get_entdata(void) {
        return m_entData;
    }

    const BSP::VisMatrix& BSP::get_visibility(void) const {
        return m_visibility;
    }

    int16_t BSP::cluster_for_pos(const Vec3<float>& pos) const {
        return BSPShared::cluster_for_pos(*this, pos);
    }

    bool BSP::is_fullbright(void) const {
        return m_fullbright;
    }

    void BSP::set_fullbright(bool fullbright) {
        m_fullbright = fullbright;
    }

    bool BSP::is_hdr(void) const {
        return m_hasHDR;
    }

    void BSP::set_hdr(bool hdr) {
        m_hasHDR = hdr;
    }

    bool BSP::has_visibility_data(void) const {
        return get_visibility().size() > 0;
    }

    void BSP::write(const std::string& filename) {
        std::ofstream f(filename, std::ios::binary);
        write(f);
    }

    static std::ofstream::off_type align_file_position(
        std::ofstream& file,
        std::ofstream::off_type alignment
    ) {
        std::ofstream::off_type currPosition = file.tellp();

        if (alignment < 2) {
            return currPosition;
        }

        std::ofstream::off_type newPosition =
            ((currPosition + alignment - 1) / alignment) * alignment;

        std::ofstream::off_type count = newPosition - currPosition;

        if (count <= 0) {
            return currPosition;
        }

        char smallBuffer[4096] = {};

        if (count <= static_cast<std::ofstream::off_type>(sizeof(smallBuffer))) {
            file.write(smallBuffer, count);
        }
        else {
            std::vector<char> buffer(static_cast<size_t>(count), 0);
            file.write(buffer.data(), buffer.size());
        }

        return newPosition;
    }

    void BSP::write(std::ofstream& file) {
        file.seekp(0);
        file.write(reinterpret_cast<char*>(&m_header), sizeof(m_header));

        align_file_position(file, 4);

        save_lump(file, LUMP_MODELS, m_models);
        save_lump(file, LUMP_PLANES, m_planes);
        save_lump(file, LUMP_VERTEXES, m_vertices);
        save_lump(file, LUMP_EDGES, m_edges);
        save_lump(file, LUMP_SURFEDGES, m_surfEdges);

        std::vector<DFace> dFaces(m_dFaces);

        if (is_fullbright()) {
            for (DFace& dFace : dFaces) {
                dFace.lightOffset = -1;
            }
        }

        if (is_hdr()) {
            save_lump(file, LUMP_FACES_HDR, dFaces);

            if (!is_fullbright()) {
                save_lump(file, LUMP_LIGHTING_HDR, m_lightSamples);
            }
        }
        else {
            save_lump(file, LUMP_FACES, dFaces);

            if (!is_fullbright()) {
                save_lump(file, LUMP_LIGHTING, m_lightSamples);
            }
        }

        save_lump(file, LUMP_TEXINFO, m_texInfos);
        save_lump(file, LUMP_TEXDATA, m_texDatas);
        save_lump(file, LUMP_NODES, m_nodes);
        save_lump(file, LUMP_LEAFS, m_leaves);

        std::vector<char> entData(m_entData.begin(), m_entData.end());
        save_lump(file, LUMP_ENTITIES, entData);

        if (!is_fullbright()) {
            save_lights(file);

            if (is_hdr()) {
                save_lump(file, LUMP_LEAF_AMBIENT_LIGHTING_HDR, m_ambientLightSamples);
            }
            else {
                save_lump(file, LUMP_LEAF_AMBIENT_LIGHTING, m_ambientLightSamples);
            }
        }

        save_visibility(file);

        save_gamelumps(file);

        save_extras(file);

        file.seekp(0);
        file.write(reinterpret_cast<char*>(&m_header), sizeof(m_header));
    }

    template<typename Container>
    void BSP::save_lump(
        std::ofstream& file,
        const LumpType lumpID,
        const Container& src,
        bool isExtraLump) 
    {

        using ValueType = typename Container::value_type;

        const int len =
            static_cast<int>(src.size() * sizeof(ValueType));

        Lump& lump = m_header.lumps[lumpID];

        lump.fileOffset = len == 0 ? 0 : static_cast<int32_t>(file.tellp());
        lump.fileLen = len;

        if (src.empty())
            return;

        align_file_position(file, 1);

        if (len > 0) {
            file.write(reinterpret_cast<const char*>(src.data()), static_cast<std::streamsize>(len));
        }

        align_file_position(file, 4);

        if (!isExtraLump) {
            m_extraLumps.erase(lumpID);
        }
    }

    void BSP::write_pending_lumps(std::ofstream& file) {
        for (int lumpID = 0; lumpID < HEADER_LUMPS; ++lumpID) {
            auto it = m_pendingLumps.find(lumpID);

            if (it == m_pendingLumps.end()) {
                continue;
            }

            PendingLump& pending = it->second;
            Lump& lump = m_header.lumps[lumpID];

            align_file_position(file, pending.alignment);

            lump.fileOffset = static_cast<int32_t>(file.tellp());
            lump.fileLen = static_cast<int32_t>(pending.data.size());
            lump.version = pending.version;

            if (pending.clearFourCC) {
                lump.fourCC[0] = 0;
                lump.fourCC[1] = 0;
                lump.fourCC[2] = 0;
                lump.fourCC[3] = 0;
            }

            if (!pending.data.empty()) {
                file.write(
                    reinterpret_cast<const char*>(pending.data.data()),
                    static_cast<std::streamsize>(pending.data.size())
                );
            }

            align_file_position(file, 4);
        }
    }

    //void BSP::save_faces(
    //        std::ofstream& file,
    //        std::unordered_map<int, std::ofstream::off_type>& offsets,
    //        std::unordered_map<int, size_t>& sizes
    //        ) {
    //
    //    std::vector<LightSample> lightSamples;
    //    std::vector<DTexData> dTexDatas;
    //    std::vector<TexInfo> texInfos;
    //    std::vector<DFace> dFaces;
    //
    //    for (Face& face : m_faces) {
    //        if (!is_fullbright()) {
    //            lightSamples.push_back(face.get_average_lighting());
    //
    //            face.set_lightlump_offset(
    //                static_cast<int32_t>(lightSamples.size())
    //            );
    //
    //            for (LightSample& lightSample : face.get_lightsamples()) {
    //                lightSamples.push_back(lightSample);
    //            }
    //        }
    //
    //        face.set_texdata_index(static_cast<int32_t>(dTexDatas.size()));
    //        dTexDatas.push_back(face.get_texdata());
    //
    //        face.set_texinfo_index(static_cast<int32_t>(texInfos.size()));
    //        texInfos.push_back(face.get_texinfo());
    //
    //        dFaces.push_back(face.get_data());
    //    }
    //
    //    if (!is_fullbright()) {
    //        save_lump(
    //            file, LUMP_LIGHTING_HDR, m_lightSamples,
    //            offsets, sizes
    //        );
    //    }
    //    else {
    //        offsets[LUMP_LIGHTING_HDR] = 0;
    //        sizes[LUMP_LIGHTING_HDR] = 0;
    //    }
    //
    //    save_lump(
    //        file, LUMP_TEXINFO, m_texInfos,
    //        offsets, sizes
    //    );
    //
    //    save_lump(
    //        file, LUMP_TEXDATA, m_texDatas,
    //        offsets, sizes
    //    );
    //
    //    save_lump(
    //        file, LUMP_FACES, m_dFaces,
    //        offsets, sizes
    //    );
    //
    //    save_lump(
    //        file, LUMP_FACES_HDR, m_dFaces,
    //        offsets, sizes
    //    );
    //}

    void BSP::save_lights(std::ofstream& file) {

        //if (m_worldLights.size() == 0) {
       //     build_worldlights();
        //}

        if (is_hdr()) {
            save_lump(file, LUMP_WORLDLIGHTS_HDR, m_worldLights);
        }
        else {
            save_lump(file, LUMP_WORLDLIGHTS, m_worldLights);
        }
    }

    void BSP::save_visibility(std::ofstream& file) {

        // Since we just keep the original visibility lump data around,
        // this is really easy.
        save_lump(file, LUMP_VISIBILITY, m_visLumpData);
    }

    void BSP::save_extras(std::ofstream& file) {

        using Pair = std::pair<int, std::vector<uint8_t>>;

        for (const Pair& pair : m_extraLumps) {
            LumpType lumpID = static_cast<LumpType>(pair.first);
            const std::vector<uint8_t>& lumpData = pair.second;

            assert(m_extraLumps.find(lumpID) != m_extraLumps.end());

            save_lump(file, lumpID, lumpData, true);
        }
    }

    void BSP::save_gamelumps(std::ofstream& file) {

        Lump& lump = m_header.lumps[LUMP_GAME_LUMP];

        const int32_t lumpCount = static_cast<int32_t>(m_gameLumps.size());

        int size =
            sizeof(int32_t) + lumpCount * sizeof(GameLump);

        for (const auto& pair : m_gameLumps) {
            const int32_t id = pair.first;
            const auto& data = m_extraGameLumps.at(id);
            size += static_cast<int>(data.size());
        }

        lump.fileOffset = static_cast<int32_t>(file.tellp());
        lump.fileLen = size;

        file.write(
            reinterpret_cast<const char*>(&lumpCount),
            sizeof(int32_t)
        );

        int32_t dataOffset =
            lump.fileOffset
            + sizeof(int32_t)
            + lumpCount * sizeof(GameLump);

        for (const auto& pair : m_gameLumps) {
            const int32_t id = pair.first;
            GameLump dict = pair.second;

            const auto& data = m_extraGameLumps.at(id);

            dict.fileOffset = dataOffset;
            dict.fileLen = static_cast<int32_t>(data.size());

            file.write(
                reinterpret_cast<const char*>(&dict),
                sizeof(GameLump)
            );

            dataOffset += dict.fileLen;
        }

        for (const auto& pair : m_gameLumps) {
            const int32_t id = pair.first;
            const auto& data = m_extraGameLumps.at(id);

            if (!data.empty()) {
                file.write(
                    reinterpret_cast<const char*>(data.data()),
                    static_cast<std::streamsize>(data.size())
                );
            }
        }

        align_file_position(file, 4);

        m_extraLumps.erase(LUMP_GAME_LUMP);
    }

    void BSP::dump_lumps() const {
        for (int i = 0; i < HEADER_LUMPS; ++i) {
            const Lump& l = m_header.lumps[i];

            if (l.fileOffset == 0 && l.fileLen == 0) {
                continue;
            }

            printf(
                "%02d ofs=%d len=%d ver=%d fourCC=%02x%02x%02x%02x\n",
                i,
                l.fileOffset,
                l.fileLen,
                l.version,
                (unsigned char)l.fourCC[0],
                (unsigned char)l.fourCC[1],
                (unsigned char)l.fourCC[2],
                (unsigned char)l.fourCC[3]
            );
        }
    }

    template<typename Container>
    void BSP::save_single_gamelump(
            std::ofstream& file,
            int32_t gameLumpID,
            const Container& src
            ) {

        std::ofstream::off_type offset = file.tellp();

        size_t size = src.size();
        size *= sizeof(typename Container::value_type);

        GameLump& gameLump = m_gameLumps[gameLumpID];
        gameLump.fileOffset = static_cast<int32_t>(offset);
        gameLump.fileLen = static_cast<int32_t>(size);

        file.write(reinterpret_cast<const char*>(src.data()), size);
    }

    const std::unordered_map<int, std::vector<uint8_t>>&
    BSP::get_extras(void) const {
        return m_extraLumps;
    }

    void BSP::build_worldlights(void) {
        m_worldLights.clear();

        int i = 0;
        for (const Light& light : get_lights()) {
            //std::cout << "WorldLight " << i;
            m_worldLights.push_back(light.to_worldlight());
            i++;
        }
    }

    void BSP::init_ambient_samples(void) {
        /*m_ambientLightSamples.clear();
        m_ambientLightSamples.resize(m_leaves.size());

        for (size_t leafId = 0; leafId < m_leaves.size(); ++leafId) {
            CompressedLightCube& cube = m_ambientLightSamples[leafId];

            for (int side = 0; side < 6; ++side) {
                cube.color[side] = RGBExp32{ 0, 0, 0, 0 };
            }

            const DLeaf& leaf = m_leaves[leafId];

            if (leaf.contents & CONTENTS_SOLID)
                continue;
        }*/
    }


    /**********************
     * EntityParser Class *
     **********************/

    EntityParser::EntityParser(const std::string& entData) :
        m_index(0),
        m_entData(entData) {}

    Entity EntityParser::next_ent(void) {
        auto skipWhitespaceAndComments = [&]() {
            while (m_index < static_cast<int>(m_entData.size())) {
                char c = m_entData[m_index];

                // whitespace
                if (c <= 32) {
                    m_index++;
                    continue;
                }

                // ; comment or # comment
                if (c == ';' || c == '#') {
                    while (m_index < static_cast<int>(m_entData.size()) &&
                        m_entData[m_index] != '\n') {
                        m_index++;
                    }
                    continue;
                }

                // // comment
                if (c == '/' &&
                    m_index + 1 < static_cast<int>(m_entData.size()) &&
                    m_entData[m_index + 1] == '/') {
                    m_index += 2;
                    while (m_index < static_cast<int>(m_entData.size()) &&
                        m_entData[m_index] != '\n') {
                        m_index++;
                    }
                    continue;
                }

                // /* */ comment
                if (c == '/' &&
                    m_index + 1 < static_cast<int>(m_entData.size()) &&
                    m_entData[m_index + 1] == '*') {
                    m_index += 2;
                    while (m_index + 1 < static_cast<int>(m_entData.size())) {
                        if (m_entData[m_index] == '*' &&
                            m_entData[m_index + 1] == '/') {
                            m_index += 2;
                            break;
                        }
                        m_index++;
                    }
                    continue;
                }

                break;
            }
            };

        auto getToken = [&]() -> std::string {
            skipWhitespaceAndComments();

            if (m_index >= static_cast<int>(m_entData.size())) {
                return "";
            }

            char c = m_entData[m_index];

            // single-character entity delimiters
            if (c == '{' || c == '}') {
                m_index++;
                return std::string(1, c);
            }

            std::string out;

            // quoted token
            if (c == '"') {
                m_index++;

                while (m_index < static_cast<int>(m_entData.size())) {
                    c = m_entData[m_index++];

                    if (c == '"') {
                        break;
                    }

                    // Optional: support escaped quotes/backslashes.
                    if (c == '\\' && m_index < static_cast<int>(m_entData.size())) {
                        char next = m_entData[m_index];

                        if (next == '"' || next == '\\') {
                            out.push_back(next);
                            m_index++;
                            continue;
                        }
                    }

                    out.push_back(c);
                }

                return out;
            }

            // regular token
            while (m_index < static_cast<int>(m_entData.size())) {
                c = m_entData[m_index];

                if (c <= 32 || c == ';' || c == '#' || c == '{' || c == '}') {
                    break;
                }

                if (c == '/' &&
                    m_index + 1 < static_cast<int>(m_entData.size()) &&
                    (m_entData[m_index + 1] == '/' || m_entData[m_index + 1] == '*')) {
                    break;
                }

                out.push_back(c);
                m_index++;
            }

            return out;
            };

        Entity nextEnt;

        std::string tok = getToken();

        if (tok.empty()) {
            return nextEnt;
        }

        assert(tok == "{");

        while (true) {
            std::string key = getToken();

            if (key.empty()) {
                break;
            }

            if (key == "}") {
                break;
            }

            std::string value = getToken();

            if (value.empty()) {
                break;
            }

            if (value == "}") {
                // malformed entity, but don't crash release builds
                break;
            }

            nextEnt.set(key, value);
        }

        return nextEnt;
    }


    /**************
     * Face Class *
     **************/

    size_t Face::s_faceCount = 0;

    Face::Face(BSP& bsp, DFace& faceData) :
            m_bsp(bsp),
            m_faceData(faceData),
            m_planeData(bsp.m_planes[faceData.planeNum]),
            m_texInfo(bsp.m_texInfos[faceData.texInfo]),
            m_texData(bsp.m_texDatas[m_texInfo.texData]),
            id(s_faceCount++) {

        const std::vector<Vec3<float>>& vertices = bsp.get_vertices();
        const std::vector<DEdge>& dEdges = bsp.get_edges();
        const std::vector<int32_t>& surfEdges = bsp.get_surfedges();

        //std::cout << "Face " << id << " has lightmap size "
        //    << get_lightmap_width() << " x "
        //    << get_lightmap_height() << std::endl;

        load_edges(faceData, vertices, dEdges, surfEdges);
        //load_lightsamples(bsp.get_lightsamples());

        /* For coordinate transformation from s/t to x/y/z */
        //precalculate_st_xyz_matrix();
        make_st_xyz_matrix(m_Ainv);
    }

    void Face::load_edges(
            const DFace& faceData,
            const std::vector<Vec3<float>>& vertices,
            const std::vector<DEdge>& dEdges,
            const std::vector<int32_t>& surfEdges
            ) {

        int firstEdge = faceData.firstEdge;
        int lastEdge = faceData.firstEdge + faceData.numEdges;

        for (int i=firstEdge; i<lastEdge; i++) {
            int32_t surfEdge = surfEdges[i];

            bool firstToSecond = (surfEdge >= 0);

            if (!firstToSecond) {
                surfEdge *= -1;
            }

            const DEdge& dEdge = dEdges[surfEdge];

            Edge edge;

            if (firstToSecond) {
                edge.vertex1 = vertices[dEdge.vertex1];
                edge.vertex2 = vertices[dEdge.vertex2];
            }
            else {
                edge.vertex1 = vertices[dEdge.vertex2];
                edge.vertex2 = vertices[dEdge.vertex1];
            }

            m_edges.push_back(edge);
        }
    }

    //void Face::load_lightsamples(
    //        std::vector<LightSample>& lightSamples
    //        ) {
    //
    //    if (m_bsp.is_fullbright()) {
    //        return;
    //    }

    //    assert(m_bsp.get_lightsamples().size() > 0);
    //    assert(lightSamples.size() > 0);

    //    m_lightSamplesBegin = lightSamples.begin()
    //        + m_faceData.lightOffset / sizeof(LightSample);

    //    m_lightSamplesEnd = m_lightSamplesBegin
    //        + get_lightmap_width() * get_lightmap_height();
    //}

    void Face::make_st_xyz_matrix(gmtl::Matrix<double, 3, 3>& Ainv) const {
        double sx = m_texInfo.lightmapVecs[0][0];
        double sy = m_texInfo.lightmapVecs[0][1];
        double sz = m_texInfo.lightmapVecs[0][2];

        double tx = m_texInfo.lightmapVecs[1][0];
        double ty = m_texInfo.lightmapVecs[1][1];
        double tz = m_texInfo.lightmapVecs[1][2];

        Vec3<float> n = m_planeData.normal;
        if (m_faceData.side)
        {
            n.x *= -1.f;
            n.y *= -1.f;
            n.z *= -1.f;
        }
            
        double nx = n.x;
        double ny = n.y;
        double nz = n.z;

        gmtl::Matrix<double, 3, 3> A;

        A.set(
            sx, sy, sz,
            tx, ty, tz,
            nx, ny, nz
        );

        gmtl::invert(Ainv, A);
    }

    const DFace& Face::get_data(void) const {
        return m_faceData;
    }

    const DPlane& Face::get_planedata(void) const {
        return m_planeData;
    }

    const TexInfo& Face::get_texinfo(void) const {
        return m_texInfo;
    }

    const DTexData& Face::get_texdata(void) const {
        return m_texData;
    }

    const gmtl::Matrix<double, 3, 3>& Face::get_st_xyz_matrix(void) const {
        return m_Ainv;
    }

    void Face::set_texinfo_index(int16_t index) {
        m_faceData.texInfo = index;
    }

    void Face::set_texdata_index(int32_t index) {
        m_texInfo.texData = index;
    }

    const std::vector<Edge>& Face::get_edges(void) const {
        return m_edges;
    }

    const std::vector<uint8_t> Face::get_styles(void) const {
        std::vector<uint8_t> styles(4);

        for (int i=0; i<4; i++) {
            styles[i] = m_faceData.styles[i];
        }

        return styles;
    }

    void Face::set_styles(const std::vector<uint8_t>& styles) {
        assert(styles.size() == 4);

        for (int i=0; i<4; i++) {
            m_faceData.styles[i] = styles[i];
        }
    }

    int32_t Face::get_lightmap_width(void) const {
        return m_faceData.lightmapTextureSizeInLuxels[0] + 1;
    }

    int32_t Face::get_lightmap_height(void) const {
        return m_faceData.lightmapTextureSizeInLuxels[1] + 1;
    }

    size_t Face::get_lightmap_size(void) const {
        return get_lightmap_width() * get_lightmap_height();
    }

    size_t Face::get_lightmap_offset(void) const {
        return m_faceData.lightOffset / sizeof(RGBExp32);
    }

    void Face::set_lightmap_offset(size_t offset) {
        m_faceData.lightOffset = static_cast<int32_t>(
            offset * sizeof(RGBExp32)
        );
    }

    FaceLightSampleProxy Face::get_lightsamples(void) {
        std::vector<RGBExp32>& lightSamples = m_bsp.get_lightsamples();

        assert(lightSamples.size() > get_lightmap_offset());

        assert(
            lightSamples.size()
                >= get_lightmap_offset() + get_lightmap_size()
        );

        FaceLightSampleProxy::Iter begin
            = lightSamples.begin() + get_lightmap_offset();

        FaceLightSampleProxy::Iter end
            = begin + get_lightmap_size();

        return FaceLightSampleProxy(begin, end);
    }

    RGBExp32 Face::get_average_lighting(void) const {
        return m_bsp.get_lightsamples().at(get_lightmap_offset() - 1);
    }

    void Face::set_average_lighting(const RGBExp32& sample) {
        m_bsp.get_lightsamples()[get_lightmap_offset() - 1] = sample;
    }

    Vec3<float> Face::xyz_from_lightmap_st(float s, float t) const {
        double ds = s - m_texInfo.lightmapVecs[0][3];
        double dt = t - m_texInfo.lightmapVecs[1][3];

        double d = m_planeData.dist;
        if (m_faceData.side)
            d = -d;

        double x = m_Ainv[0][0] * ds + m_Ainv[0][1] * dt + m_Ainv[0][2] * d;
        double y = m_Ainv[1][0] * ds + m_Ainv[1][1] * dt + m_Ainv[1][2] * d;
        double z = m_Ainv[2][0] * ds + m_Ainv[2][1] * dt + m_Ainv[2][2] * d;

        return Vec3<float>{ float(x), float(y), float(z) };
    }


    /******************************
    * FaceLightSampleProxy Class *
    ******************************/

    FaceLightSampleProxy::FaceLightSampleProxy(
            FaceLightSampleProxy::Iter& begin,
            FaceLightSampleProxy::Iter& end
            ) :
            m_begin(begin),
            m_end(end) {}

    RGBExp32& FaceLightSampleProxy::operator[](size_t i) {
        return *(m_begin + i);
    }

    const RGBExp32& FaceLightSampleProxy::operator[](size_t i) const {
        return *(m_begin + i);
    }

    FaceLightSampleProxy::Iter FaceLightSampleProxy::begin(void) {
        return m_begin;
    }

    FaceLightSampleProxy::Iter FaceLightSampleProxy::end(void) {
        return m_end;
    }


    /***************
     * Light Class *
     ***************/

    template<typename T>
    static inline T convert_str(const std::string& str) {
        T result;

        std::stringstream converter;
        converter << str;
        converter >> result;

        return result;
    }

    static Vec3<float> vec3_from_str(const std::string& str) {
        std::stringstream stream(str);

        float x;
        float y;
        float z;

        std::string s;

        std::getline(stream, s, ' ');
        x = convert_str<float>(s);

        std::getline(stream, s, ' ');
        y = convert_str<float>(s);

        std::getline(stream, s, ' ');
        z = convert_str<float>(s);

        return Vec3<float> {x, y, z};
    }

    static const double PI = 3.14159265358979323846264;

    static inline double radians(double degrees) {
        return degrees / 180.0 * PI;
    }

    static inline double degrees(double radians) {
        return radians / PI * 180.0;
    }

    static Vec3<double> direction_from_angles(Vec3<double> angles) {
        double pitch = radians(angles.x);
        double yaw = radians(angles.y);

        //std::cout << "cos(pitch = " << pitch << ") = " << cos(pitch) << std::endl;
        //std::cout << "cos(yaw = " << yaw << ") = " << cos(yaw) << std::endl;
        //std::cout << "sin(yaw = " << yaw << ") = " << sin(yaw) << std::endl;
        //std::cout << "sin(pitch = " << pitch << ") = " << sin(pitch) << std::endl;

        return Vec3<double> {
            cos(pitch) * cos(yaw),
            cos(pitch) * sin(yaw),
            sin(pitch),
        };
    }

    /* Gamma-correction */
    static inline double linear_from_encoded(double encoded) {
        return pow(encoded / 255.0, GAMMA) * 255.0;
    }

    static inline double encoded_from_linear(double linear) {
        return pow(linear / 255.0, INV_GAMMA) * 255.0;
    }

    Light::Light(const BSP& bsp, const Entity& entity) :
        m_coords(vec3_from_str(entity.get("origin", "0 0 0"))),
        m_cluster(bsp.cluster_for_pos(m_coords)),
        direction(Vec3<double>{1.0, 0.0, 0.0}),
        emitType(EMIT_POINT),
        r(0.0), g(0.0), b(0.0),
        c(0.0), l(0.0), q(0.0),
        innerCone(30.0), outerCone(45.0)
    {
        const std::string classname = entity.get("classname");

        if (classname == "light") {
            parse_point(bsp, entity);
        }
        else if (classname == "light_spot") {
            parse_spot(bsp, entity);
        }
        else if (classname == "light_environment") {
            parse_environment(bsp, entity);
        }
    }

    bool Light::parse_light_value_raw_optional(const std::string& value) {
        if (value.empty())
            return false;

        std::stringstream stream(value);
        std::string s;

        double rr, gg, bb;

        if (!std::getline(stream, s, ' ')) return false;
        rr = convert_str<double>(s);

        if (!std::getline(stream, s, ' ')) return false;
        gg = convert_str<double>(s);

        if (!std::getline(stream, s, ' ')) return false;
        bb = convert_str<double>(s);

        if (rr == -1.0 && gg == -1.0 && bb == -1.0)
            return false;

        double brightness = 1.0;
        if (std::getline(stream, s, ' '))
            brightness = convert_str<double>(s) / 255.0;

        r = rr * brightness;
        g = gg * brightness;
        b = bb * brightness;

        return true;
    }

    void Light::parse_generic(const BSP& bsp, const Entity& entity) {
        style = convert_str<int>(entity.get("style", "0"));

        bool parsed = false;

        if (bsp.is_hdr() && entity.has_key("_lightHDR")) {
            parsed = parse_light_value_raw_optional(entity.get("_lightHDR"));

            if (r == -1.0 && g == -1.0 && b == -1.0) {
                parsed = false;
            }
        }

        if (!parsed) {
            parse_light_value_raw(entity.get("_light", "255 255 255 200"));
        }

        parse_direction(entity);
    }

    void Light::parse_falloff(const Entity& entity) {
        c = convert_str<double>(entity.get("_constant_attn", "0"));
        l = convert_str<double>(entity.get("_linear_attn", "0"));
        q = convert_str<double>(entity.get("_quadratic_attn", "0"));

        radius = convert_str<double>(entity.get("_distance", "0"));

        constexpr double EPS = 1e-6;

        if (c < EPS) c = 0.0;
        if (l < EPS) l = 0.0;
        if (q < EPS) q = 0.0;

        if (c < EPS && l < EPS && q < EPS)
            c = 1.0;

        double ratio = c + 100.0 * l + 100.0 * 100.0 * q;

        if (ratio > 0.0) {
            r *= ratio;
            g *= ratio;
            b *= ratio;
        }
    }

    void Light::parse_point(const BSP& bsp, const Entity& entity) {
        emitType = EMIT_POINT;

        parse_generic(bsp, entity);
        parse_falloff(entity);
    }

    void Light::parse_spot(const BSP& bsp, const Entity& entity) {
        parse_generic(bsp, entity);

        emitType = EMIT_SPOTLIGHT;

        innerCone = convert_str<double>(entity.get("_inner_cone", "0"));
        if (innerCone == 0.0)
            innerCone = 10.0;

        outerCone = convert_str<double>(entity.get("_cone", "0"));
        if (outerCone == 0.0)
            outerCone = innerCone;

        if (outerCone < innerCone)
            outerCone = innerCone;

        if (innerCone == 180.0 && outerCone == 180.0) {
            innerCone = 0.0;
            outerCone = 0.0;
            emitType = EMIT_POINT;
            exponent = 0.0;
        }
        else {
            if (innerCone > 90.0)
                innerCone = 90.0;

            if (outerCone > 90.0)
                outerCone = 90.0;

            stopdot = std::cos(radians(innerCone));
            stopdot2 = std::cos(radians(outerCone));

            exponent = convert_str<double>(entity.get("_exponent", "0"));
        }

        parse_falloff(entity);
    }

    void Light::parse_environment(const BSP& bsp, const Entity& entity) {
        parse_generic(bsp, entity);
        
        emitType = EMIT_SKYLIGHT;
    }

    void Light::parse_direction(const Entity& entity) {
        Vec3<float> angles = vec3_from_str(entity.get("angles", "0 0 0"));

        std::string pitchOverride = entity.get("pitch", "");
        if (!pitchOverride.empty()) {
            angles.x = convert_str<float>(pitchOverride);
        }

        direction = direction_from_angles(
            Vec3<double>{angles.x, angles.y, angles.z}
        );
    }

    void Light::parse_light_value_raw(const std::string& value) {
        std::stringstream stream(value);
        std::string s;

        std::getline(stream, s, ' ');
        r = convert_str<double>(s);

        std::getline(stream, s, ' ');
        g = convert_str<double>(s);

        std::getline(stream, s, ' ');
        b = convert_str<double>(s);

        std::getline(stream, s, ' ');
        double brightness = convert_str<double>(s);

        r *= brightness / 255.0;
        g *= brightness / 255.0;
        b *= brightness / 255.0;
    }

    const Vec3<float>& Light::get_coords(void) const {
        return m_coords;
    }

    void Light::parse_color(const std::string& value) {
        std::stringstream stream(value);
        std::string s;

        std::getline(stream, s, ' ');
        r = linear_from_encoded(convert_str<double>(s));

        std::getline(stream, s, ' ');
        g = linear_from_encoded(convert_str<double>(s));

        std::getline(stream, s, ' ');
        b = linear_from_encoded(convert_str<double>(s));

        std::getline(stream, s, ' ');
        double brightness = convert_str<double>(s) / 255.0;

        r *= brightness;
        g *= brightness;
        b *= brightness;
    }

    DWorldLight Light::to_worldlight(void) const {
        return DWorldLight{
            get_coords(),

            Vec3<float> {
                static_cast<float>(r),
                static_cast<float>(g),
                static_cast<float>(b),
            },

            Vec3<float> {
                static_cast<float>(direction.x),
                static_cast<float>(direction.y),
                static_cast<float>(direction.z),
            },

            m_cluster,
            emitType,
            style,

            stopdot,
            stopdot2,
            exponent,
            radius,

            static_cast<float>(c),
            static_cast<float>(l),
            static_cast<float>(q),

            0x0, // flags
            0,   // texinfo
            0,   // owner
        };
    }


    /****************
     * Entity Class *
     ****************/

    Entity::Entity() {}

    const std::string& Entity::get(const std::string& key) const {
        return m_data.at(key);
    }

    const std::string& Entity::get(
            const std::string& key,
            const std::string& defaultVal
            ) const {

        std::unordered_map<std::string, std::string>::const_iterator
            pValue = m_data.find(key);

        if (pValue != m_data.end()) {
            return pValue->second;
        }
        else {
            return defaultVal;
        }
    }

    bool Entity::has_key(const std::string& key) const {
        return m_data.find(key) != m_data.end();
    }

    void Entity::set(const std::string& key, const std::string& value) {
        m_data[key] = value;
    }

    size_t Entity::size(void) const {
        return m_data.size();
    }
}
