#include "source_map.h"

#include <stdexcept>

namespace SilkRAD::V2::BSP {
    SourceMap::SourceMap(const ::BSP::BSP& bsp)
        : m_bsp(&bsp)
    {
    }

    const ::BSP::BSP& SourceMap::raw_bsp() const
    {
        return *m_bsp;
    }

    size_t SourceMap::face_count() const
    {
        return m_bsp->get_faces().size();
    }

    size_t SourceMap::dface_count() const
    {
        return m_bsp->get_dfaces().size();
    }

    size_t SourceMap::plane_count() const
    {
        return m_bsp->get_planes().size();
    }

    size_t SourceMap::texinfo_count() const
    {
        return m_bsp->get_texinfos().size();
    }

    size_t SourceMap::dispinfo_count() const
    {
        return m_bsp->get_dispinfos().size();
    }

    size_t SourceMap::brush_count() const
    {
        return m_bsp->get_brushes().size();
    }

    size_t SourceMap::model_count() const
    {
        return m_bsp->get_models().size();
    }

    const ::BSP::Face& SourceMap::face(size_t faceIndex) const
    {
        return m_bsp->get_faces().at(faceIndex);
    }

    const ::BSP::DFace& SourceMap::dface(size_t faceIndex) const
    {
        return m_bsp->get_dfaces().at(faceIndex);
    }

    const ::BSP::DPlane& SourceMap::plane(size_t planeIndex) const
    {
        return m_bsp->get_planes().at(planeIndex);
    }

    const ::BSP::TexInfo& SourceMap::texinfo(size_t texinfoIndex) const
    {
        return m_bsp->get_texinfos().at(texinfoIndex);
    }

    const ::BSP::DModel& SourceMap::model(size_t modelIndex) const
    {
        return m_bsp->get_models().at(modelIndex);
    }

    const ::BSP::DBrush& SourceMap::brush(size_t brushIndex) const
    {
        return m_bsp->get_brushes().at(brushIndex);
    }

    const ::BSP::DBrushSide& SourceMap::brushside(size_t brushSideIndex) const
    {
        return m_bsp->get_brushsides().at(brushSideIndex);
    }

    const std::vector<::BSP::DispVert>& SourceMap::dispverts() const
    {
        return m_bsp->get_dispverts();
    }

    const std::vector<::BSP::DispTri>& SourceMap::disptris() const
    {
        return m_bsp->get_disptris();
    }

    const ::BSP::StaticPropData& SourceMap::static_props() const
    {
        return m_bsp->get_static_props();
    }

    const ::BSP::DispInfo* SourceMap::dispinfo_for_face(size_t faceIndex) const
    {
        const ::BSP::DFace& faceData = dface(faceIndex);
        if (faceData.dispInfo < 0) {
            return nullptr;
        }

        const size_t dispInfoIndex = static_cast<size_t>(faceData.dispInfo);
        if (dispInfoIndex >= m_bsp->get_dispinfos().size()) {
            throw std::out_of_range("Displacement info index is out of range.");
        }

        return &m_bsp->get_dispinfos()[dispInfoIndex];
    }

    const std::vector<uint8_t>& SourceMap::disp_lightmap_sample_positions() const
    {
        return m_bsp->get_disp_lightmap_sample_positions();
    }

    std::vector<size_t> SourceMap::world_face_indices() const
    {
        std::vector<size_t> faceIndices;

        if (m_bsp->get_models().empty()) {
            return faceIndices;
        }

        const ::BSP::DModel& worldModel = m_bsp->get_models()[0];
        if (worldModel.firstFace < 0 || worldModel.numFaces <= 0) {
            return faceIndices;
        }

        const size_t firstFace = static_cast<size_t>(worldModel.firstFace);
        const size_t numFaces = static_cast<size_t>(worldModel.numFaces);
        const size_t endFace = std::min(firstFace + numFaces, m_bsp->get_faces().size());

        faceIndices.reserve(endFace - firstFace);
        for (size_t faceIndex = firstFace; faceIndex < endFace; ++faceIndex) {
            faceIndices.push_back(faceIndex);
        }

        return faceIndices;
    }

    std::vector<uint8_t> SourceMap::world_brush_mask() const
    {
        std::vector<uint8_t> usedBrushes(m_bsp->get_brushes().size(), 0);
        if (m_bsp->get_models().empty()) {
            return usedBrushes;
        }

        mark_model_brushes_r(m_bsp->get_models()[0].headNode, usedBrushes);
        return usedBrushes;
    }

    void SourceMap::mark_model_brushes_r(
        int32_t nodeIndex,
        std::vector<uint8_t>& usedBrushes
    ) const
    {
        if (nodeIndex < 0) {
            const int32_t leafIndex = -nodeIndex - 1;
            const std::vector<::BSP::DLeaf>& leaves = m_bsp->get_leaves();
            const std::vector<uint16_t>& leafBrushes = m_bsp->get_leafbrushes();

            if (leafIndex < 0
                || static_cast<size_t>(leafIndex) >= leaves.size()) {
                return;
            }

            const ::BSP::DLeaf& leaf = leaves[leafIndex];
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

        const std::vector<::BSP::DNode>& nodes = m_bsp->get_nodes();
        if (static_cast<size_t>(nodeIndex) >= nodes.size()) {
            return;
        }

        const ::BSP::DNode& node = nodes[nodeIndex];
        mark_model_brushes_r(node.children[0], usedBrushes);
        mark_model_brushes_r(node.children[1], usedBrushes);
    }
}
