#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../bsp.h"

namespace SilkRAD::Core::BSP {
    class SourceMap {
    public:
        explicit SourceMap(const ::BSP::BSP& bsp);

        const ::BSP::BSP& raw_bsp() const;

        size_t face_count() const;
        size_t dface_count() const;
        size_t plane_count() const;
        size_t texinfo_count() const;
        size_t dispinfo_count() const;
        size_t brush_count() const;
        size_t model_count() const;

        const ::BSP::Face& face(size_t faceIndex) const;
        const ::BSP::DFace& dface(size_t faceIndex) const;
        const ::BSP::DPlane& plane(size_t planeIndex) const;
        const ::BSP::TexInfo& texinfo(size_t texinfoIndex) const;
        const ::BSP::DModel& model(size_t modelIndex) const;
        const ::BSP::DBrush& brush(size_t brushIndex) const;
        const ::BSP::DBrushSide& brushside(size_t brushSideIndex) const;
        const std::vector<::BSP::DispVert>& dispverts() const;
        const std::vector<::BSP::DispTri>& disptris() const;
        const ::BSP::StaticPropData& static_props() const;
        const ::BSP::DispInfo* dispinfo_for_face(size_t faceIndex) const;
        const std::vector<uint8_t>& disp_lightmap_sample_positions() const;
        std::vector<size_t> world_face_indices() const;
        std::vector<uint8_t> world_brush_mask() const;

    private:
        void mark_model_brushes_r(
            int32_t nodeIndex,
            std::vector<uint8_t>& usedBrushes
        ) const;

        const ::BSP::BSP* m_bsp;
    };
}
