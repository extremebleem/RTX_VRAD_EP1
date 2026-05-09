#include "state.h"

namespace SilkRAD::Core {
    RuntimeState build_runtime_state(
        const BSP::SourceMap& sourceMap,
        const BuildOptions& options
    )
    {
        RuntimeState state;
        state.faceGeometry.resize(sourceMap.face_count());
        state.dispGeometry.resize(sourceMap.face_count());

        for (size_t faceIndex = 0; faceIndex < sourceMap.face_count(); ++faceIndex) {
            state.faceGeometry[faceIndex] =
                Geometry::build_face_geometry(sourceMap, faceIndex);

            if (sourceMap.dispinfo_for_face(faceIndex) != nullptr) {
                state.dispGeometry[faceIndex] =
                    Geometry::build_disp_geometry(sourceMap, faceIndex);
            }
        }

        state.scene = Scene::build_scene(
            sourceMap,
            state.faceGeometry,
            state.dispGeometry,
            Scene::SceneBuildOptions{ options.assetSearchRoots }
        );
        state.directLightingInputs = Lighting::build_direct_lighting_inputs(
            state.faceGeometry,
            state.dispGeometry,
            state.scene
        );
        return state;
    }
}
