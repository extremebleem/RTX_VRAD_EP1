#ifndef __CUDARAD_H_
#define __CUDARAD_H_

#include <string>
#include <vector>

#include "bsp.h"
#include "cudabsp.h"

static inline bool vec_equal(
    const BSP::Vec3<float>& a,
    const BSP::Vec3<float>& b
) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

namespace CUDARAD {
    struct FaceInfo {
        BSP::DFace face;
        BSP::DPlane plane;
        BSP::TexInfo texInfo;
        CUDAMatrix::CUDAMatrix<double, 3, 3> Ainv;
        float3 faceNorm;
        float3 totalLight;
        float3 avgLight;
        size_t faceIndex;
        size_t lightmapWidth;
        size_t lightmapHeight;
        size_t lightmapSize;
        size_t lightmapStartIndex;
        size_t patchStartIndex;

        __device__ FaceInfo();
        __device__ FaceInfo(CUDABSP::CUDABSP& cudaBSP, size_t faceIndex);

        __device__ float3 xyz_from_st(float s, float t);
    };
}

namespace DirectLighting {
    __device__ float3 sample_at(
        CUDABSP::CUDABSP& cudaBSP, CUDARAD::FaceInfo& faceInfo,
        float s, float t
    );
}

namespace CUDARAD {
    void set_asset_root(const std::string& assetRoot);
    void init(BSP::BSP& bsp);
    void cleanup(void);

    void compute_direct_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP);
    void compute_leaf_ambient(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP);
}

#endif
