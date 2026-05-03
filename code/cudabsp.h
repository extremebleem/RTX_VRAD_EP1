#ifndef __CUDABSP_H_
#define __CUDABSP_H_

#include <cstdint>

#include "cuda_runtime.h"
#include "gmtl/Matrix.h"

#include "bsp.h"

#include "cudamatrix.h"


namespace CUDABSP {
    struct CUDABSP;

    // Technically it should really only be 32x32, but occasionally you'll see
    // a face with 33 luxels in at least one dimension, and that really screws
    // things up since 33x33 isn't a multiple of 256...
    // So we allocate an additional 256 to account for possible overflow.
    __device__ const size_t MAX_LUXELS_PER_FACE = 4900;

    __device__ BSP::RGBExp32 rgbexp32_from_float3(float3 color);

    __device__ int16_t cluster_for_pos(const CUDABSP& cudaBSP, float3 pos);
    __device__ uint8_t* pvs_for_pos(const CUDABSP& cudaBSP, float3 pos);
    __device__ bool cluster_in_pvs(
        int16_t cluster, uint8_t* pvs, size_t numClusters
    );

    const uint32_t TAG = 0xdeadbeef;

    struct CUDABSP {
        uint32_t tag;

        BSP::DModel* models;
        BSP::DPlane* planes;
        float3* vertices;
        BSP::DEdge* edges;
        int32_t* surfEdges;
        BSP::DFace* faces;
        CUDAMatrix::CUDAMatrix<double, 3, 3>* xyzMatrices;
        float3* lightSamples;
        BSP::RGBExp32* rgbExp32LightSamples;
        BSP::TexInfo* texInfos;
        BSP::DTexData* texDatas;
        BSP::DNode* nodes;
        BSP::DLeaf* leaves;
        BSP::CompressedLightCube* ambientLightSamples;
        BSP::DWorldLight* worldLights;
        uint8_t* visMatrix;

        size_t numModels;
        size_t numPlanes;
        size_t numVertices;
        size_t numEdges;
        size_t numSurfEdges;
        size_t numFaces;
        size_t numLightSamples;
        size_t numTexInfos;
        size_t numTexDatas;
        size_t numNodes;
        size_t numLeaves;
        size_t numAmbientLightSamples;
        size_t numWorldLights;
        size_t numVisClusters;
    };

    /** Creates a new CUDABSP on the device, and returns a pointer to it. */
    CUDABSP* make_cudabsp(BSP::BSP& bsp);

    /** Destroys the given CUDABSP located on the device. */
    void destroy_cudabsp(CUDABSP* pCudaBSP);

    /** Clears imported BSP lighting so output cannot reuse old lightmaps. */
    void clear_lighting(CUDABSP* pCudaBSP);

    /** Convert lightsamples from float3 to RGBExp32 format. */
    void convert_lightsamples(CUDABSP* pCudaBSP);

    /**
     * Updates the given BSP using the information contained in the given
     * CUDABSP (which should be on the device).
     */
    void update_bsp(BSP::BSP& bsp, CUDABSP* pCudaBSP);
}


#endif
