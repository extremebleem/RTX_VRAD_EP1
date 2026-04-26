#include "cub/cub.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>

#include "cudarad.h"

#include "bsp.h"
#include "bsp_shared.h"

#include "cudabsp.h"
#include "cudamatrix.h"
#include "raytracer.h"

#include "cudautils.h"


namespace CUDARAD {
    static std::unique_ptr<RayTracer::CUDARayTracer> g_pRayTracer;
    static __device__ RayTracer::CUDARayTracer* g_pDeviceRayTracer;

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

            const float EPSILON = 1e-3f;

            // Nudge the sample position towards the light slightly, to avoid
            // colliding with triangles that directly contain the sample
            // position.
            samplePos -= dir * EPSILON;

            bool lightBlocked = CUDARAD::g_pDeviceRayTracer->LOS_blocked(
                lightPos, samplePos
            );

            if (lightBlocked) {
                // This light can't be seen from the position of the sample.
                // Ignore it.
                continue;
            }

            /* I CAN SEE THE LIGHT */
            float attenuation = attenuate(light, dist);

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

    __device__ float3 sample_at(
            CUDABSP::CUDABSP& cudaBSP,
            CUDARAD::FaceInfo& faceInfo,
            float s, float t
            ) {

        float3 samplePos = faceInfo.xyz_from_st(s, t);
        return sample_at(cudaBSP, samplePos, faceInfo.faceNorm);
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
            // Map block numbers to faces.
            faceInfo = CUDARAD::FaceInfo(*pCudaBSP, blockIdx.x);

            //printf(
            //    "Processing Face %u...\n",
            //    static_cast<unsigned int>(faceInfo.faceIndex)
            //);
        }

        __syncthreads();

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

            pCudaBSP->lightSamples[faceInfo.lightmapStartIndex - 1]
                = faceInfo.avgLight;

            // Still have no idea how this works. But if we don't do this,
            // EVERYTHING becomes a disaster...
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


namespace CUDARAD {
    void init(BSP::BSP& bsp) {
        std::cout << "Setting up ray-trace acceleration structure... "
            << std::flush;

        using Clock = std::chrono::high_resolution_clock;

        auto start = Clock::now();

        g_pRayTracer = std::unique_ptr<RayTracer::CUDARayTracer>(
            new RayTracer::CUDARayTracer()
        );

        std::vector<RayTracer::Triangle> triangles;

        /* Put all of the BSP's face triangles into the ray-tracer. */
        for (const BSP::Face& face : bsp.get_faces()) {
            int32_t flags = face.get_texinfo().flags;

            if ((flags & BSP::SURF_TRANS) && !(flags & BSP::SURF_NODRAW)) {
                // Skip translucent faces, but keep nodraw faces.
                continue;
            }

            std::vector<BSP::Edge>::const_iterator pEdge
                = face.get_edges().begin();

            BSP::Vec3<float> vertex1 = (pEdge++)->vertex1;
            BSP::Vec3<float> vertex2;
            BSP::Vec3<float> vertex3 = (pEdge++)->vertex1;

            do {
                vertex2 = vertex3;
                vertex3 = (pEdge++)->vertex1;

                RayTracer::Triangle tri {
                    {
                        make_float3(vertex1),
                        make_float3(vertex2),
                        make_float3(vertex3),
                    },
                    face.id
                };

                triangles.push_back(tri);

            } while (pEdge != face.get_edges().end());
        }

        g_pRayTracer->add_triangles(triangles);

        bsp.init_ambient_samples();

        auto end = Clock::now();
        std::chrono::milliseconds ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start
            );

        std::cout << "Done! (" << ms.count() << " ms)" << std::endl;

        std::cout << "Moving ray-tracer to device..." << std::endl;

        RayTracer::CUDARayTracer* pDeviceRayTracer;

        CUDA_CHECK_ERROR(
            cudaMalloc(&pDeviceRayTracer, sizeof(RayTracer::CUDARayTracer))
        );
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                pDeviceRayTracer, g_pRayTracer.get(),
                sizeof(RayTracer::CUDARayTracer),
                cudaMemcpyHostToDevice
            )
        );
        CUDA_CHECK_ERROR(
            cudaMemcpyToSymbol(
                g_pDeviceRayTracer, &pDeviceRayTracer,
                sizeof(RayTracer::CUDARayTracer*), 0,
                cudaMemcpyHostToDevice
            )
        );
    }

    void cleanup(void) {
        RayTracer::CUDARayTracer* pDeviceRayTracer;

        CUDA_CHECK_ERROR(
            cudaMemcpyFromSymbol(
                &pDeviceRayTracer, g_pDeviceRayTracer,
                sizeof(RayTracer::CUDARayTracer*), 0,
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK_ERROR(cudaFree(pDeviceRayTracer));

        g_pRayTracer = nullptr;
    }

    void compute_direct_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
        volatile size_t* pFacesCompleted;
        CUDA_CHECK_ERROR(
            cudaHostAlloc(
                &pFacesCompleted, sizeof(size_t),
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

        const size_t BLOCK_WIDTH = 16;
        const size_t BLOCK_HEIGHT = 16;

        size_t numFaces = bsp.get_faces().size();

        dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);

        std::cout << "Launching "
            << numFaces * BLOCK_WIDTH * BLOCK_HEIGHT << " threads ("
            << numFaces << " faces)..."
            << std::endl;

        cudaEvent_t startEvent;
        cudaEvent_t stopEvent;

        CUDA_CHECK_ERROR(cudaEventCreate(&startEvent));
        CUDA_CHECK_ERROR(cudaEventCreate(&stopEvent));

        CUDA_CHECK_ERROR(cudaEventRecord(startEvent));

        KERNEL_LAUNCH(
            DirectLighting::map_faces,
            numFaces, blockDim,
            pCudaBSP, const_cast<size_t*>(pDeviceFacesCompleted)
        );

        flush_wddm_queue();

        size_t lastFacesCompleted = 0;
        size_t facesCompleted;

        /* Progress notification logic */
        do {
            CUDA_CHECK_ERROR(cudaPeekAtLastError());

            facesCompleted = *pFacesCompleted;

            if (facesCompleted > lastFacesCompleted) {
                std::cout << "    " << facesCompleted << "/"
                    << numFaces
                    << " faces processed..." << std::endl;
            }

            lastFacesCompleted = facesCompleted;

            std::this_thread::sleep_for(std::chrono::milliseconds(5));

        } while (facesCompleted < numFaces);

        CUDA_CHECK_ERROR(cudaEventRecord(stopEvent));
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        float time;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&time, startEvent, stopEvent));

        std::cout << "Done! (" << time << " ms)" << std::endl;

        cudaFreeHost(const_cast<size_t*>(pFacesCompleted));
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

    void bounce_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
        using Clock = std::chrono::high_resolution_clock;

        auto start = Clock::now();

        auto end = Clock::now();
        std::chrono::milliseconds ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start
            );

        std::cout << "Done! (" << ms.count() << " ms)" << std::endl;
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
}
