#include "fxaa.h"

#include "bsp.h"
#include "cudabsp.h"
#include "cudarad.h"

#include "cudautils.h"


static __device__ inline float luma_from_rgb(float3 rgb) {
    //return sqrt(dot(rgb / 255.0, make_float3(0.299, 0.587, 0.114)));
    return sqrt(dot(rgb / 255.0f, make_float3(1.0f)));
}


static __device__ inline float clamp(float x, float lower, float upper) {
    return fmaxf(lower, fminf(upper, x));
}


static __device__ float3 sample(
        CUDABSP::CUDABSP& cudaBSP, CUDARAD::FaceInfo& faceInfo,
        float3* samplesIn, size_t width, size_t height,
        float s, float t
        ) {

    if (0.0f <= s && s < width) {
        if (0.0f <= t && t < height) {
            return samplesIn[static_cast<size_t>(t * width + s)];
        }
    }

    // If we're not in bounds, we have no choice but to take another light
    // sample.
    return DirectLighting::sample_at(cudaBSP, faceInfo, s, t);
}


static __device__ float3 subsample(
        CUDABSP::CUDABSP& cudaBSP, CUDARAD::FaceInfo& faceInfo,
        float3* samplesIn, size_t width, size_t height,
        float s, float t
        ) {

    const float EPSILON = 1e-3;

    s = fminf(fmaxf(0.0f, s), width - EPSILON);
    t = fminf(fmaxf(0.0f, t), height - EPSILON);

    int s0 = static_cast<int>(floorf(s));
    int t0 = static_cast<int>(floorf(t));
    int s1 = s0 + 1;
    int t1 = t0 + 1;

    float rWeight = s - floorf(s);
    float lWeight = 1.0f - rWeight;

    float dWeight = t - floorf(t);
    float uWeight = 1.0f - dWeight;

    auto get_sample = [&] (float s, float t) -> float3 {
        return sample(cudaBSP, faceInfo, samplesIn, width, height, s, t);
    };

    float3 sampleUL = get_sample(s0, t0);
    float3 sampleUR = get_sample(s1, t0);
    float3 sampleDL = get_sample(s0, t1);
    float3 sampleDR = get_sample(s1, t1);

    float3 sampleU = lWeight * sampleUL + rWeight * sampleUR;
    float3 sampleD = lWeight * sampleDL + rWeight * sampleDR;

    return uWeight * sampleU + dWeight * sampleD;
}


static __device__ const float EDGE_THRESHOLD = 0.125;           // 1/8
static __device__ const float EDGE_THRESHOLD_MIN = 0.03125;     // 1/32
static __device__ const size_t MAX_ITERATIONS = 12;
static __device__ const float SUBPIXEL_QUALITY = 0.75;


/**
 * CUDA FXAA implementation based on shader code at:
 *  http://blog.simonrodriguez.fr/articles/30-07-2016_implementing_fxaa.html
 * and also:
 *  http://developer.download.nvidia.com/assets/gamedev/files/sdk/11/FXAA_WhitePaper.pdf
 */
__global__ void map_samples_fxaa(
        CUDABSP::CUDABSP* pCudaBSP, CUDARAD::FaceInfo* pFaceInfo,
        float3* samplesIn, /* output */ float3* samplesOut,
        size_t width, size_t height
        ) {

    float s = static_cast<float>(blockIdx.x * blockDim.x + threadIdx.x);
    float t = static_cast<float>(blockIdx.y * blockDim.y + threadIdx.y);

    if (s >= width || t >= height) {
        return;
    }

    auto get_sample = [&] (float s, float t) -> float3 {
        return sample(*pCudaBSP, *pFaceInfo, samplesIn, width, height, s, t);
    };

    auto get_subsample = [&] (float s, float t) -> float3 {
        return subsample(*pCudaBSP, *pFaceInfo, samplesIn, width, height, s, t);
    };

    float3 rgbSample = get_sample(s, t);

    float lumaCenter = luma_from_rgb(rgbSample);

    /* Grab the lumas of our four direct neighbors. */
    float lumaUp = luma_from_rgb(get_sample(s, t - 1));
    float lumaDown = luma_from_rgb(get_sample(s, t + 1));
    float lumaLeft = luma_from_rgb(get_sample(s - 1, t));
    float lumaRight = luma_from_rgb(get_sample(s + 1, t));

    /* Determine the color contrast between ourselves and our neighbors. */
    float lumaMin = fminf(
        lumaCenter,
        fminf(
            fminf(lumaUp, lumaDown),
            fminf(lumaLeft, lumaRight)
        )
    );

    float lumaMax = fmaxf(
        lumaCenter,
        fmaxf(
            fmaxf(lumaUp, lumaDown),
            fmaxf(lumaLeft, lumaRight)
        )
    );

    float lumaRange = lumaMax - lumaMin;

    /*
     * Luma contrast too low (or this is a really dark spot).
     * Don't perform AA.
     */
    if (lumaRange < fmaxf(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD)) {
        samplesOut[static_cast<size_t>(t * width + s)] = rgbSample;
        return;
    }
    //else {
    //    samplesOut[t * width + s] = make_float3(255.0, 0.0, 0.0);
    //    return;
    //}

    /* Grab the lumas of our remaining corner neighbors. */
    float lumaUL = luma_from_rgb(get_sample(s - 1, t - 1));
    float lumaUR = luma_from_rgb(get_sample(s + 1, t - 1));
    float lumaDL = luma_from_rgb(get_sample(s - 1, t + 1));
    float lumaDR = luma_from_rgb(get_sample(s + 1, t + 1));

    /* Combine the edge lumas. */
    float lumaUD = lumaUp + lumaDown;
    float lumaLR = lumaLeft + lumaRight;

    /* Combine the corner lumas. */
    float lumaULUR = lumaUL + lumaUR;
    float lumaDLDR = lumaDL + lumaDR;
    float lumaULDL = lumaUL + lumaDL;
    float lumaURDR = lumaUR + lumaDR;

    /* Estimate horizontal and vertical gradients. */
    float gradientHoriz = (
        fabsf(-2.0f * lumaLeft + lumaULDL)
        + fabsf(-2.0f * lumaCenter + lumaUD) * 2.0f
        + fabsf(-2.0f * lumaRight + lumaURDR)
    );

    float gradientVerti = (
        fabsf(-2.0f * lumaUp + lumaULUR)
        + fabsf(-2.0f * lumaCenter + lumaLR) * 2.0f
        + fabsf(-2.0f * lumaDown + lumaDLDR)
    );

    /* Are we at a horizontal or vertical edge? */
    bool isHoriz = (gradientHoriz >= gradientVerti);

    //if (isHoriz) {
    //    samplesOut[t * width + s] = make_float3(255.0, 0.0, 0.0);
    //}
    //else {
    //    samplesOut[t * width + s] = make_float3(0.0, 255.0, 0.0);
    //}
    //return;

    /* Choose two lumas in the direction opposite of the edge. */
    float luma1 = isHoriz ? lumaUp : lumaLeft;
    float luma2 = isHoriz ? lumaDown : lumaRight;

    /* Compute their gradients. */
    float gradient1 = luma1 - lumaCenter;
    float gradient2 = luma2 - lumaCenter;

    /* Choose the steeper gradient. */
    bool grad1Steeper = fabsf(gradient1) >= fabsf(gradient2);

    /* Normalize the gradients. */
    float gradientNorm = 0.25f * fmaxf(fabsf(gradient1), fabsf(gradient2));

    /* Determine directional luma average. */
    float lumaLocalAvg;

    if (grad1Steeper) {
        lumaLocalAvg = 0.5f * (luma1 + lumaCenter);
    }
    else {
        lumaLocalAvg = 0.5f * (luma2 + lumaCenter);
    }

    /* Subsample locations for each iteration. */
    float iteration1S = static_cast<float>(s);
    float iteration1T = static_cast<float>(t);
    float iteration2S = iteration1S;
    float iteration2T = iteration1T;

    /* Offset our sample locations toward the edge by half a pixel. */
    if (isHoriz) {
        iteration1T += grad1Steeper ? -0.5f : 0.5f;
        iteration2T += grad1Steeper ? -0.5f : 0.5f;
    }
    else {
        iteration1S += grad1Steeper ? -0.5f : 0.5f;
        iteration2S += grad1Steeper ? -0.5f : 0.5f;
    }

    /* Determine iteration offsets. */
    size_t offsetS = isHoriz ? 1 : 0;
    size_t offsetT = isHoriz ? 0 : 1;

    iteration1S -= offsetS;
    iteration1T -= offsetT;

    iteration2S += offsetS;
    iteration2T += offsetT;

    /* Iterate! */
    float lumaEnd1;
    float lumaEnd2;
    float lumaDiff1;
    float lumaDiff2;

    bool reached1 = false;
    bool reached2 = false;

    for (size_t i=0; i<MAX_ITERATIONS; i++) {
        /* Sample lumas in both directions along the edge. */
        if (!reached1) {
            lumaEnd1 = luma_from_rgb(get_subsample(iteration1S, iteration1T));
            lumaDiff1 = lumaEnd1 - lumaLocalAvg;
        }

        if (!reached2) {
            lumaEnd2 = luma_from_rgb(get_subsample(iteration2S, iteration2T));
            lumaDiff2 = lumaEnd2 - lumaLocalAvg;
        }

        /* Did we reach the end of the edge? */
        reached1 = lumaDiff1 < 0.0f && (fabsf(lumaDiff1) >= gradientNorm);
        reached2 = lumaDiff2 < 0.0f && (fabsf(lumaDiff2) >= gradientNorm);

        /* If we've reached the end, stop iteration. */
        if (reached1 && reached2) {
            break;
        }

        /* But if we HAVEN'T reached the end, continue... */
        if (!reached1) {
            iteration1S -= offsetS;
            iteration1T -= offsetT;
        }

        if (!reached2) {
            iteration2S += offsetS;
            iteration2T += offsetT;
        }
    }

    /* Determine how far we've traveled along the edge. */
    float dist1 = isHoriz ? (s - iteration1S) : (t - iteration1T);
    float dist2 = isHoriz ? (iteration2S - s) : (iteration2T - t);

    /* Which way is closer? */
    bool dir1Closer = dist1 < dist2;
    float closerDist = fminf(dist1, dist2);

    /* Total length of the edge. */
    float edgeLen = dist1 + dist2;

    /*
     * The pixel offset where we should subsample, in the direction of the
     * closer edge endpoint.
     */
    float pixelOffset;

    if ((lumaCenter < lumaLocalAvg)
            != ((dir1Closer ? lumaEnd1 : lumaEnd2) < 0.0f)) {
        pixelOffset = 0.0f;
    }
    else {
        pixelOffset = -closerDist / edgeLen + 0.5f;
    }

    //printf(
    //    "(%u, %u) %s distance: %f / %f (%f) Offset: %f\n",
    //    static_cast<unsigned int>(s), static_cast<unsigned int>(t),
    //    isHoriz ? "horizontal" : "vertical",
    //    closerDist, edgeLen, closerDist / edgeLen,
    //    pixelOffset
    //);

    ///*
    // * Subpixel antialiasing
    // */

    ///* Weighted average of all the lumas in our local 3x3 grid. */
    //float lumaAvg = (
    //    (1.0 / 12.0) * (2.0 * (lumaUD + lumaLR) + lumaULDL + lumaURDR)
    //);

    //float subpixelOffset1 = clamp(
    //    fabsf(lumaAvg - lumaCenter) / lumaRange,
    //    0.0, 1.0
    //);
    //float subpixelOffset2 = (
    //    (-2.0 * subpixelOffset1 + 3.0) * subpixelOffset1 * subpixelOffset1
    //);

    //float subpixelOffset = (
    //    subpixelOffset2 * subpixelOffset2 * SUBPIXEL_QUALITY
    //);

    //float finalOffset = fmaxf(subpixelOffset, pixelOffset);
    float finalOffset = pixelOffset;

    if (grad1Steeper) {
        finalOffset = -finalOffset;
    }

    /* Determine the final subsample coordinates. */
    float finalS = static_cast<float>(s);
    float finalT = static_cast<float>(t);

    if (isHoriz) {
        finalT += finalOffset;
    }
    else {
        finalS += finalOffset;
    }

    /* Final subsample... */
    float3 color = get_subsample(finalS, finalT);

    //{
    //    int s0 = static_cast<int>(floorf(s));
    //    int t0 = static_cast<int>(floorf(t));
    //    int s1 = s0 + 1;
    //    int t1 = t0 + 1;

    //    if (s0 < 0) {
    //        s0 = 0;
    //    }

    //    if (t0 < 0) {
    //        t0 = 0;
    //    }

    //    if (s1 >= width) {
    //        s1 = width - 1;
    //    }

    //    if (t1 >= height) {
    //        t1 = height - 1;
    //    }

    //    float3 sampleUL = samplesIn[t0 * width + s0];
    //    float3 sampleUR = samplesIn[t0 * width + s1];
    //    float3 sampleDL = samplesIn[t1 * width + s0];
    //    float3 sampleDR = samplesIn[t1 * width + s1];

    //    printf(
    //        "(%u, %u) sampled at (%f, %f)\n"
    //        "\tUL(%f, %f, %f) UR(%f, %f, %f)\n"
    //        "\tDL(%f, %f, %f) DR(%f, %f, %f)\n"
    //        "\tyields (%f, %f, %f)\n",
    //        static_cast<unsigned int>(s), static_cast<unsigned int>(t),
    //        finalS, finalT,
    //        sampleUL.x, sampleUL.y, sampleUL.z,
    //        sampleUR.x, sampleUR.y, sampleUR.z,
    //        sampleDL.x, sampleDL.y, sampleDL.z,
    //        sampleDR.x, sampleDR.y, sampleDR.z,
    //        color.x, color.y, color.z
    //    );
    //}

    //color = isHoriz ?
    //    make_float3(color.x * 10.0, color.y, color.z) :
    //    make_float3(color.x, color.y * 10.0, color.z);

    /* ... and we're done! */
    samplesOut[static_cast<size_t>(t * width + s)] = color;
}


__global__ void map_faces(
    CUDABSP::CUDABSP* pCudaBSP,
    CUDARAD::FaceInfo* faceInfos
) {

    size_t faceIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (faceIndex >= pCudaBSP->numFaces) {
        return;
    }

    faceInfos[faceIndex] = CUDARAD::FaceInfo(*pCudaBSP, faceIndex);
}

namespace CUDAFXAA {

    void antialias_lightsamples(CUDABSP::CUDABSP* pCudaBSP) {
        size_t numFaces;

        CUDA_CHECK_ERROR(
            cudaMemcpy(
                &numFaces,
                &pCudaBSP->numFaces,
                sizeof(size_t),
                cudaMemcpyDeviceToHost
            )
        );

        CUDARAD::FaceInfo* faceInfos;

        CUDA_CHECK_ERROR(
            cudaMalloc(&faceInfos, sizeof(CUDARAD::FaceInfo) * numFaces)
        );

        const size_t FACE_BLOCK_WIDTH = 128;
        size_t numBlocks = div_ceil(numFaces, FACE_BLOCK_WIDTH);

        KERNEL_LAUNCH(
            map_faces,
            numBlocks, FACE_BLOCK_WIDTH,
            pCudaBSP, faceInfos
        );

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        CUDARAD::FaceInfo* hostFaceInfos =
            (CUDARAD::FaceInfo*)malloc(sizeof(CUDARAD::FaceInfo) * numFaces);

        CUDA_CHECK_ERROR(
            cudaMemcpy(
                hostFaceInfos,
                faceInfos,
                sizeof(CUDARAD::FaceInfo) * numFaces,
                cudaMemcpyDeviceToHost
            )
        );

        const size_t BLOCK_WIDTH = 16;
        const size_t BLOCK_HEIGHT = 16;

        for (size_t faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
            CUDARAD::FaceInfo& faceInfo = hostFaceInfos[faceIndex];

            size_t width = faceInfo.lightmapWidth;
            size_t height = faceInfo.lightmapHeight;
            size_t numSamples = faceInfo.lightmapSize;
            size_t startIndex = faceInfo.lightmapStartIndex;

            if (width == 0 || height == 0 || numSamples == 0) {
                continue;
            }

            float3* lightSamples = &pCudaBSP->lightSamples[startIndex];

            float3* results = nullptr;

            CUDA_CHECK_ERROR(
                cudaMalloc(&results, sizeof(float3) * numSamples)
            );

            dim3 gridDim(
                div_ceil(width, BLOCK_WIDTH),
                div_ceil(height, BLOCK_HEIGHT)
            );

            dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);

            KERNEL_LAUNCH(
                map_samples_fxaa,
                gridDim, blockDim,
                pCudaBSP, &faceInfos[faceIndex],
                lightSamples, results,
                width, height
            );

            CUDA_CHECK_ERROR(cudaDeviceSynchronize());

            CUDA_CHECK_ERROR(
                cudaMemcpy(
                    lightSamples,
                    results,
                    sizeof(float3) * numSamples,
                    cudaMemcpyDeviceToDevice
                )
            );

            CUDA_CHECK_ERROR(cudaFree(results));
        }

        free(hostFaceInfos);

        CUDA_CHECK_ERROR(cudaFree(faceInfos));
    }

}
