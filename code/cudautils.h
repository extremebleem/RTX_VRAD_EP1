#ifndef __CUDAUTILS_H_
#define __CUDAUTILS_H_

#include <iostream>
#include <cstdlib>
#include <cstdio>

#ifdef __CUDACC__
#include "cub/cub.cuh"
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bsp.h"


inline void cuda_assert(
        cudaError_t code, const char* file, int line,
        bool abort=true
        ) {

    if (code != cudaSuccess) {
        std::cerr << "cuda_assert(): "
            << cudaGetErrorString(code) << " (File '"
            << file << "', Line "
            << line << ")"
            << std::endl;

        if (abort) {
            exit(code);
        }
    }
}


inline __device__ void cuda_device_abort(void) {
#ifdef __CUDACC__
    asm("trap;");
#endif
}


template <typename T, size_t COUNT, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
inline __device__ void prefix_sum(T (&in)[COUNT], T (&out)[COUNT]) {
    static_assert(
        COUNT % (BLOCK_WIDTH * BLOCK_HEIGHT) == 0,
        "prefix_sum() element count must be divisible by thread count!"
    );

#ifdef __CUDACC__
    const size_t ITEMS_PER_THREAD = COUNT / (BLOCK_WIDTH * BLOCK_HEIGHT);

    using Loader = cub::BlockLoad<
        T,
        BLOCK_WIDTH,
        ITEMS_PER_THREAD,
        cub::BLOCK_LOAD_DIRECT,
        BLOCK_HEIGHT
    >;

    using Scanner = cub::BlockScan<
        T,
        BLOCK_WIDTH,
        cub::BLOCK_SCAN_RAKING,
        BLOCK_HEIGHT
    >;

    using Storer = cub::BlockStore<
        T,
        BLOCK_WIDTH,
        ITEMS_PER_THREAD,
        cub::BLOCK_STORE_DIRECT,
        BLOCK_HEIGHT
    >;

    __shared__ union {
        typename Loader::TempStorage load;
        typename Scanner::TempStorage scan;
        typename Storer::TempStorage store;
    } tempStorage;

    T data[ITEMS_PER_THREAD];

    Loader(tempStorage.load).Load(in, data);

    __syncthreads();

    Scanner(tempStorage.scan).ExclusiveSum(data, data);

    __syncthreads();

    Storer(tempStorage.store).Store(out, data);

    __syncthreads();
#endif
}

template<
    typename T,
    size_t COUNT,
    size_t BLOCK_WIDTH,
    size_t BLOCK_HEIGHT
>
__device__ inline void prefix_sum(T* in, T(&out)[COUNT]) {
    out[0] = in[0];

    for (size_t i = 1; i < COUNT; ++i) {
        out[i] = out[i - 1] + in[i];
    }
}

template<
    typename T,
    size_t COUNT,
    size_t BLOCK_WIDTH,
    size_t BLOCK_HEIGHT
>
__device__ inline void prefix_sum(T* in, T* out) {
    out[0] = in[0];

    for (size_t i = 1; i < COUNT; ++i) {
        out[i] = out[i - 1] + in[i];
    }
}

/** Block-wide filter algorithm. */
template <
    typename T,
    size_t COUNT,
    size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT,
    typename F
>
inline __device__ size_t filter(F predicate, T (&in)[COUNT], T (&out)[COUNT]) {
    static_assert(
        COUNT % (BLOCK_WIDTH * BLOCK_HEIGHT) == 0,
        "filter() element count must be divisible by thread count!"
    );

#ifdef __CUDACC__
    size_t threadID = threadIdx.y * BLOCK_WIDTH + threadIdx.x;
    const size_t THREADS_PER_BLOCK = BLOCK_WIDTH * BLOCK_HEIGHT;

    __shared__ uint32_t selected[COUNT];

    /* Select all elements that match the predicate. */
    for (size_t i=threadID; i<COUNT; i+=THREADS_PER_BLOCK) {
        if (i >= COUNT) {
            continue;
        }

        selected[i] = (predicate(in[i])) ? 1 : 0;
    }

    __syncthreads();

    __shared__ uint32_t scanned[COUNT];

    /* Parallel prefix sum voodoo magic. */
    prefix_sum<T, COUNT, BLOCK_WIDTH, BLOCK_HEIGHT>(selected, scanned);

    __syncthreads();

    /* Gather all elements. */
    for (size_t i=threadID; i<COUNT; i+=THREADS_PER_BLOCK) {
        if (i >= COUNT) {
            continue;
        }

        if (selected[i]) {
            size_t finalIndex = scanned[i];
            out[finalIndex] = in[i];
        }
    }

    __syncthreads();

    // Return the number of elements that matched the predicate.
    return scanned[COUNT - 1] + selected[COUNT - 1];
#else
    return 0;
#endif
}


//inline __device__ __host__ cudaError_t _cudaMalloc(void** ptr, size_t size) {
//    printf("Allocate %u\n", static_cast<unsigned int>(size));
//    return cudaMalloc(ptr, size);
//}
//
//
//#define cudaMalloc(ptr, size) _cudaMalloc(reinterpret_cast<void**>(ptr), (size))


#define CUDA_CHECK_ERROR(ans) do {\
    cuda_assert((ans), __FILE__, __LINE__);\
} while (0)


#define CUDA_CHECK_ERROR_DEVICE(code) do {\
    if ((code) != cudaSuccess) {\
        printf(\
            "CUDA Device Error: %s (File '%s', Line %d)\n",\
            cudaGetErrorString(code), __FILE__, __LINE__\
        );\
        return cuda_device_abort();\
    }\
} while (0)


#define CUDA_RETRY_UNTIL_SUCCESS(action) while (1) {\
    cudaError code = (action);\
    if (code == cudaSuccess) {\
        break;\
    }\
    else {\
        printf(\
            "CUDA Device Error: %s (File '%s', Line %d)... Retrying!\n",\
            cudaGetErrorString(code), __FILE__, __LINE__\
        );\
    }\
}


// If we don't do this, Visual Studio will keep flipping its $#!& because it
// apparently can't parse kernel launch syntax correctly.
#define KERNEL_LAUNCH(kernel, gridDim, blockDim, ...) do {\
    kernel<<<gridDim, blockDim>>>(__VA_ARGS__);\
    CUDA_CHECK_ERROR(cudaPeekAtLastError());\
} while (0)


#define KERNEL_LAUNCH_SHARED(kernel, gridDim, blockDim, shared, ...) do {\
    kernel<<<gridDim, blockDim, shared>>>(__VA_ARGS__);\
    CUDA_CHECK_ERROR(cudaPeekAtLastError());\
} while (0)


#define KERNEL_LAUNCH_DEVICE(kernel, gridDim, blockDim, ...) do {\
    kernel<<<gridDim, blockDim>>>(__VA_ARGS__);\
    CUDA_CHECK_ERROR_DEVICE(cudaPeekAtLastError());\
} while (0)


#define KERNEL_LAUNCH_SHARED_DEVICE(kernel, gridDim, blockDim, shared, ...) \
do {\
    kernel<<<gridDim, blockDim, shared>>>(__VA_ARGS__);\
    CUDA_CHECK_ERROR_DEVICE(cudaPeekAtLastError());\
} while (0)


inline __host__ __device__ size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}


inline constexpr __host__ __device__
size_t const_div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}


/** Hack to get the Windows WDDM thingy to work properly. */
inline __host__ void flush_wddm_queue(void) {
    cudaEvent_t event;

    CUDA_CHECK_ERROR(cudaEventCreate(&event));
    CUDA_CHECK_ERROR(cudaEventRecord(event));

    cudaEventQuery(event);

    // Acknowledge and ignore the error generated by cudaEventQuery().
    cudaGetLastError();
}


inline __host__ __device__ float3 make_float3(float f) {
    return make_float3(f, f, f);
}


inline __host__ __device__ float3 make_float3(void) {
    return make_float3(0.0f);
}


inline __host__ __device__ float3 make_float3(const BSP::Vec3<float>& f) {
    return make_float3(f.x, f.y, f.z);
}


inline __host__ __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


inline __host__ __device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}


inline __host__ __device__ float3 operator-(
        const float3& a, const float3& b
        ) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


inline __host__ __device__ float3 operator+(
        const float3& a, const float3& b
        ) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


inline __host__ __device__ float3& operator+=(
        float3& a, const float3& b
        ) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}


inline __host__ __device__ float3& operator-=(
        float3& a, const float3& b
        ) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}


inline __host__ __device__ float3 operator*(const float3& v, float c) {
    return make_float3(v.x * c, v.y * c, v.z * c);
}


inline __host__ __device__ float3 operator*=(float3& a, float c) {
    a.x *= c;
    a.y *= c;
    a.z *= c;
    return a;
}


inline __host__ __device__ float3 operator*(float c, const float3& v) {
    return v * c;
}


inline __host__ __device__ float3 operator/(const float3& v, float c) {
    return v * (1.0f / c);
}


inline __host__ __device__ float3 operator/=(float3& a, float c) {
    a.x /= c;
    a.y /= c;
    a.z /= c;
    return a;
}


inline __host__ __device__ float dist(const float3& a, const float3& b) {
    float3 diff = b - a;
    return sqrt(dot(diff, diff));
}


inline __host__ __device__ float len(const float3& v) {
    return dist(make_float3(), v);
}


inline __host__ __device__ float3 normalized(const float3& v) {
    return v / len(v);
}


#endif
