#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace OptixRT
{
    struct SunRay {
        float3 origin;
        float3 direction;
        float tmin;
        float tmax;
        uint32_t visibilityMask;
    };

    struct RayHit {
        uint8_t hit;
        uint32_t primitiveIndex;
        float t;
    };

    struct Params {
        OptixTraversableHandle world;
        SunRay* rays;
        RayHit* hits;
        uint32_t ray_count;
    };

    extern "C" {
        __constant__ Params optix_params;
    }

    extern "C" __global__ void __raygen__los_blocked_sun()
    {
        uint32_t idx = optixGetLaunchIndex().x;
        if (idx >= optix_params.ray_count)
            return;

        SunRay r = optix_params.rays[idx];

        unsigned int hit = 0;
        unsigned int primitiveIndex = 0xffffffffu;
        unsigned int hitTBits = 0u;

        optixTrace(
            optix_params.world,
            r.origin,
            r.direction,
            r.tmin,
            r.tmax,
            0.0f,
            OptixVisibilityMask(r.visibilityMask & 0xffu),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0, 1, 0,
            hit,
            primitiveIndex,
            hitTBits
        );

        RayHit out = {};
        out.hit = hit ? 1 : 0;
        out.primitiveIndex = primitiveIndex;
        out.t = __uint_as_float(hitTBits);
        optix_params.hits[idx] = out;
    }

    extern "C" __global__ void __miss__los_blocked_sun()
    {
        optixSetPayload_0(0);
        optixSetPayload_1(0xffffffffu);
        optixSetPayload_2(0u);
    }

    extern "C" __global__ void __closesthit__los_blocked_sun()
    {
        optixSetPayload_0(1);
        optixSetPayload_1(optixGetPrimitiveIndex());
        optixSetPayload_2(__float_as_uint(optixGetRayTmax()));
    }
}
