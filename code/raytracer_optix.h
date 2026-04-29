#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace OptixRT {
    struct Triangle {
        float3 v0;
        float3 v1;
        float3 v2;
        uint32_t sourceId = 0;
        uint32_t role = 0;
        uint32_t visibilityMask = 0xff;
    };

    struct SunRay {
        float3 origin;
        float3 direction;
        float tmin;
        float tmax;
        uint32_t visibilityMask = 0xff;
    };

    struct RayHit {
        uint8_t hit = 0;
        uint32_t primitiveIndex = 0xffffffffu;
        float t = 0.0f;
    };

    class OptixSunLosTracer {
        public:
            OptixSunLosTracer();
            ~OptixSunLosTracer();

            OptixSunLosTracer(const OptixSunLosTracer& other) = delete;
            OptixSunLosTracer& operator=(const OptixSunLosTracer& other) = delete;

            void init(const char* ptx, size_t ptx_size);
            void build_world_gas(const std::vector<Triangle>& triangles);
            void trace_batch(
                const std::vector<SunRay>& rays,
                std::vector<RayHit>& hits
            );
            void los_blocked_sun_batch(
                const std::vector<SunRay>& rays,
                std::vector<uint8_t>& blocked
            );

        private:
            class Impl;
            std::unique_ptr<Impl> m_impl;
    };

    SunRay make_sun_los_ray(float3 start, float3 sun_dir, float max_dist);
}
