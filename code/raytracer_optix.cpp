#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "raytracer_optix.h"

namespace OptixRT
{
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t rc__ = (call);                                                       \
        if (rc__ != cudaSuccess) {                                                       \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(rc__)); \
        }                                                                                \
    } while (0)

#define OPTIX_CHECK(call)                                                                \
    do {                                                                                 \
        OptixResult rc__ = (call);                                                       \
        if (rc__ != OPTIX_SUCCESS) {                                                     \
            throw std::runtime_error(std::string("OptiX error code: ") + std::to_string((int)rc__)); \
        }                                                                                \
    } while (0)

    static void optix_log_callback(unsigned int level, const char* tag, const char* message, void*)
    {
        // Replace with Msg()/Warning() in VRAD if desired.
        (void)level;
        (void)tag;
        (void)message;
    }

    // -------------------------------------------------------------------------------------------------
    // Input/output structs shared by host and device
    // -------------------------------------------------------------------------------------------------

    struct RaytracerParams
    {
        OptixTraversableHandle world;
        SunRay* rays;
        RayHit* hits;
        uint32_t ray_count;
    };

    // -------------------------------------------------------------------------------------------------
    // SBT records
    // -------------------------------------------------------------------------------------------------

    template <typename T>
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
    {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    struct EmptySbtData {};

    using RaygenRecord = SbtRecord<EmptySbtData>;
    using MissRecord = SbtRecord<EmptySbtData>;
    using HitgroupRecord = SbtRecord<EmptySbtData>;

    // -------------------------------------------------------------------------------------------------
    // Host-side OptiX raytracer
    // -------------------------------------------------------------------------------------------------

    class OptixSunLosTracer::Impl
    {
    public:
        Impl() = default;
        ~Impl() { destroy(); }

        Impl(const Impl&) = delete;
        Impl& operator=(const Impl&) = delete;

        void init(const char* ptx, size_t ptx_size)
        {
            CUDA_CHECK(cudaFree(nullptr));
            OPTIX_CHECK(optixInit());

            OptixDeviceContextOptions ctx_options = {};
            ctx_options.logCallbackFunction = &optix_log_callback;
            ctx_options.logCallbackLevel = 3;

            CUcontext cu_ctx = 0; // current CUDA context
            OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &ctx_options, &m_context));

            create_module(ptx, ptx_size);
            create_program_groups();
            create_pipeline();
            create_sbt();
        }

        void destroy()
        {
            if (m_d_temp_gas)      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_temp_gas)));
            if (m_d_gas_output)    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_gas_output)));
            if (m_d_vertices)      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_vertices)));
            if (m_d_indices)       CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_indices)));
            if (m_d_raygen_record) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_raygen_record)));
            if (m_d_miss_record)   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_miss_record)));
            if (m_d_hit_record)    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_hit_record)));
            if (m_pipeline)        OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
            if (m_raygen_pg)       OPTIX_CHECK(optixProgramGroupDestroy(m_raygen_pg));
            if (m_miss_pg)         OPTIX_CHECK(optixProgramGroupDestroy(m_miss_pg));
            if (m_hitgroup_pg)     OPTIX_CHECK(optixProgramGroupDestroy(m_hitgroup_pg));
            if (m_module)          OPTIX_CHECK(optixModuleDestroy(m_module));
            if (m_context)         OPTIX_CHECK(optixDeviceContextDestroy(m_context));

            m_d_temp_gas = 0;
            m_d_gas_output = 0;
            m_d_vertices = 0;
            m_d_indices = 0;
            m_d_raygen_record = 0;
            m_d_miss_record = 0;
            m_d_hit_record = 0;
            m_world = 0;
            m_context = nullptr;
            m_module = nullptr;
            m_raygen_pg = nullptr;
            m_miss_pg = nullptr;
            m_hitgroup_pg = nullptr;
            m_pipeline = nullptr;
        }

        // Prepare OptiX geometry from your current world/BSP triangles.
        // First port version intentionally duplicates vertices: 3 vertices per triangle.
        // Optimize/deduplicate later only after correctness matches the old tracer.
        void build_world_gas(const std::vector<Triangle>& triangles)
        {
            if (triangles.empty()) {
                throw std::runtime_error("build_world_gas: no triangles");
            }

            std::vector<float3> vertices;
            std::vector<uint3> indices;
            vertices.reserve(triangles.size() * 3);
            indices.reserve(triangles.size());

            for (size_t i = 0; i < triangles.size(); ++i) {
                const Triangle& t = triangles[i];

                // Important Source/VRAD note:
                // If old manual tracer had to flip winding for Source CW faces, you have two safe choices:
                //   1) leave winding as-is and disable culling, which this first port does;
                //   2) later flip to v2,v1,v0 and enable culling only after validation.
                const uint32_t base = static_cast<uint32_t>(vertices.size());
                vertices.push_back(t.v0);
                vertices.push_back(t.v1);
                vertices.push_back(t.v2);
                indices.push_back(make_uint3(base + 0, base + 1, base + 2));
            }

            m_num_triangles = static_cast<uint32_t>(indices.size());

            const size_t vertices_bytes = vertices.size() * sizeof(float3);
            const size_t indices_bytes = indices.size() * sizeof(uint3);

            if (m_d_vertices) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_vertices)));
            if (m_d_indices)  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_indices)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_vertices), vertices_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_indices), indices_bytes));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_d_vertices), vertices.data(), vertices_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_d_indices), indices.data(), indices_bytes, cudaMemcpyHostToDevice));

            OptixBuildInput build_input = {};
            build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            CUdeviceptr vertex_buffers[] = { m_d_vertices };
            uint32_t triangle_input_flags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

            build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
            build_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
            build_input.triangleArray.vertexBuffers = vertex_buffers;

            build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
            build_input.triangleArray.numIndexTriplets = m_num_triangles;
            build_input.triangleArray.indexBuffer = m_d_indices;

            build_input.triangleArray.flags = triangle_input_flags;
            build_input.triangleArray.numSbtRecords = 1;
            build_input.triangleArray.sbtIndexOffsetBuffer = 0;
            build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
            build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes gas_sizes = {};
            OPTIX_CHECK(optixAccelComputeMemoryUsage(
                m_context,
                &accel_options,
                &build_input,
                1,
                &gas_sizes));

            if (m_d_temp_gas)   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_temp_gas)));
            if (m_d_gas_output) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_gas_output)));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_temp_gas), gas_sizes.tempSizeInBytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_gas_output), gas_sizes.outputSizeInBytes));

            OPTIX_CHECK(optixAccelBuild(
                m_context,
                0,
                &accel_options,
                &build_input,
                1,
                m_d_temp_gas,
                gas_sizes.tempSizeInBytes,
                m_d_gas_output,
                gas_sizes.outputSizeInBytes,
                &m_world,
                nullptr,
                0));

            CUDA_CHECK(cudaDeviceSynchronize());
        }

        void trace_batch(const std::vector<SunRay>& rays, std::vector<RayHit>& hits)
        {
            if (!m_world) {
                throw std::runtime_error("trace_batch: GAS is not built");
            }

            hits.resize(rays.size());
            if (rays.empty()) {
                return;
            }

            CUdeviceptr d_rays = 0;
            CUdeviceptr d_hits = 0;
            CUdeviceptr d_params = 0;

            const size_t rays_bytes = rays.size() * sizeof(SunRay);
            const size_t hits_bytes = rays.size() * sizeof(RayHit);

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rays), rays_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hits), hits_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(RaytracerParams)));

            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rays), rays.data(), rays_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_hits), 0, hits_bytes));

            RaytracerParams params = {};
            params.world = m_world;
            params.rays = reinterpret_cast<SunRay*>(d_rays);
            params.hits = reinterpret_cast<RayHit*>(d_hits);
            params.ray_count = static_cast<uint32_t>(rays.size());

            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(params), cudaMemcpyHostToDevice));

            OPTIX_CHECK(optixLaunch(
                m_pipeline,
                0,
                d_params,
                sizeof(RaytracerParams),
                &m_sbt,
                static_cast<unsigned int>(rays.size()),
                1,
                1));

            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hits.data(), reinterpret_cast<void*>(d_hits), hits_bytes, cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rays)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hits)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
        }

        // Batch version of los_blocked_sun.
        // Returns blocked[i] = 1 if the ray hits world geometry before tmax.
        void los_blocked_sun_batch(const std::vector<SunRay>& rays, std::vector<uint8_t>& blocked)
        {
            std::vector<RayHit> hits;
            trace_batch(rays, hits);

            blocked.resize(hits.size());
            for (size_t i = 0; i < hits.size(); ++i) {
                blocked[i] = hits[i].hit ? 1 : 0;
            }
        }

    private:
        void create_module(const char* ptx, size_t ptx_size)
        {
            OptixModuleCompileOptions module_options = {};
            module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

            m_pipeline_options = {};
            m_pipeline_options.usesMotionBlur = false;
            m_pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            m_pipeline_options.numPayloadValues = 3;
            m_pipeline_options.numAttributeValues = 2;
            m_pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
            m_pipeline_options.pipelineLaunchParamsVariableName = "optix_params";

            char log[4096];
            size_t log_size = sizeof(log);
            OPTIX_CHECK(optixModuleCreate(
                m_context,
                &module_options,
                &m_pipeline_options,
                ptx,
                ptx_size,
                log,
                &log_size,
                &m_module));
        }

        void create_program_groups()
        {
            OptixProgramGroupOptions pg_options = {};
            char log[4096];
            size_t log_size = sizeof(log);

            OptixProgramGroupDesc raygen_desc = {};
            raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_desc.raygen.module = m_module;
            raygen_desc.raygen.entryFunctionName = "__raygen__los_blocked_sun";
            OPTIX_CHECK(optixProgramGroupCreate(m_context, &raygen_desc, 1, &pg_options, log, &log_size, &m_raygen_pg));

            log_size = sizeof(log);
            OptixProgramGroupDesc miss_desc = {};
            miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_desc.miss.module = m_module;
            miss_desc.miss.entryFunctionName = "__miss__los_blocked_sun";
            OPTIX_CHECK(optixProgramGroupCreate(m_context, &miss_desc, 1, &pg_options, log, &log_size, &m_miss_pg));

            log_size = sizeof(log);
            OptixProgramGroupDesc hit_desc = {};
            hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_desc.hitgroup.moduleCH = m_module;
            hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__los_blocked_sun";
            hit_desc.hitgroup.moduleAH = nullptr;
            hit_desc.hitgroup.entryFunctionNameAH = nullptr;
            hit_desc.hitgroup.moduleIS = nullptr;
            hit_desc.hitgroup.entryFunctionNameIS = nullptr;
            OPTIX_CHECK(optixProgramGroupCreate(m_context, &hit_desc, 1, &pg_options, log, &log_size, &m_hitgroup_pg));
        }

        void create_pipeline()
        {
            std::vector<OptixProgramGroup> groups = { m_raygen_pg, m_miss_pg, m_hitgroup_pg };

            OptixPipelineLinkOptions link_options = {};
            link_options.maxTraceDepth = 1;
            link_options.maxContinuationCallableDepth = 0;
            link_options.maxDirectCallableDepthFromState = 0;
            link_options.maxDirectCallableDepthFromTraversal = 0;
            link_options.maxTraversableGraphDepth = 1;

            char log[4096];
            size_t log_size = sizeof(log);
            OPTIX_CHECK(optixPipelineCreate(
                m_context,
                &m_pipeline_options,
                &link_options,
                groups.data(),
                static_cast<unsigned int>(groups.size()),
                log,
                &log_size,
                &m_pipeline));

            OPTIX_CHECK(optixPipelineSetStackSize(
                m_pipeline,
                2 * 1024, // direct callable stack from traversal
                2 * 1024, // direct callable stack from state
                2 * 1024, // continuation stack
                1));      // max traversable graph depth: single GAS
        }

        void create_sbt()
        {
            RaygenRecord rg = {};
            MissRecord ms = {};
            HitgroupRecord hg = {};

            OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_pg, &rg));
            OPTIX_CHECK(optixSbtRecordPackHeader(m_miss_pg, &ms));
            OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroup_pg, &hg));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_raygen_record), sizeof(RaygenRecord)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_miss_record), sizeof(MissRecord)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_hit_record), sizeof(HitgroupRecord)));

            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_d_raygen_record), &rg, sizeof(rg), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_d_miss_record), &ms, sizeof(ms), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_d_hit_record), &hg, sizeof(hg), cudaMemcpyHostToDevice));

            m_sbt = {};
            m_sbt.raygenRecord = m_d_raygen_record;
            m_sbt.missRecordBase = m_d_miss_record;
            m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
            m_sbt.missRecordCount = 1;
            m_sbt.hitgroupRecordBase = m_d_hit_record;
            m_sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
            m_sbt.hitgroupRecordCount = 1;
        }

    private:
        OptixDeviceContext m_context = nullptr;
        OptixModule m_module = nullptr;
        OptixProgramGroup m_raygen_pg = nullptr;
        OptixProgramGroup m_miss_pg = nullptr;
        OptixProgramGroup m_hitgroup_pg = nullptr;
        OptixPipeline m_pipeline = nullptr;
        OptixPipelineCompileOptions m_pipeline_options = {};
        OptixShaderBindingTable m_sbt = {};

        OptixTraversableHandle m_world = 0;

        CUdeviceptr m_d_vertices = 0;
        CUdeviceptr m_d_indices = 0;
        CUdeviceptr m_d_temp_gas = 0;
        CUdeviceptr m_d_gas_output = 0;

        CUdeviceptr m_d_raygen_record = 0;
        CUdeviceptr m_d_miss_record = 0;
        CUdeviceptr m_d_hit_record = 0;

        uint32_t m_num_triangles = 0;
    };

    OptixSunLosTracer::OptixSunLosTracer()
        : m_impl(new Impl())
    {
    }

    OptixSunLosTracer::~OptixSunLosTracer() = default;

    void OptixSunLosTracer::init(const char* ptx, size_t ptx_size)
    {
        m_impl->init(ptx, ptx_size);
    }

    void OptixSunLosTracer::build_world_gas(const std::vector<Triangle>& triangles)
    {
        m_impl->build_world_gas(triangles);
    }

    void OptixSunLosTracer::trace_batch(
        const std::vector<SunRay>& rays,
        std::vector<RayHit>& hits
    ) {
        m_impl->trace_batch(rays, hits);
    }

    void OptixSunLosTracer::los_blocked_sun_batch(
        const std::vector<SunRay>& rays,
        std::vector<uint8_t>& blocked
    ) {
        m_impl->los_blocked_sun_batch(rays, blocked);
    }

    // -------------------------------------------------------------------------------------------------
    // Helper for generating sun rays from VRAD-side data
    // -------------------------------------------------------------------------------------------------

    static inline float3 normalize3(float3 v)
    {
        const float len2 = v.x * v.x + v.y * v.y + v.z * v.z;
        if (len2 <= 0.0f) return make_float3(0.0f, 0.0f, 1.0f);
        const float inv_len = 1.0f / std::sqrt(len2);
        return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
    }

    SunRay make_sun_los_ray(float3 start, float3 sun_dir, float max_dist)
    {
        // Offset avoids self-intersection acne. Tune against old VRAD epsilon.
        const float kRayEpsilon = 0.25f;

        SunRay r = {};
        r.direction = normalize3(sun_dir);
        r.origin = make_float3(
            start.x + r.direction.x * kRayEpsilon,
            start.y + r.direction.y * kRayEpsilon,
            start.z + r.direction.z * kRayEpsilon);
        r.tmin = 0.0f;
        r.tmax = max_dist;
        return r;
    }
}
