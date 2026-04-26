#ifndef __RAYTRACER_H_
#define __RAYTRACER_H_

#include <vector>

#include "cuda_runtime.h"
#include "bsp.h"

namespace RayTracer {
    struct Triangle {
        float3 vertices[3];
        int faceId;
        int32_t flags;
    };

    enum class KDNodeType {NODE, LEAF};
    enum class Axis {X, Y, Z};

    struct KDNode {
        KDNodeType type;
        Axis axis;
        float pos;

        float3 tmin;
        float3 tmax;

        size_t* triangleIDs;
        size_t numTris;

        KDNode* children;
    };

    struct RayHit {
        bool hit;
        size_t triangleId;
        size_t faceId;
        float t;
        float3 position;
    };

    //class KDTree {
    //    private:
    //        KDNode* m_root;
    //
    //    public:
    //        KDTree();
    //};

    /**
     * Ray-trace acceleration structure whose data resides solely in device
     * memory.
     */
    class CUDARayTracer {
        private:
            static const size_t MAX_LEAVES;

            Triangle* m_triangles;
            size_t* m_triangleIDs;
            size_t m_numTriangles;

            KDNode* m_pTreeRoot;

            float3 m_tmin;
            float3 m_tmax;

            __host__ void build_tree(void);
            __host__ void destroy_tree(void);

        public:
            CUDARayTracer();
            CUDARayTracer(const CUDARayTracer& other) = delete;
            ~CUDARayTracer();

            CUDARayTracer& operator=(const CUDARayTracer& other) = delete;

            __host__ void add_triangles(const std::vector<Triangle>& tris);

            __device__ RayHit trace_closest(
                const float3& start,
                const float3& end
            );

            //__device__ Triangle* get_triangles(void);
            //__device__ size_t* get_tri_ids(void);
            //__device__ Triangle& get_tri_indirect(size_t i);

            __device__ bool LOS_blocked(
                const float3& start, const float3& end
            );

            __device__ bool LOS_blocked_sun(
                const float3& start, const float3& end
            );
    };
}

#endif
