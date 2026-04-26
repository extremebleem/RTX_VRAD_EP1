#include <iostream>
#include <algorithm>
#include <cassert>

#include "raytracer.h"
#include "cudautils.h"


namespace RayTracer {
    /* Implements the M-T ray-triangle intersection algorithm. */
    static __device__ bool intersects(
            const float3& vertex1,
            const float3& vertex2,
            const float3& vertex3,
            const float3& startPos, const float3& endPos
            ) {

        const float EPSILON = 1e-6f;

        float3 diff = endPos - startPos;
        float dist = len(diff);
        float3 dir = diff / dist;

        float3 edge1 = vertex2 - vertex1;
        float3 edge2 = vertex3 - vertex1;

        float3 pVec = cross(dir, edge2);

        float det = dot(edge1, pVec);

        if (det < EPSILON) {
            return false;
        }

        float3 tVec = startPos - vertex1;

        float u = dot(tVec, pVec);
        if (u < 0.0f || u > det) {
            return false;
        }

        float3 qVec = cross(tVec, edge1);

        float v = dot(dir, qVec);

        if (v < 0.0f || u + v > det) {
            return false;
        }

        float t = dot(edge2, qVec) / det;

        return (0.0f < t && t < dist);
    }

    /**
     * Widens a triangle by a small margin, to better deal with ray-tracing
     * edge cases.
     */
    __global__ void map_widen_tris(Triangle* triangles, size_t numTris) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= numTris) {
            return;
        }

        const float EPSILON = 1e-3f;

        Triangle& tri = triangles[index];

        float3& vertex1 = tri.vertices[0];
        float3& vertex2 = tri.vertices[1];
        float3& vertex3 = tri.vertices[2];

        float3 center = (vertex1 + vertex2 + vertex3) / 3.0f;

        vertex1 += normalized(vertex1 - center) * EPSILON;
        vertex2 += normalized(vertex2 - center) * EPSILON;
        vertex3 += normalized(vertex3 - center) * EPSILON;
    }

    static const size_t MAX_DEPTH = 10;

    __global__ void split_nodes(
            Triangle* triangles,
            KDNode* nodes,
            size_t depth
            ) {

        //printf("Split nodes (depth %u)...\n", static_cast<unsigned int>(depth));

        KDNode& node = nodes[threadIdx.x];

        if (depth <= 1 || node.numTris <= 3) {
            //printf(
            //    "Found a leaf! (%u tris)\n",
            //    static_cast<unsigned int>(node.numTris)
            //);
            node.type = KDNodeType::LEAF;
            return;
        }

        float3 nodeSize = node.tmax - node.tmin;
        float3 rightTMin;

        //printf(
        //    "(%u) Node size: (%f, %f, %f)\n",
        //    static_cast<unsigned int>(depth),
        //    nodeSize.x, nodeSize.y, nodeSize.z
        //);

        if (nodeSize.x > nodeSize.y && nodeSize.x > nodeSize.z
                || nodeSize.x == nodeSize.y && nodeSize.x > nodeSize.z
                || nodeSize.x > nodeSize.y && nodeSize.x == nodeSize.z
                ) {

            /* Split along the x-axis. */

            nodeSize.x *= 0.5;
            node.axis = Axis::X;
            node.pos = node.tmin.x + nodeSize.x;
            rightTMin = node.tmin + make_float3(nodeSize.x, 0.0f, 0.0f);

            //printf(
            //    "(%u) Split at x = %f\n",
            //    static_cast<unsigned int>(depth),
            //    node.pos
            //);
        }
        else if (nodeSize.y > nodeSize.x && nodeSize.y > nodeSize.z
                || nodeSize.y == nodeSize.x && nodeSize.y > nodeSize.z
                || nodeSize.y > nodeSize.x && nodeSize.y == nodeSize.z
                ) {

            /* Split along the y-axis. */

            nodeSize.y *= 0.5;
            node.axis = Axis::Y;
            node.pos = node.tmin.y + nodeSize.y;
            rightTMin = node.tmin + make_float3(0.0f, nodeSize.y, 0.0f);

            //printf(
            //    "(%u) Split at y = %f\n",
            //    static_cast<unsigned int>(depth),
            //    node.pos
            //);
        }
        else {
            /* Split along the z-axis. */
            nodeSize.z *= 0.5;
            node.axis = Axis::Z;
            node.pos = node.tmin.z + nodeSize.z;
            rightTMin = node.tmin + make_float3(0.0f, 0.0f, nodeSize.z);
            //printf(
            //    "(%u) Split at z = %f\n",
            //    static_cast<unsigned int>(depth),
            //    node.pos
            //);
        }

        size_t* leftTriIDs;
        CUDA_CHECK_ERROR_DEVICE(
            cudaMalloc(&leftTriIDs, sizeof(size_t) * node.numTris)
        );

        size_t* rightTriIDs;
        CUDA_CHECK_ERROR_DEVICE(
            cudaMalloc(&rightTriIDs, sizeof(size_t) * node.numTris)
        );

        size_t numLeft = 0;
        size_t numRight = 0;

        for (size_t i=0; i<node.numTris; i++) {
            size_t triangleID = node.triangleIDs[i];
            Triangle& tri = triangles[triangleID];

            bool onLeft = false;
            bool onRight = false;

            for (int vertex=0; vertex<3; vertex++) {
                switch (node.axis) {
                    case Axis::X:
                        if (tri.vertices[vertex].x <= node.pos) {
                            onLeft = true;
                        }
                        if (tri.vertices[vertex].x >= node.pos) {
                            onRight = true;
                        }

                        break;

                    case Axis::Y:
                        if (tri.vertices[vertex].y <= node.pos) {
                            onLeft = true;
                        }
                        if (tri.vertices[vertex].y >= node.pos) {
                            onRight = true;
                        }

                        break;

                    case Axis::Z:
                        if (tri.vertices[vertex].z <= node.pos) {
                            onLeft = true;
                        }
                        if (tri.vertices[vertex].z >= node.pos) {
                            onRight = true;
                        }

                        break;
                }

                if (onLeft && onRight) {
                    break;
                }
            }

            if (onLeft) {
                leftTriIDs[numLeft++] = triangleID;
            }

            if (onRight) {
                rightTriIDs[numRight++] = triangleID;
            }
        }

        //printf("cudaFree %p\n", node.triangleIDs);

        //cudaFree(node.triangleIDs);

        CUDA_CHECK_ERROR_DEVICE(
            cudaMalloc(&node.children, sizeof(KDNode) * 2)
        );

        node.children[0].type = KDNodeType::NODE;
        node.children[0].tmin = node.tmin;
        node.children[0].tmax = node.tmin + nodeSize;
        node.children[0].triangleIDs = leftTriIDs;
        node.children[0].numTris = numLeft;

        node.children[1].type = KDNodeType::NODE;
        node.children[1].tmin = rightTMin;
        node.children[1].tmax = node.tmax;
        node.children[1].triangleIDs = rightTriIDs;
        node.children[1].numTris = numRight;

        KERNEL_LAUNCH_DEVICE(
            split_nodes, 1, 2,
            triangles, node.children, depth - 1
        );
    }

    __global__ void cleanup_nodes(KDNode* nodes) {
        KDNode& node = nodes[threadIdx.x];

        CUDA_CHECK_ERROR_DEVICE(cudaFree(node.triangleIDs));

        if (node.type == KDNodeType::NODE) {
            KDNode* children = node.children;

            if (threadIdx.x == 0) {
                CUDA_CHECK_ERROR_DEVICE(cudaFree(nodes));
            }

            KERNEL_LAUNCH_DEVICE(
                cleanup_nodes, 1, 2,
                children
            );
        }
    }

    static __device__ bool intersect_t(
        const float3& vertex1,
        const float3& vertex2,
        const float3& vertex3,
        const float3& startPos,
        const float3& endPos,
        float& outT
    ) {
        const float EPSILON = 1e-6f;

        float3 diff = endPos - startPos;
        float dist = len(diff);
        float3 dir = diff / dist;

        float3 edge1 = vertex2 - vertex1;
        float3 edge2 = vertex3 - vertex1;

        float3 pVec = cross(dir, edge2);
        float det = dot(edge1, pVec);

        if (det < EPSILON)
            return false;

        float3 tVec = startPos - vertex1;

        float u = dot(tVec, pVec);
        if (u < 0.0f || u > det)
            return false;

        float3 qVec = cross(tVec, edge1);

        float v = dot(dir, qVec);
        if (v < 0.0f || u + v > det)
            return false;

        float t = dot(edge2, qVec) / det;

        if (!(0.0f < t && t < dist))
            return false;

        outT = t;
        return true;
    }

    /***********************
     * CUDARayTracer Class *
     ***********************/

    const size_t CUDARayTracer::MAX_LEAVES = 1024;

    CUDARayTracer::CUDARayTracer() :
            m_triangles(nullptr),
            m_triangleIDs(nullptr),
            m_numTriangles(0),
            m_pTreeRoot(nullptr),
            m_tmin(make_float3()),
            m_tmax(make_float3()) {}

    CUDARayTracer::~CUDARayTracer() {
        if (m_triangles != nullptr) {
            cudaFree(m_triangles);
        }

        if (m_pTreeRoot != nullptr) {
            destroy_tree();
        }
    }

    __host__ void CUDARayTracer::build_tree(void) {
        KDNode root;

        root.type = KDNodeType::NODE;

        root.tmin = m_tmin;
        root.tmax = m_tmax;

        root.triangleIDs = m_triangleIDs;
        root.numTris = m_numTriangles;

        CUDA_CHECK_ERROR(cudaMalloc(&m_pTreeRoot, sizeof(KDNode)));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                m_pTreeRoot, &root, sizeof(KDNode),
                cudaMemcpyHostToDevice
            )
        );

        KERNEL_LAUNCH(
            split_nodes, 1, 1,
            m_triangles, m_pTreeRoot, MAX_DEPTH
        );

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    __host__ void CUDARayTracer::destroy_tree(void) {
        KDNode root;

        CUDA_CHECK_ERROR(
            cudaMemcpy(
                &root, m_pTreeRoot, sizeof(KDNode),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK_ERROR(cudaFree(root.triangleIDs));

        if (root.type == KDNodeType::NODE) {
            KDNode* children = root.children;

            CUDA_CHECK_ERROR(cudaFree(m_pTreeRoot));

            KERNEL_LAUNCH(
                cleanup_nodes, 1, 2,
                children
            );
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        }
    }

    __device__ RayHit CUDARayTracer::trace_closest(
        const float3& startPos,
        const float3& endPos
    ) {
        RayHit best;
        best.hit = false;
        best.triangleId = 0;
        best.faceId = 0;
        best.t = 3.402823466e+38F;
        best.position = make_float3();

        float3 dir = normalized(endPos - startPos);
        float3 invDir = { 1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z };

        struct StackEntry {
            KDNode* pNode;
            float3 start;
            float3 end;
        };

        StackEntry stack[1024];
        size_t stackSize = 0;

        stack[stackSize++] = { m_pTreeRoot, startPos, endPos };

        while (stackSize > 0) {
            StackEntry& entry = stack[--stackSize];

            KDNode* pNode = entry.pNode;
            float3 start = entry.start;
            float3 end = entry.end;
            float len = dist(start, end);

            switch (pNode->type) {
            case KDNodeType::LEAF: {
                for (size_t ti = 0; ti < pNode->numTris; ++ti) {
                    size_t triId = pNode->triangleIDs[ti];
                    Triangle& tri = m_triangles[triId];

                    float t;

                    bool hit = intersect_t(
                        tri.vertices[2],
                        tri.vertices[1],
                        tri.vertices[0],
                        startPos,
                        endPos,
                        t
                    );

                    if (hit && t < best.t) {
                        best.hit = true;
                        best.triangleId = triId;
                        best.faceId = tri.faceId;
                        best.t = t;
                        best.position = startPos + dir * t;
                    }
                }

                break;
            }

            case KDNodeType::NODE: {
                KDNode* children = pNode->children;

                float t;
                bool dirPositive;

                switch (pNode->axis) {
                case Axis::X:
                    t = (pNode->pos - start.x) * invDir.x;
                    dirPositive = dir.x >= 0.0f;
                    break;

                case Axis::Y:
                    t = (pNode->pos - start.y) * invDir.y;
                    dirPositive = dir.y >= 0.0f;
                    break;

                case Axis::Z:
                default:
                    t = (pNode->pos - start.z) * invDir.z;
                    dirPositive = dir.z >= 0.0f;
                    break;
                }

                if (t < 0.0f) {
                    stack[stackSize++] = {
                        &children[dirPositive ? 1 : 0],
                        start,
                        end
                    };
                }
                else if (t >= len) {
                    stack[stackSize++] = {
                        &children[dirPositive ? 0 : 1],
                        start,
                        end
                    };
                }
                else {
                    float3 clipPoint = start + t * dir;

                    stack[stackSize++] = {
                        &children[dirPositive ? 0 : 1],
                        start,
                        clipPoint
                    };

                    stack[stackSize++] = {
                        &children[dirPositive ? 1 : 0],
                        clipPoint,
                        end
                    };
                }

                break;
            }
            }
        }

        return best;
    }

    __host__ void CUDARayTracer::add_triangles(
            const std::vector<Triangle>& tris
            ) {

        // This method should only ever be called exactly once.
        assert(m_triangles == nullptr);
        assert(m_numTriangles == 0);

        m_numTriangles = tris.size();

        size_t trianglesSize = sizeof(Triangle) * m_numTriangles;

        CUDA_CHECK_ERROR(cudaMalloc(&m_triangles, trianglesSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                m_triangles, tris.data(), trianglesSize,
                cudaMemcpyHostToDevice
            )
        );

        /*
         * Widen each triangle by a small margin, to better deal with ray-
         * tracing edge cases.
         */
        const size_t BLOCK_WIDTH = 1024;
        size_t numBlocks = div_ceil(m_numTriangles, BLOCK_WIDTH);

        KERNEL_LAUNCH(
            map_widen_tris,
            numBlocks, BLOCK_WIDTH,
            m_triangles, m_numTriangles
        );

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        std::vector<size_t> triangleIDs;
        for (size_t i=0; i<m_numTriangles; i++) {
            triangleIDs.push_back(i);
        }

        size_t triangleIDsSize = sizeof(size_t) * m_numTriangles;

        CUDA_CHECK_ERROR(cudaMalloc(&m_triangleIDs, triangleIDsSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                m_triangleIDs, triangleIDs.data(),
                triangleIDsSize,
                cudaMemcpyHostToDevice
            )
        );

        //std::cout << "cudaMalloc: " << m_triangleIDs << std::endl;

        /* Figure out the bounding box of the KD Tree. */
        for (const Triangle& tri : tris) {
            for (int vertex=0; vertex<3; vertex++) {
                m_tmin.x = std::min(m_tmin.x, tri.vertices[vertex].x);
                m_tmin.y = std::min(m_tmin.y, tri.vertices[vertex].y);
                m_tmin.z = std::min(m_tmin.z, tri.vertices[vertex].z);

                m_tmax.x = std::max(m_tmax.x, tri.vertices[vertex].x);
                m_tmax.y = std::max(m_tmax.y, tri.vertices[vertex].y);
                m_tmax.z = std::max(m_tmax.z, tri.vertices[vertex].z);
            }
        }

        build_tree();
    }

    //__device__ Triangle* CUDARayTracer::get_triangles(void) {
    //    return m_triangles;
    //}

    //__device__ size_t* CUDARayTracer::get_tri_ids(void) {
    //    return m_triangleIDs;
    //}

    //__device__ Triangle& CUDARayTracer::get_tri_indirect(size_t i) {
    //    return m_triangles[m_triangleIDs[i]];
    //}

    __device__ bool CUDARayTracer::LOS_blocked(
            const float3& startPos, const float3& endPos
            ) {

        const float EPSILON = 1e-6f;

        float3 dir = normalized(endPos - startPos);
        float3 invDir = make_float3(
            1.0f / (dir.x + ((dir.x < 0) ? -EPSILON : EPSILON)),
            1.0f / (dir.y + ((dir.y < 0) ? -EPSILON : EPSILON)),
            1.0f / (dir.z + ((dir.z < 0) ? -EPSILON : EPSILON))
        );

        struct StackEntry {
            KDNode* pNode;
            float3 start;
            float3 end;
        };

        StackEntry stack[1024];   // empty ascending stack
        size_t stackSize = 0;

        stack[stackSize++] = {
            m_pTreeRoot,
            startPos,
            endPos,
        };

        while (stackSize > 0) {
            if (stackSize >= 1024) {
                printf("ALERT: Raytracer stack size too big!!!\n");
                return false;
            }

            StackEntry& entry = stack[--stackSize];

            KDNode* pNode = entry.pNode;
            float3 start = entry.start;
            float3 end = entry.end;

            float len = dist(start, end);

            KDNode* children = pNode->children;

            float t;

            switch (pNode->type) {
                case KDNodeType::LEAF:
                    for (size_t ti=0; ti<pNode->numTris; ti++) {
                        Triangle& tri = m_triangles[pNode->triangleIDs[ti]];

                        // The M-T intersection algorithm uses CCW vertex
                        // winding, but Source uses CW winding. So, we need to
                        // pass the vertices in reverse order to get backface
                        // culling to work correctly.
                        bool isLOSBlocked = intersects(
                            tri.vertices[2], tri.vertices[1], tri.vertices[0],
                            startPos, endPos
                        );

                        if (isLOSBlocked) {
                            return true;
                        }
                    }

                    break;

                case KDNodeType::NODE:
                    bool dirPositive;

                    switch (pNode->axis) {
                        case Axis::X:
                            t = (pNode->pos - start.x) * invDir.x;
                            dirPositive = dir.x >= 0.0f;
                            break;

                        case Axis::Y:
                            t = (pNode->pos - start.y) * invDir.y;
                            dirPositive = dir.y >= 0.0f;
                            break;

                        case Axis::Z:
                            t = (pNode->pos - start.z) * invDir.z;
                            dirPositive = dir.z >= 0.0f;
                            break;
                    }

                    if (t < 0.0) {
                        // Plane is "behind" the line start.
                        // Recurse on the right side if dir is positive.
                        // Recurse on the left side if dir is negative.

                        stack[stackSize++] = {
                            &children[dirPositive ? 1 : 0],
                            start,
                            end,
                        };
                    }
                    else if (t >= len) {
                        // Plane is "ahead" of the line end.
                        // Recurse on the left side if dir is positive.
                        // Recurse on the right side if dir is negative.

                        stack[stackSize++] = {
                            &children[dirPositive ? 0 : 1],
                            start,
                            end
                        };
                    }
                    else {
                        // The line segment straddles the plane.
                        // Clip the line and recurse on both sides.

                        float3 clipPoint = start + t * dir;

                        stack[stackSize++] = {
                            &children[dirPositive ? 0 : 1],
                            start,
                            clipPoint,
                        };

                        if (stackSize >= 1024) {
                            printf("ALERT: Stack size too big!!!\n");
                            return false;
                        }

                        stack[stackSize++] = {
                            &children[dirPositive ? 1 : 0],
                            clipPoint,
                            end,
                        };
                    }

                    break;
            }
        }

        return false;
    }

    __device__ bool intersects_shadow_twosided(
        const float3& v0,
        const float3& v1,
        const float3& v2,
        const float3& rayStart,
        const float3& rayEnd,
        float tMin = 0.0005f,
        float tMaxBias = 0.0005f
    ) {
        constexpr float DET_EPS = 1e-8f;
        constexpr float BARY_EPS = 1e-5f;

        float3 ray = rayEnd - rayStart;
        float rayLen = len(ray);

        if (rayLen <= tMin + tMaxBias)
            return false;

        float3 dir = ray / rayLen;

        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;

        float3 p = cross(dir, e2);
        float det = dot(e1, p);

        if (fabsf(det) < DET_EPS)
            return false;

        float invDet = 1.0f / det;

        float3 s = rayStart - v0;
        float u = dot(s, p) * invDet;

        if (u < -BARY_EPS || u > 1.0f + BARY_EPS)
            return false;

        float3 q = cross(s, e1);
        float v = dot(dir, q) * invDet;

        if (v < -BARY_EPS || u + v > 1.0f + BARY_EPS)
            return false;

        float t = dot(e2, q) * invDet;

        return t > tMin && t < rayLen - tMaxBias;
    }

    __device__ bool CUDARayTracer::LOS_blocked_sun(
        const float3& startPos,
        const float3& endPos
    ) {
        constexpr float EPS = 1e-5f;
        constexpr float PLANE_EPS = 1e-4f;
        constexpr int MAX_STACK = 1024;

        float3 ray = endPos - startPos;
        float rayLen = len(ray);

        if (rayLen <= EPS)
            return false;

        float3 dir = ray / rayLen;

        struct StackEntry {
            KDNode* node;
            float tMin;
            float tMax;
        };

        StackEntry stack[MAX_STACK];
        int stackSize = 0;

        if (!m_pTreeRoot)
            return false;

        stack[stackSize++] = { m_pTreeRoot, 0.0f, rayLen };

        while (stackSize > 0) {
            StackEntry e = stack[--stackSize];

            KDNode* pNode = e.node;
            if (!pNode)
                continue;

            if (e.tMax <= e.tMin + EPS)
                continue;

            if (pNode->type == KDNodeType::LEAF) {
                for (size_t ti = 0; ti < pNode->numTris; ++ti) {
                    Triangle& tri = m_triangles[pNode->triangleIDs[ti]];

                    if (tri.flags & BSP::SURF_SKY)
                        continue;

                    // Осторожно: SURF_TRANS часто бывает у alpha-test текстур.
                    // Если такие текстуры должны блокировать солнце — не skip-ать их вслепую.
                    if ((tri.flags & BSP::SURF_TRANS) && !(tri.flags & BSP::SURF_NODRAW))
                        continue;

                    if (intersects_shadow_twosided(
                        tri.vertices[0],
                        tri.vertices[1],
                        tri.vertices[2],
                        startPos,
                        endPos,
                        0.001f,
                        0.001f
                    )) {
                        return true;
                    }
                }

                continue;
            }

            KDNode* children = pNode->children;

            float originAxis;
            float dirAxis;

            switch (pNode->axis) {
            case Axis::X:
                originAxis = startPos.x;
                dirAxis = dir.x;
                break;
            case Axis::Y:
                originAxis = startPos.y;
                dirAxis = dir.y;
                break;
            case Axis::Z:
                originAxis = startPos.z;
                dirAxis = dir.z;
                break;
            default:
                continue;
            }

            int leftChild = 0;
            int rightChild = 1;

            // Луч почти параллелен split-plane.
            if (fabsf(dirAxis) < EPS) {
                float distToPlane = originAxis - pNode->pos;

                if (distToPlane < -PLANE_EPS) {
                    stack[stackSize++] = { &children[leftChild], e.tMin, e.tMax };
                }
                else if (distToPlane > PLANE_EPS) {
                    stack[stackSize++] = { &children[rightChild], e.tMin, e.tMax };
                }
                else {
                    // На самой плоскости — надо проверить оба child-а.
                    if (stackSize + 2 >= MAX_STACK)
                        return true; // conservative: лучше тень, чем leak

                    stack[stackSize++] = { &children[rightChild], e.tMin, e.tMax };
                    stack[stackSize++] = { &children[leftChild],  e.tMin, e.tMax };
                }

                continue;
            }

            float splitT = (pNode->pos - originAxis) / dirAxis;

            int nearChild = dirAxis >= 0.0f ? leftChild : rightChild;
            int farChild = dirAxis >= 0.0f ? rightChild : leftChild;

            if (splitT <= e.tMin - PLANE_EPS) {
                stack[stackSize++] = { &children[farChild], e.tMin, e.tMax };
            }
            else if (splitT >= e.tMax + PLANE_EPS) {
                stack[stackSize++] = { &children[nearChild], e.tMin, e.tMax };
            }
            else {
                if (stackSize + 2 >= MAX_STACK)
                    return true; // conservative

                float t0 = fmaxf(e.tMin, splitT);
                float t1 = fminf(e.tMax, splitT);

                // Far first, near second.
                stack[stackSize++] = { &children[farChild],  t0, e.tMax };
                stack[stackSize++] = { &children[nearChild], e.tMin, t1 };
            }
        }

        return false;
    }
}
