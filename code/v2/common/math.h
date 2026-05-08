#pragma once

#include <cmath>

#include "types.h"

namespace SilkRAD::V2::Common {
    inline Vec2f make_vec2(float x, float y)
    {
        Vec2f v;
        v.x = x;
        v.y = y;
        return v;
    }

    inline Vec3f make_vec3(float x, float y, float z)
    {
        Vec3f v;
        v.x = x;
        v.y = y;
        v.z = z;
        return v;
    }

    inline bool is_finite_scalar(float value)
    {
        return std::isfinite(value) != 0;
    }

    inline Vec3f add(Vec3f a, Vec3f b)
    {
        return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline Vec3f sub(Vec3f a, Vec3f b)
    {
        return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    inline Vec3f scale(Vec3f v, float scalar)
    {
        return make_vec3(v.x * scalar, v.y * scalar, v.z * scalar);
    }

    inline float dot(Vec3f a, Vec3f b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline Vec3f cross(Vec3f a, Vec3f b)
    {
        return make_vec3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    inline float length(Vec3f v)
    {
        return std::sqrt(dot(v, v));
    }

    inline Vec3f normalized(Vec3f v)
    {
        const float vlen = length(v);
        if (vlen <= 1e-20f) {
            return make_vec3(0.0f, 0.0f, 0.0f);
        }

        return scale(v, 1.0f / vlen);
    }

    inline float cross(Vec2f a, Vec2f b)
    {
        return a.x * b.y - a.y * b.x;
    }

    inline bool is_finite(Vec3f v)
    {
        return is_finite_scalar(v.x)
            && is_finite_scalar(v.y)
            && is_finite_scalar(v.z);
    }
}
