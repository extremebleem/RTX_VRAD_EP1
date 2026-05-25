#ifndef __GEOMETRY_RULES_H_
#define __GEOMETRY_RULES_H_

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>

#include "bsp.h"

namespace GeometryRules {
    enum SurfaceRole : uint32_t {
        RTX_ROLE_RECEIVER = 1u << 0,
        RTX_ROLE_OCCLUDER = 1u << 1,
        RTX_ROLE_SKY = 1u << 2,
        RTX_ROLE_TOOL = 1u << 3,
        RTX_ROLE_TRANSLUCENT = 1u << 4,
        RTX_ROLE_DISPLACEMENT = 1u << 5,
        RTX_ROLE_STATIC_PROP = 1u << 6,
    };

    inline bool starts_with(const std::string& text, const char* prefix)
    {
        const size_t prefixLen = std::strlen(prefix);
        return text.size() >= prefixLen
            && text.compare(0, prefixLen, prefix) == 0;
    }

    inline std::string lower_material_name(std::string name)
    {
        std::replace(name.begin(), name.end(), '\\', '/');
        std::transform(
            name.begin(),
            name.end(),
            name.begin(),
            [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            }
        );
        return name;
    }

    inline bool is_service_material(const std::string& materialName)
    {
        const std::string name = lower_material_name(materialName);

        if (!starts_with(name, "tools/")) {
            return false;
        }

        return name.find("trigger") != std::string::npos
            || name.find("clip") != std::string::npos
            || name.find("skip") != std::string::npos
            || name.find("hint") != std::string::npos
            || name.find("nodraw") != std::string::npos
            || name.find("origin") != std::string::npos
            || name.find("areaportal") != std::string::npos
            || name.find("occluder") != std::string::npos
            || name.find("invisible") != std::string::npos;
    }

    inline uint32_t surface_role_from_flags(
        int32_t flags,
        bool displacement,
        const std::string& materialName
    )
    {
        uint32_t role = 0;

        if (displacement) {
            role |= RTX_ROLE_DISPLACEMENT;
        }

        if (flags & BSP::SURF_SKY) {
            return role | RTX_ROLE_SKY;
        }

        if (flags & (BSP::SURF_TRIGGER | BSP::SURF_SKIP | BSP::SURF_HINT)) {
            return role | RTX_ROLE_TOOL;
        }

        if ((flags & BSP::SURF_NODRAW)
            || (flags & BSP::SURF_NOLIGHT)
            || is_service_material(materialName)) {
            return role | RTX_ROLE_TOOL;
        }

        if (flags & BSP::SURF_TRANS) {
            return role | RTX_ROLE_TRANSLUCENT;
        }

        role |= RTX_ROLE_RECEIVER;

        if (!(flags & BSP::SURF_NOSHADOWS)) {
            role |= RTX_ROLE_OCCLUDER;
        }

        return role;
    }

    inline bool brush_contents_block_light(int32_t brushContents)
    {
        const int32_t maskOpaque =
            BSP::CONTENTS_SOLID
            | BSP::CONTENTS_MOVEABLE
            | BSP::CONTENTS_OPAQUE;

        return (brushContents & maskOpaque) != 0;
    }

    inline bool invisible_tool_material_blocks_light(const std::string& materialName)
    {
        const std::string name = lower_material_name(materialName);

        if (!starts_with(name, "tools/")) {
            return false;
        }

        if (name.find("sky") != std::string::npos
            || name.find("trigger") != std::string::npos
            || name.find("skip") != std::string::npos
            || name.find("hint") != std::string::npos
            || name.find("origin") != std::string::npos
            || name.find("areaportal") != std::string::npos) {
            return false;
        }

        return name.find("nodraw") != std::string::npos
            || name.find("invisible") != std::string::npos
            || name.find("black") != std::string::npos
            || name.find("clip") != std::string::npos
            || name.find("occluder") != std::string::npos
            || name.find("blocklight") != std::string::npos;
    }

    inline bool world_brush_side_blocks_light(
        int32_t brushContents,
        int32_t surfaceFlags,
        const std::string& materialName
    )
    {
        if (surfaceFlags & BSP::SURF_SKY) {
            return false;
        }

        if (invisible_tool_material_blocks_light(materialName)) {
            return true;
        }

        return brush_contents_block_light(brushContents);
    }
}

#endif
