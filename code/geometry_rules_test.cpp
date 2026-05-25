#include <cstdlib>
#include <iostream>

#include "geometry_rules.h"

static void expect_true(bool value, const char* message)
{
    if (!value) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}

static void expect_false(bool value, const char* message)
{
    expect_true(!value, message);
}

int main()
{
    expect_true(
        GeometryRules::world_brush_side_blocks_light(
            BSP::CONTENTS_SOLID,
            BSP::SURF_TRANS,
            "brick/wall"
        ),
        "Opaque brush side with SURF_TRANS must block light like VRAD."
    );

    expect_true(
        GeometryRules::world_brush_side_blocks_light(
            BSP::CONTENTS_SOLID,
            BSP::SURF_NODRAW,
            "tools/toolsnodraw"
        ),
        "Opaque brush side with SURF_NODRAW must still block light."
    );

    expect_true(
        GeometryRules::world_brush_side_blocks_light(
            BSP::CONTENTS_SOLID,
            BSP::SURF_NOLIGHT,
            "tools/toolsnodraw"
        ),
        "Opaque brush side with SURF_NOLIGHT must still block light."
    );

    expect_false(
        GeometryRules::world_brush_side_blocks_light(
            BSP::CONTENTS_SOLID,
            BSP::SURF_SKY,
            "tools/toolsskybox"
        ),
        "Sky brush side should be handled as sky, not as an opaque occluder."
    );

    expect_false(
        GeometryRules::world_brush_side_blocks_light(
            BSP::CONTENTS_GRATE,
            0,
            "metal/grate"
        ),
        "Non-opaque brush contents must not block light."
    );

    expect_true(
        GeometryRules::world_brush_side_blocks_light(
            BSP::CONTENTS_GRATE,
            BSP::SURF_NODRAW,
            "tools/toolsnodraw"
        ),
        "Invisible tool materials must block light even without opaque contents."
    );

    return 0;
}
