#pragma once

#include <string>

#include "../bsp.h"
#include "state.h"

namespace SilkRAD::Core::Debug {
    void export_lighting_receivers_ply(
        const ::BSP::BSP& bsp,
        const RuntimeState& state,
        const std::string& filename
    );
}
