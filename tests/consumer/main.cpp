// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#include <friction/analytic_derivatives.hpp>
#include <friction/batch.hpp>
#include <friction/friction.hpp>

#include <array>
#include <cmath>

int main() {
    const double factor = friction::darcy_friction_factor(700000.0, 0.01);
    const auto gradient = friction::analytic::darcy_friction_factor(700000.0, 0.01);
    constexpr std::array reynolds{700000.0, 1.0e6};
    constexpr std::array roughness{0.01, 0.001};
    std::array<double, 2> batch{};
    friction::batch::darcy_friction_factors(reynolds, roughness, batch);

    return factor > 0.0 && std::isfinite(gradient.d_reynolds) &&
           batch[0] == factor ? 0 : 1;
}
