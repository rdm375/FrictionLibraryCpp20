// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#include <friction/enzyme_derivatives.hpp>
#include <friction/friction.hpp>
#include <friction/analytic_derivatives.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void require_close(double actual, double expected, double relative_tolerance, const char* message) {
    const double scale = std::max({1.0e-12, std::abs(actual), std::abs(expected)});
    if (std::abs(actual - expected) > relative_tolerance * scale) {
        std::cerr << "FAIL: " << message << " actual=" << actual << " expected=" << expected << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void validate(double re, double roughness, bool swamee_jain) {
    constexpr double relative_step = 2.0e-5;
    const auto grad = friction::ad::darcy_friction_factor(re, roughness, swamee_jain);
    const auto analytic = friction::analytic::darcy_friction_factor(
        re, roughness, swamee_jain ? friction::Correlation::swamee_jain
                                   : friction::Correlation::colebrook);
    const double hre = relative_step * re;
    const double hrough = relative_step * std::max(roughness, 1.0e-4);
    const auto model = swamee_jain ? friction::Correlation::swamee_jain
                                   : friction::Correlation::colebrook;
    const double d_re_fd = (friction::darcy_friction_factor(re + hre, roughness, model) -
                            friction::darcy_friction_factor(re - hre, roughness, model)) / (2.0 * hre);
    const double d_rough_fd = roughness >= hrough
        ? (friction::darcy_friction_factor(re, roughness + hrough, model) -
           friction::darcy_friction_factor(re, roughness - hrough, model)) / (2.0 * hrough)
        : (friction::darcy_friction_factor(re, roughness + hrough, model) - grad.value) / hrough;
    require_close(grad.value, friction::darcy_friction_factor(re, roughness, model), 2e-14,
                  "AD primal matches friction API");
    require_close(grad.d_reynolds, analytic.d_reynolds, 2e-9,
                  "Enzyme d(f)/dRe matches analytic derivative");
    require_close(grad.d_relative_roughness, analytic.d_relative_roughness, 2e-9,
                  "Enzyme d(f)/d(k/D) matches analytic derivative");
    require_close(grad.d_reynolds, d_re_fd, 3e-5, "Enzyme d(f)/dRe matches central difference");
    require_close(grad.d_relative_roughness, d_rough_fd, 3e-5,
                  "Enzyme d(f)/d(k/D) matches central difference");
}

} // namespace

int main() {
    for (double roughness : {0.0, 1.0e-6, 1.0e-3, 1.0e-2}) {
        validate(3.0e3, roughness, false);
        validate(1.0e6, roughness, false);
        validate(1.0e6, roughness, true);
    }
    validate(1500.0, 1.0e-3, false);
    validate(2150.0, 1.0e-3, false);
    validate(2150.0, 1.0e-3, true);
    std::cout << "Enzyme derivative checks passed.\n";
}
