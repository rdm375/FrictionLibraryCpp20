// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#include <friction/colebrook.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_solvers() {
    constexpr std::array reynolds{2000.0, 2300.0, 5000.0, 1.0e4,
                                  1.0e6, 1.0e8, 1.0e13, 1.0e100};
    constexpr std::array roughness{0.0, 1.0e-6, 1.0e-3, 1.0e-2, 1.0e-1};
    for (double re : reynolds) {
        for (double rr : roughness) {
            const auto clamond = friction::colebrook_turbulent(re, rr);
            const auto fixed = friction::colebrook_fixed_point(re, rr);
            const auto newton = friction::colebrook_newton(re, rr);
            for (const auto result : {fixed, newton}) {
                const double relative_error = std::abs(
                    (result.darcy_friction_factor - clamond.darcy_friction_factor) /
                    clamond.darcy_friction_factor);
                require(relative_error < 8.0e-14,
                        "iterative solver agrees with Clamond's solution");
                const double y = 1.0 / std::sqrt(result.darcy_friction_factor);
                const double argument = rr / 3.7 + 2.51 * y / re;
                const double residual = y + 2.0 * std::log10(argument);
                require(std::abs(residual) < 1.0e-13,
                        "iterative solver satisfies the implicit Colebrook equation");
                require(result.iterations > 0, "iteration count is reported");
            }
        }
    }
}

void test_invalid_inputs() {
    for (const auto& [re, rr] : {std::pair{1999.0, 0.0}, std::pair{1.0e5, -1.0e-6},
                                 std::pair{1.0e5, 3.7}}) {
        bool fixed_threw = false;
        bool newton_threw = false;
        try { (void)friction::colebrook_fixed_point(re, rr); }
        catch (const std::domain_error&) { fixed_threw = true; }
        try { (void)friction::colebrook_newton(re, rr); }
        catch (const std::domain_error&) { newton_threw = true; }
        require(fixed_threw && newton_threw, "invalid domain is rejected by both solvers");
    }
}

} // namespace

int main() {
    test_solvers();
    test_invalid_inputs();
    std::cout << "Colebrook solver-method checks passed.\n";
}
