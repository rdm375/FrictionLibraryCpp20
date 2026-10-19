#include <friction/analytic_derivatives.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void require_close(double actual, double expected, double tolerance, const char* label) {
    const double scale = std::max({1.0e-12, std::abs(actual), std::abs(expected)});
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * scale) {
        std::cerr << "FAIL: " << label << " actual=" << actual
                  << " expected=" << expected << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void check(double re, double roughness, friction::Correlation correlation) {
    constexpr double relative_step = 2.0e-5;
    const auto analytic = friction::analytic::darcy_friction_factor(
        re, roughness, correlation);
    const double hre = relative_step * re;
    // Roughness sensitivity can be strongly attenuated near the lower blend
    // endpoint. Use a larger but still local perturbation to avoid subtracting
    // nearly identical O(1e-2) friction factors at machine precision.
    const double hr = std::min(1.0e-2 * std::max(roughness, 1.0e-4),
                               0.25 * roughness);
    const double re_plus = friction::darcy_friction_factor(re + hre, roughness, correlation);
    const double re_minus = friction::darcy_friction_factor(re - hre, roughness, correlation);
    const double rough_plus = friction::darcy_friction_factor(re, roughness + hr, correlation);
    const double fd_re = (re_plus - re_minus) / (2.0 * hre);
    const double rough_minus = friction::darcy_friction_factor(re, roughness - hr, correlation);
    const double fd_rough = (rough_plus - rough_minus) / (2.0 * hr);

    require_close(analytic.value,
                  friction::darcy_friction_factor(re, roughness, correlation),
                  2.0e-14, "analytic primal");
    require_close(analytic.d_reynolds, fd_re, 3.0e-4, "analytic d(f)/dRe");
    require_close(analytic.d_relative_roughness, fd_rough, 3.0e-4,
                  "analytic d(f)/d(relative roughness)");
}

} // namespace

int main() {
    for (double roughness : {1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2}) {
        for (double re : {1500.0, 2001.0, 2150.0, 2299.0, 2301.0, 1.0e6}) {
            check(re, roughness, friction::Correlation::colebrook);
        }
        for (double re : {1500.0, 2001.0, 3500.0, 4999.0, 5001.0, 1.0e6}) {
            check(re, roughness, friction::Correlation::swamee_jain);
        }

        const auto laminar_edge = friction::analytic::darcy_friction_factor(
            2000.0, roughness, friction::Correlation::colebrook);
        require_close(laminar_edge.d_reynolds, -64.0 / (2000.0 * 2000.0),
                      2.0e-14, "Colebrook blend start derivative");
        const auto colebrook_edge = friction::analytic::darcy_friction_factor(
            2300.0, roughness, friction::Correlation::colebrook);
        const auto colebrook_turbulent = friction::analytic::detail::colebrook(2300.0, roughness);
        require_close(colebrook_edge.d_reynolds, colebrook_turbulent.d_reynolds,
                      2.0e-14, "Colebrook blend end derivative");
        const auto swamee_edge = friction::analytic::darcy_friction_factor(
            5000.0, roughness, friction::Correlation::swamee_jain);
        const auto swamee_turbulent = friction::analytic::detail::swamee_jain(5000.0, roughness);
        require_close(swamee_edge.d_reynolds, swamee_turbulent.d_reynolds,
                      2.0e-14, "Swamee-Jain blend end derivative");
    }
    std::cout << "Analytic derivative checks passed.\n";
}
