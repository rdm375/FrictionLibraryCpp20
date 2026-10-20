#include "friction/adaptive_bilinear_tensor_surrogate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
}

int main() {
    friction::surrogate::AdaptiveTensorConfig config;
    friction::surrogate::AdaptiveBilinearColebrookTensor table(config);
    const auto report = table.build();
    require(report.grid_points > 9, "adaptive bilinear build refines its initial grid");
    require(report.largest_unresolved_indicator <= config.relative_tolerance,
            "adaptive bilinear build meets its configured indicator");
    require(table.cell_polynomials().size() ==
                (table.u_nodes().size() - 1) * (table.v_nodes().size() - 1),
            "one bilinear polynomial is stored per tensor cell");

    double max_value_error = 0.0;
    double max_log_re_slope_error = 0.0;
    double max_log_roughness_slope_error = 0.0;
    constexpr int re_points = 37, rr_points = 29;
    for (int i = 0; i < re_points; ++i) {
        const double u = (static_cast<double>(i) + 0.371) / re_points;
        const double re = config.reynolds_min *
            std::pow(config.reynolds_max / config.reynolds_min, u);
        for (int j = 0; j < rr_points; ++j) {
            const double v = (static_cast<double>(j) + 0.613) / rr_points;
            const double rr = config.roughness_scale * std::expm1(
                v * std::log1p(config.relative_roughness_max / config.roughness_scale));
            const auto approximate = table.evaluate(re, rr);
            const auto exact = friction::analytic::darcy_friction_factor(
                re, rr, friction::Correlation::colebrook);
            require(approximate.used_surrogate, "in-domain point uses the bilinear tensor");
            max_value_error = std::max(max_value_error,
                std::abs(approximate.derivatives.value - exact.value) / exact.value);
            max_log_re_slope_error = std::max(max_log_re_slope_error,
                re * std::abs(approximate.derivatives.d_reynolds - exact.d_reynolds) /
                    exact.value);
            max_log_roughness_slope_error = std::max(max_log_roughness_slope_error,
                (config.roughness_scale + rr) * std::abs(
                    approximate.derivatives.d_relative_roughness - exact.d_relative_roughness) /
                    exact.value);
        }
    }

    // Bilinear values are accurate at this sampling density. Its piecewise
    // constant-in-one-coordinate slopes converge more slowly, so record
    // realistic derivative bounds separately from bicubic expectations.
    require(max_value_error < 2.0e-4, "held-out bilinear value error");
    require(max_log_re_slope_error < 0.35, "held-out bilinear Reynolds slope error");
    require(max_log_roughness_slope_error < 0.04, "held-out bilinear roughness slope error");

    const auto laminar = table.evaluate(1500.0, 0.01);
    require(!laminar.used_surrogate, "laminar input uses the direct fallback");
    require(std::abs(laminar.derivatives.value - 64.0 / 1500.0) < 1.0e-15,
            "laminar fallback returns 64/Re");

    std::cout << "Adaptive bilinear tensor checks passed; grid=" << report.grid_points
              << ", max value error=" << max_value_error
              << ", max scaled d/dlog(Re) error=" << max_log_re_slope_error
              << ", max scaled roughness-slope error=" << max_log_roughness_slope_error
              << '\n';
}
