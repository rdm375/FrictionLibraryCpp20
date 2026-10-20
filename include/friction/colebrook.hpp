// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#pragma once

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace friction {

struct ColebrookResult {
    double darcy_friction_factor{};
    int iterations{};
};

// Pure Clamond formula kernel. Keep validation and flow-regime decisions outside
// this function: Enzyme differentiates this smooth turbulent-flow kernel.
[[nodiscard]] inline double detail_colebrook_formula(double reynolds_number,
                                                       double relative_roughness) {
    constexpr double ln10 = 2.3025850929940456840179914546843642;
    constexpr double x1_scale = ln10 / 18.574;
    constexpr double x2_offset = -0.7793974884556818028;
    const double x1 = relative_roughness * reynolds_number * x1_scale;
    const double x2 = std::log(reynolds_number) + x2_offset;
    double z = x2 - 0.2;

    for (int iteration = 0; iteration < 2; ++iteration) {
        const double shifted = x1 + z;
        const double epsilon = (z + std::log(shifted) - x2) / (1.0 + shifted);
        const double numerator_scale = 1.0 + shifted + 0.5 * epsilon;
        const double denominator = 1.0 + shifted + epsilon * (1.0 + epsilon / 3.0);
        z -= (numerator_scale / denominator) * epsilon * shifted;
    }
    const double inverse_sqrt_f = (2.0 / ln10) * z;
    return 1.0 / (inverse_sqrt_f * inverse_sqrt_f);
}

// Solve the Darcy-Weisbach Colebrook-White equation using Clamond's shifted
// pi-function iteration (Didier Clamond, arXiv:0810.5564, equations 17 and 20).
// The two-update kernel uses double precision for CPU throughput and for broad
// Enzyme/LLVM compatibility. The raw correlation accepts Re >= 2000 to support
// all-regime transition blending.
[[nodiscard]] inline ColebrookResult colebrook_turbulent(
    double reynolds_number,
    double relative_roughness) {
    if (!std::isfinite(reynolds_number) || reynolds_number < 2000.0) {
        throw std::domain_error("Colebrook solver requires finite Reynolds number >= 2000");
    }
    if (!std::isfinite(relative_roughness) || relative_roughness < 0.0) {
        throw std::domain_error("relative roughness must be finite and non-negative");
    }
    const double result = detail_colebrook_formula(reynolds_number, relative_roughness);
    if (!(result > 0.0) || !std::isfinite(result)) {
        throw std::overflow_error("Colebrook friction factor is outside double range");
    }
    return {result, 2};
}

namespace detail {

inline void validate_iterative_inputs(double reynolds_number, double relative_roughness) {
    if (!std::isfinite(reynolds_number) || reynolds_number < 2000.0) {
        throw std::domain_error("Colebrook solver requires finite Reynolds number >= 2000");
    }
    if (!std::isfinite(relative_roughness) || relative_roughness < 0.0 ||
        relative_roughness >= 3.7) {
        throw std::domain_error("Colebrook solver requires finite relative roughness in [0, 3.7)");
    }
}

[[nodiscard]] inline double colebrook_residual(double y,
                                               double reynolds_number,
                                               double relative_roughness) {
    const double argument = relative_roughness / 3.7 + (2.51 / reynolds_number) * y;
    return y + (2.0 / 2.3025850929940456840179914546843642) * std::log(argument);
}

[[nodiscard]] inline ColebrookResult finish_iterative(double y, int iterations) {
    const double factor = 1.0 / (y * y);
    if (!(factor > 0.0) || !std::isfinite(factor)) {
        throw std::overflow_error("Colebrook friction factor is outside double range");
    }
    return {factor, iterations};
}

} // namespace detail

// Solve the actual implicit Colebrook equation by fixed-point iteration in
// y=1/sqrt(f): y_(n+1)=-2 log10(k/D/3.7 + 2.51 y_n/Re). The reported iteration
// count excludes the initial estimate. Clamond's specialized solver above is
// expected to be faster for repeated evaluations; this implementation is
// useful as a simple independent numerical method.
[[nodiscard]] inline ColebrookResult colebrook_fixed_point(
    double reynolds_number,
    double relative_roughness) {
    detail::validate_iterative_inputs(reynolds_number, relative_roughness);
    constexpr double tolerance = 4.0 * std::numeric_limits<double>::epsilon();
    constexpr int maximum_iterations = 128;
    double y = 7.0;
    for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
        const double argument = relative_roughness / 3.7 + (2.51 / reynolds_number) * y;
        if (!(argument > 0.0) || !std::isfinite(argument)) {
            throw std::runtime_error("Colebrook fixed-point iteration left its real domain");
        }
        const double next = -2.0 * std::log10(argument);
        if (!(next > 0.0) || !std::isfinite(next)) {
            throw std::runtime_error("Colebrook fixed-point iteration failed to find positive root");
        }
        if (std::abs(next - y) <= tolerance * std::max(1.0, std::abs(next))) {
            return detail::finish_iterative(next, iteration);
        }
        y = next;
    }
    throw std::runtime_error("Colebrook fixed-point iteration did not converge");
}

// Solve the actual implicit equation with Newton's method in y=1/sqrt(f).
// The derivative is analytic and strictly positive over the physical domain.
[[nodiscard]] inline ColebrookResult colebrook_newton(
    double reynolds_number,
    double relative_roughness) {
    detail::validate_iterative_inputs(reynolds_number, relative_roughness);
    constexpr double two_over_ln10 = 2.0 / 2.3025850929940456840179914546843642;
    constexpr double tolerance = 4.0 * std::numeric_limits<double>::epsilon();
    constexpr int maximum_iterations = 32;
    double y = 7.0;
    for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
        const double coefficient = 2.51 / reynolds_number;
        const double argument = relative_roughness / 3.7 + coefficient * y;
        if (!(argument > 0.0) || !std::isfinite(argument)) {
            throw std::runtime_error("Colebrook Newton iteration left its real domain");
        }
        const double residual = y + two_over_ln10 * std::log(argument);
        const double derivative = 1.0 + two_over_ln10 * coefficient / argument;
        const double step = residual / derivative;
        const double next = y - step;
        if (!(next > 0.0) || !std::isfinite(next)) {
            throw std::runtime_error("Colebrook Newton iteration failed to find positive root");
        }
        if (std::abs(step) <= tolerance * std::max(1.0, std::abs(next))) {
            return detail::finish_iterative(next, iteration);
        }
        y = next;
    }
    throw std::runtime_error("Colebrook Newton iteration did not converge");
}

} // namespace friction
