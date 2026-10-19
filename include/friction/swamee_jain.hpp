#pragma once

#include <cmath>
#include <stdexcept>

namespace friction {

namespace detail {

// Smooth algebraic kernel used by the standalone correlation, the transition
// wrapper, and Enzyme. Input validation is performed by public entry points.
[[nodiscard]] inline double swamee_jain_formula(double reynolds_number,
                                                  double relative_roughness) {
    const double reynolds_power = std::exp2(0.9 * std::log2(reynolds_number));
    const double log_argument = relative_roughness / 3.7 +
        5.74 / reynolds_power;
    const double log_value = std::log10(log_argument);
    return 0.25 / (log_value * log_value);
}

} // namespace detail

// Explicit Swamee-Jain approximation to the Darcy friction factor for
// turbulent flow. Commonly reported use is approximately 5e3 <= Re <= 1e8;
// the formula is defined here only for finite Re >= 5000 and k/D >= 0.
[[nodiscard]] inline double swamee_jain_turbulent(
    double reynolds_number,
    double relative_roughness) {
    if (!std::isfinite(reynolds_number) || reynolds_number < 5000.0) {
        throw std::domain_error("Swamee-Jain requires finite Reynolds number >= 5000");
    }
    if (!std::isfinite(relative_roughness) || relative_roughness < 0.0) {
        throw std::domain_error("relative roughness must be finite and non-negative");
    }
    const double factor = detail::swamee_jain_formula(reynolds_number, relative_roughness);
    if (!(factor > 0.0) || !std::isfinite(factor)) {
        throw std::overflow_error("Swamee-Jain friction factor is outside double range");
    }
    return factor;
}

} // namespace friction
