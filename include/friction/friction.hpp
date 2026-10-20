// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#pragma once

#include "friction/colebrook.hpp"
#include "friction/swamee_jain.hpp"

#include <cmath>
#include <stdexcept>

namespace friction {

enum class Correlation {
    colebrook,
    swamee_jain
};

namespace detail {

inline void validate_general_inputs(double reynolds_number, double relative_roughness) {
    if (!std::isfinite(reynolds_number) || reynolds_number <= 0.0) {
        throw std::domain_error("Reynolds number must be finite and positive");
    }
    if (!std::isfinite(relative_roughness) || relative_roughness < 0.0) {
        throw std::domain_error("relative roughness must be finite and non-negative");
    }
}

// Pure scalar kernel: no exceptions or validation branches, so Enzyme can
// differentiate the selected piecewise correlation away from regime boundaries.
[[nodiscard]] inline double darcy_factor_unchecked(double reynolds_number,
                                                    double relative_roughness,
                                                    Correlation correlation) {
    if (reynolds_number < 2000.0) {
        return 64.0 / reynolds_number;
    }
    const double transition_end = correlation == Correlation::colebrook ? 2300.0 : 5000.0;
    const double turbulent = correlation == Correlation::colebrook
        ? detail_colebrook_formula(reynolds_number, relative_roughness)
        : detail::swamee_jain_formula(reynolds_number, relative_roughness);
    if (reynolds_number >= transition_end) {
        return turbulent;
    }
    const double t = (reynolds_number - 2000.0) / (transition_end - 2000.0);
    const double weight = t * t * (3.0 - 2.0 * t);
    const double laminar = 64.0 / reynolds_number;
    return (1.0 - weight) * laminar + weight * turbulent;
}

} // namespace detail

// All-regime Darcy friction factor. Transitional flow is joined with a cubic
// smoothstep: Re=2000..2300 for Colebrook and Re=2000..5000 for Swamee-Jain.
[[nodiscard]] inline double darcy_friction_factor(
    double reynolds_number,
    double relative_roughness,
    Correlation correlation = Correlation::colebrook) {
    detail::validate_general_inputs(reynolds_number, relative_roughness);
    if (correlation != Correlation::colebrook && correlation != Correlation::swamee_jain) {
        throw std::invalid_argument("unknown friction correlation");
    }
    const double factor = detail::darcy_factor_unchecked(
        reynolds_number, relative_roughness, correlation);
    if (!(factor > 0.0) || !std::isfinite(factor)) {
        throw std::overflow_error("friction factor is outside double range");
    }
    return factor;
}
} // namespace friction
