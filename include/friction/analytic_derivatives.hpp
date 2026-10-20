// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#pragma once

#include "friction/friction.hpp"

#include <cmath>
#include <stdexcept>

namespace friction::analytic {

struct FirstDerivatives {
    double value{};
    double d_reynolds{};
    double d_relative_roughness{};
};

namespace detail {

[[nodiscard]] inline FirstDerivatives checked(FirstDerivatives result) {
    if (!(result.value > 0.0) || !std::isfinite(result.value) ||
        !std::isfinite(result.d_reynolds) ||
        !std::isfinite(result.d_relative_roughness)) {
        throw std::overflow_error("friction factor or derivative is outside double range");
    }
    return result;
}

[[nodiscard]] inline FirstDerivatives colebrook(double reynolds_number,
                                                 double relative_roughness) {
    const double value = friction::detail_colebrook_formula(
        reynolds_number, relative_roughness);
    const double y = 1.0 / std::sqrt(value);
    const double argument = relative_roughness / 3.7 + 2.51 * y / reynolds_number;
    constexpr double two_over_ln10 = 2.0 / 2.3025850929940456840179914546843642;

    // Implicitly differentiate y + 2 log10(epsilon/3.7 + 2.51 y/Re) = 0,
    // where y = 1/sqrt(f). This is the same Colebrook relation solved by
    // Clamond's iteration, without differentiating the iteration itself.
    const double denominator = argument + two_over_ln10 * 2.51 / reynolds_number;
    return checked({
        value,
        -2.0 * two_over_ln10 * 2.51 * value /
            (reynolds_number * reynolds_number * denominator),
        2.0 * two_over_ln10 * value / (3.7 * y * denominator)});
}

[[nodiscard]] inline FirstDerivatives swamee_jain(double reynolds_number,
                                                   double relative_roughness) {
    const double reynolds_power = std::exp2(0.9 * std::log2(reynolds_number));
    const double reynolds_term = 5.74 / reynolds_power;
    const double argument = relative_roughness / 3.7 + reynolds_term;
    const double log_value = std::log10(argument);
    const double value = 0.25 / (log_value * log_value);
    const double common = -0.5 /
        (2.3025850929940456840179914546843642 * argument * log_value * log_value * log_value);
    const double argument_re = -0.9 * reynolds_term / reynolds_number;
    const double argument_roughness = 1.0 / 3.7;
    return checked({value, common * argument_re, common * argument_roughness});
}

[[nodiscard]] inline FirstDerivatives turbulent(double reynolds_number,
                                                 double relative_roughness,
                                                 Correlation correlation) {
    return correlation == Correlation::colebrook
        ? colebrook(reynolds_number, relative_roughness)
        : swamee_jain(reynolds_number, relative_roughness);
}

[[nodiscard]] inline FirstDerivatives darcy_friction_factor_unchecked(
    double reynolds_number,
    double relative_roughness,
    Correlation correlation) {
    if (reynolds_number < 2000.0) {
        return {64.0 / reynolds_number,
                -64.0 / (reynolds_number * reynolds_number), 0.0};
    }

    const double transition_end = correlation == Correlation::colebrook ? 2300.0 : 5000.0;
    const FirstDerivatives turbulent_value = turbulent(
        reynolds_number, relative_roughness, correlation);
    if (reynolds_number >= transition_end) return turbulent_value;

    const double t = (reynolds_number - 2000.0) / (transition_end - 2000.0);
    const double width = transition_end - 2000.0;
    const double weight = t * t * (3.0 - 2.0 * t);
    const double weight_re = 6.0 * t * (1.0 - t) / width;
    const double laminar = 64.0 / reynolds_number;
    const double laminar_re = -64.0 / (reynolds_number * reynolds_number);
    return {
        (1.0 - weight) * laminar + weight * turbulent_value.value,
        (1.0 - weight) * laminar_re + weight * turbulent_value.d_reynolds +
            weight_re * (turbulent_value.value - laminar),
        weight * turbulent_value.d_relative_roughness};
}

} // namespace detail

// Analytic first derivatives of the all-regime correlation. At Re=2000 and
// the blend endpoint, the cubic smoothstep has zero slope, so this closure is
// C1 when joined to the laminar and turbulent branches.
[[nodiscard]] inline FirstDerivatives darcy_friction_factor(
    double reynolds_number,
    double relative_roughness,
    Correlation correlation = Correlation::colebrook) {
    friction::detail::validate_general_inputs(reynolds_number, relative_roughness);
    if (correlation != Correlation::colebrook && correlation != Correlation::swamee_jain) {
        throw std::invalid_argument("unknown friction correlation");
    }

    return detail::checked(detail::darcy_friction_factor_unchecked(
        reynolds_number, relative_roughness, correlation));
}

} // namespace friction::analytic
