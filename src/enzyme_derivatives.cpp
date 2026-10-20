// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#include <friction/enzyme_derivatives.hpp>
#include <friction/friction.hpp>

extern "C" {
int enzyme_dup;
void __enzyme_autodiff(void*, ...);
}

namespace {

double colebrook_primal(double* x) {
    return friction::detail_colebrook_formula(x[0], x[1]);
}

double swamee_jain_primal(double* x) {
    return friction::detail::swamee_jain_formula(x[0], x[1]);
}

double all_regime_colebrook_primal(double* x) {
    return friction::detail::darcy_factor_unchecked(
        x[0], x[1], friction::Correlation::colebrook);
}

double all_regime_swamee_jain_primal(double* x) {
    return friction::detail::darcy_factor_unchecked(
        x[0], x[1], friction::Correlation::swamee_jain);
}

template <double (*Primal)(double*)>
friction::ad::FirstDerivatives reverse_gradient(double reynolds_number,
                                                 double relative_roughness,
                                                 double value) {
    double x[2]{reynolds_number, relative_roughness};
    double dx[2]{0.0, 0.0};
    __enzyme_autodiff((void*)Primal, enzyme_dup, x, dx);
    return {value, dx[0], dx[1]};
}

} // namespace

namespace friction::ad {

FirstDerivatives colebrook_turbulent(double reynolds_number, double relative_roughness) {
    const double value = friction::colebrook_turbulent(
        reynolds_number, relative_roughness).darcy_friction_factor;
    return reverse_gradient<colebrook_primal>(reynolds_number, relative_roughness, value);
}

FirstDerivatives swamee_jain_turbulent(double reynolds_number, double relative_roughness) {
    const double value = friction::swamee_jain_turbulent(reynolds_number, relative_roughness);
    return reverse_gradient<swamee_jain_primal>(reynolds_number, relative_roughness, value);
}

FirstDerivatives darcy_friction_factor(double reynolds_number,
                                        double relative_roughness,
                                        bool use_swamee_jain) {
    const auto correlation = use_swamee_jain
        ? friction::Correlation::swamee_jain
        : friction::Correlation::colebrook;
    const double value = friction::darcy_friction_factor(
        reynolds_number, relative_roughness, correlation);
    if (use_swamee_jain) {
        return reverse_gradient<all_regime_swamee_jain_primal>(
            reynolds_number, relative_roughness, value);
    }
    return reverse_gradient<all_regime_colebrook_primal>(
        reynolds_number, relative_roughness, value);
}

} // namespace friction::ad
