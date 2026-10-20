// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#include <friction/batch.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_batch_matches_scalar() {
    const std::array re{1500.0, 2000.0, 2150.0, 2300.0, 5000.0, 1.0e6};
    const std::array roughness{0.0, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-3};
    std::array<double, re.size()> factors{};
    std::array<friction::analytic::FirstDerivatives, re.size()> derivatives{};

    for (const auto correlation : {friction::Correlation::colebrook,
                                   friction::Correlation::swamee_jain}) {
        friction::batch::darcy_friction_factors(re, roughness, factors, correlation);
        friction::batch::darcy_friction_factors_with_derivatives(
            re, roughness, derivatives, correlation);
        for (std::size_t i = 0; i < re.size(); ++i) {
            const double scalar = friction::darcy_friction_factor(re[i], roughness[i], correlation);
            const auto scalar_derivative = friction::analytic::darcy_friction_factor(
                re[i], roughness[i], correlation);
            require(factors[i] == scalar, "batch primal matches scalar primal");
            require(derivatives[i].value == scalar_derivative.value,
                    "batch derivative value matches scalar");
            require(derivatives[i].d_reynolds == scalar_derivative.d_reynolds,
                    "batch Reynolds derivative matches scalar");
            require(derivatives[i].d_relative_roughness ==
                        scalar_derivative.d_relative_roughness,
                    "batch roughness derivative matches scalar");
        }
    }
}

void test_shape_and_input_validation() {
    std::array re{1.0e4, 2.0e4};
    std::array roughness{1.0e-4, 1.0e-3};
    std::array<double, 1> short_output{42.0};
    bool threw = false;
    try { friction::batch::darcy_friction_factors(re, roughness, short_output); }
    catch (const std::invalid_argument&) { threw = true; }
    require(threw, "mismatched span sizes are rejected");

    std::array<double, 2> output{42.0, 43.0};
    re[1] = -1.0;
    threw = false;
    try { friction::batch::darcy_friction_factors(re, roughness, output); }
    catch (const std::domain_error&) { threw = true; }
    require(threw, "invalid input is rejected");
    require(output[0] == 42.0 && output[1] == 43.0,
            "all inputs are validated before writing outputs");
}

void test_overlapping_output_rejected() {
    std::array<double, 4> storage{1.0e5, 2.0e5, 3.0e5, 0.0};
    const std::array roughness{1.0e-4, 1.0e-3, 1.0e-2};
    const auto reynolds = std::span<const double>(storage.data(), 3);
    const auto overlapping_output = std::span<double>(storage.data() + 1, 3);
    const auto before = storage;
    bool threw = false;
    try {
        friction::batch::darcy_friction_factors(
            reynolds, roughness, overlapping_output);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "overlapping input and output spans are rejected");
    require(storage == before, "overlap rejection leaves inputs and output unchanged");
}

} // namespace

int main() {
    test_batch_matches_scalar();
    test_shape_and_input_validation();
    test_overlapping_output_rejected();
    std::cout << "Batch friction checks passed.\n";
}
