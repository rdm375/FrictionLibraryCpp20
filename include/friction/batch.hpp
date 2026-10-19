#pragma once

#include "friction/analytic_derivatives.hpp"
#include "friction/friction.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>

namespace friction::batch {

namespace detail {

inline void validate_shape(std::span<const double> reynolds_numbers,
                           std::span<const double> relative_roughnesses,
                           std::size_t output_size) {
    if (reynolds_numbers.size() != relative_roughnesses.size() ||
        reynolds_numbers.size() != output_size) {
        throw std::invalid_argument("batch input and output spans must have equal sizes");
    }
}

inline void validate_correlation(Correlation correlation) {
    if (correlation != Correlation::colebrook && correlation != Correlation::swamee_jain) {
        throw std::invalid_argument("unknown friction correlation");
    }
}

inline void validate_inputs(std::span<const double> reynolds_numbers,
                            std::span<const double> relative_roughnesses) {
    for (std::size_t i = 0; i < reynolds_numbers.size(); ++i) {
        friction::detail::validate_general_inputs(
            reynolds_numbers[i], relative_roughnesses[i]);
    }
}

template <class Input, class Output>
[[nodiscard]] inline bool overlaps(std::span<const Input> input,
                                   std::span<Output> output) {
    if (input.empty() || output.empty()) return false;
    const auto* input_begin = reinterpret_cast<const std::byte*>(input.data());
    const auto* output_begin = reinterpret_cast<const std::byte*>(output.data());
    const auto* input_end = input_begin + input.size_bytes();
    const auto* output_end = output_begin + output.size_bytes();
    const std::less<const std::byte*> less;
    return less(input_begin, output_end) && less(output_begin, input_end);
}

inline void validate_no_overlap(std::span<const double> reynolds_numbers,
                               std::span<const double> relative_roughnesses,
                               std::span<double> output) {
    if (overlaps(reynolds_numbers, output) || overlaps(relative_roughnesses, output)) {
        throw std::invalid_argument("batch output must not overlap either input span");
    }
}

inline void validate_no_overlap(
    std::span<const double> reynolds_numbers,
    std::span<const double> relative_roughnesses,
    std::span<analytic::FirstDerivatives> output) {
    if (overlaps(reynolds_numbers, output) || overlaps(relative_roughnesses, output)) {
        throw std::invalid_argument("batch output must not overlap either input span");
    }
}

} // namespace detail

// Evaluate many all-regime friction factors. Inputs are checked before any
// output is written. For throughput, the caller can partition the spans and
// invoke this routine concurrently on disjoint output spans. Output must not
// overlap either input span.
inline void darcy_friction_factors(std::span<const double> reynolds_numbers,
                                   std::span<const double> relative_roughnesses,
                                   std::span<double> output,
                                   Correlation correlation = Correlation::colebrook) {
    detail::validate_shape(reynolds_numbers, relative_roughnesses, output.size());
    detail::validate_correlation(correlation);
    detail::validate_no_overlap(reynolds_numbers, relative_roughnesses, output);
    detail::validate_inputs(reynolds_numbers, relative_roughnesses);

    for (std::size_t i = 0; i < output.size(); ++i) {
        const double value = friction::detail::darcy_factor_unchecked(
            reynolds_numbers[i], relative_roughnesses[i], correlation);
        if (!(value > 0.0) || !std::isfinite(value)) {
            throw std::overflow_error("friction factor is outside double range");
        }
        output[i] = value;
    }
}

// Evaluate all-regime friction factors and their first derivatives with
// respect to Reynolds number and relative roughness. Inputs are checked before
// any output is written. Output must not overlap either input span.
inline void darcy_friction_factors_with_derivatives(
    std::span<const double> reynolds_numbers,
    std::span<const double> relative_roughnesses,
    std::span<analytic::FirstDerivatives> output,
    Correlation correlation = Correlation::colebrook) {
    detail::validate_shape(reynolds_numbers, relative_roughnesses, output.size());
    detail::validate_correlation(correlation);
    detail::validate_no_overlap(reynolds_numbers, relative_roughnesses, output);
    detail::validate_inputs(reynolds_numbers, relative_roughnesses);

    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] = analytic::detail::checked(
            analytic::detail::darcy_friction_factor_unchecked(
                reynolds_numbers[i], relative_roughnesses[i], correlation));
    }
}

} // namespace friction::batch
