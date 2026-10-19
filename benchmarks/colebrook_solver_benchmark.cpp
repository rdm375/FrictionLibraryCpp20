#include <friction/colebrook.hpp>
#include <friction/analytic_derivatives.hpp>
#include <friction/friction.hpp>
#include <friction/swamee_jain.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t sample_count = 8192;
constexpr std::size_t repetitions = 250;
struct Point { double reynolds; double roughness; };

// Direct implicit chain-rule baseline retained for target-specific A/B tests.
friction::analytic::FirstDerivatives colebrook_gradients_implicit_baseline(double re, double rr) {
    const double value = friction::detail_colebrook_formula(re, rr);
    const double y = 1.0 / std::sqrt(value);
    const double argument = rr / 3.7 + 2.51 * y / re;
    constexpr double c = 2.0 / 2.3025850929940456840179914546843642;
    const double residual_y = 1.0 + c * (2.51 / re) / argument;
    const double y_re = (c * 2.51 * y / (re * re * argument)) / residual_y;
    const double y_rr = -(c / (3.7 * argument)) / residual_y;
    const double df_dy = -2.0 / (y * y * y);
    return {value, df_dy * y_re, df_dy * y_rr};
}

double swamee_jain_pow_reference(double re, double rr) {
    const double argument = rr / 3.7 + 5.74 / std::pow(re, 0.9);
    const double log_value = std::log10(argument);
    return 0.25 / (log_value * log_value);
}

friction::analytic::FirstDerivatives swamee_jain_gradients_pow_reference(
    double re, double rr) {
    const double re_power = std::pow(re, 0.9);
    const double re_term = 5.74 / re_power;
    const double argument = rr / 3.7 + re_term;
    const double log_value = std::log10(argument);
    const double value = 0.25 / (log_value * log_value);
    constexpr double ln10 = 2.3025850929940456840179914546843642;
    const double common = -0.5 /
        (ln10 * argument * log_value * log_value * log_value);
    return friction::analytic::detail::checked({
        value, common * (-0.9 * re_term / re), common / 3.7});
}

template <typename Solver>
void run(const char* name, const std::vector<Point>& points, Solver solver) {
    double checksum = 0.0;
    std::size_t iteration_sum = 0;
    int maximum_iterations = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
        for (const Point& point : points) {
            const auto result = solver(point.reynolds, point.roughness);
            checksum += result.darcy_friction_factor;
            iteration_sum += static_cast<std::size_t>(result.iterations);
            if (result.iterations > maximum_iterations) maximum_iterations = result.iterations;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    const std::size_t evaluations = points.size() * repetitions;
    const double rate = static_cast<double>(evaluations) / seconds;
    const double mean_iterations = static_cast<double>(iteration_sum) /
        static_cast<double>(evaluations);
    std::cout << std::left << std::setw(25) << name << std::right << std::fixed
              << std::setprecision(2) << rate / 1.0e6 << " M solves/s"
              << "  mean_iter=" << std::setprecision(2) << mean_iterations
              << "  max_iter=" << maximum_iterations
              << "  checksum=" << std::setprecision(10) << checksum << '\n';
}

template <typename Function>
void run_factor(const char* name, const std::vector<Point>& points, Function function) {
    double checksum = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
        for (const Point& point : points) checksum += function(point.reynolds, point.roughness);
    }
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    const std::size_t evaluations = points.size() * repetitions;
    std::cout << std::left << std::setw(33) << name << std::right << std::fixed
              << std::setprecision(2)
              << static_cast<double>(evaluations) / seconds / 1.0e6 << " M eval/s"
              << "  checksum=" << std::setprecision(10) << checksum << '\n';
}

template <typename Function>
void run_gradients(const char* name, const std::vector<Point>& points, Function function) {
    double checksum = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
        for (const Point& point : points) {
            const auto result = function(point.reynolds, point.roughness);
            checksum += result.value + 1.0e-3 * result.d_reynolds +
                1.0e-6 * result.d_relative_roughness;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    const std::size_t evaluations = points.size() * repetitions;
    std::cout << std::left << std::setw(33) << name << std::right << std::fixed
              << std::setprecision(2)
              << static_cast<double>(evaluations) / seconds / 1.0e6 << " M eval/s"
              << "  checksum=" << std::setprecision(10) << checksum << '\n';
}

} // namespace

int main() {
    std::vector<Point> points;
    points.reserve(sample_count);
    for (std::size_t i = 0; i < sample_count; ++i) {
        points.push_back({5000.0 + static_cast<double>((i * 7919U) % 100000000U),
                          1.0e-6 + static_cast<double>((i * 104729U) % 50000U) * 1.0e-6});
    }
    run("Clamond (existing)", points, [](double re, double rr) {
        return friction::colebrook_turbulent(re, rr);
    });
    run("Fixed point", points, [](double re, double rr) {
        return friction::colebrook_fixed_point(re, rr);
    });
    run("Newton", points, [](double re, double rr) {
        return friction::colebrook_newton(re, rr);
    });
    run_factor("Swamee-Jain primal", points, [](double re, double rr) {
        return friction::swamee_jain_turbulent(re, rr);
    });
    run_factor("Swamee-Jain std::pow baseline", points, [](double re, double rr) {
        return swamee_jain_pow_reference(re, rr);
    });
    run_gradients("Colebrook gradients (production)", points, [](double re, double rr) {
        return friction::analytic::darcy_friction_factor(
            re, rr, friction::Correlation::colebrook);
    });
    run_gradients("Colebrook implicit baseline", points, [](double re, double rr) {
        return colebrook_gradients_implicit_baseline(re, rr);
    });
    run_gradients("Swamee-Jain gradients (production)", points, [](double re, double rr) {
        return friction::analytic::darcy_friction_factor(
            re, rr, friction::Correlation::swamee_jain);
    });
    run_gradients("Swamee-Jain gradients (pow ref)", points, [](double re, double rr) {
        return swamee_jain_gradients_pow_reference(re, rr);
    });
}
