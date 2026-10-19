#include <friction/enzyme_derivatives.hpp>
#include <friction/friction.hpp>
#include <friction/analytic_derivatives.hpp>

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t sample_count = 4096;
constexpr std::size_t repetitions = 250;

struct Point { double reynolds; double relative_roughness; };

void report(const char* name, double elapsed_seconds, double checksum, std::size_t count) {
    const double evaluations_per_second = static_cast<double>(count) / elapsed_seconds;
    std::cout << std::left << std::setw(34) << name
              << " " << std::right << std::setw(11) << std::setprecision(2)
              << evaluations_per_second / 1.0e6 << " M eval/s"
              << "  checksum=" << std::setprecision(12) << checksum << '\n';
}

template <typename Function>
void benchmark(const char* name, const std::vector<Point>& points, Function function) {
    double checksum = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
        for (const Point& point : points) checksum += function(point);
    }
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(stop - start).count();
    report(name, elapsed, checksum, points.size() * repetitions);
}

} // namespace

int main() {
    std::vector<Point> points;
    points.reserve(sample_count);
    for (std::size_t i = 0; i < sample_count; ++i) {
        const double re = 5000.0 + static_cast<double>((i * 7919U) % 100000000U);
        const double roughness = 1.0e-6 + static_cast<double>((i * 104729U) % 50000U) * 1.0e-6;
        points.push_back({re, roughness});
    }

    benchmark("Colebrook all-regime primal", points, [](const Point& p) {
        return friction::darcy_friction_factor(p.reynolds, p.relative_roughness);
    });
    benchmark("Swamee-Jain all-regime primal", points, [](const Point& p) {
        return friction::darcy_friction_factor(
            p.reynolds, p.relative_roughness, friction::Correlation::swamee_jain);
    });
    benchmark("Colebrook primal + two gradients", points, [](const Point& p) {
        const auto result = friction::ad::darcy_friction_factor(
            p.reynolds, p.relative_roughness, false);
        return result.value + 1.0e-3 * result.d_reynolds + 1.0e-6 * result.d_relative_roughness;
    });
    benchmark("Swamee-Jain primal + two gradients", points, [](const Point& p) {
        const auto result = friction::ad::darcy_friction_factor(
            p.reynolds, p.relative_roughness, true);
        return result.value + 1.0e-3 * result.d_reynolds + 1.0e-6 * result.d_relative_roughness;
    });
    benchmark("Colebrook primal + analytic gradients", points, [](const Point& p) {
        const auto result = friction::analytic::darcy_friction_factor(
            p.reynolds, p.relative_roughness);
        return result.value + 1.0e-3 * result.d_reynolds + 1.0e-6 * result.d_relative_roughness;
    });
    benchmark("Swamee-Jain primal + analytic gradients", points, [](const Point& p) {
        const auto result = friction::analytic::darcy_friction_factor(
            p.reynolds, p.relative_roughness, friction::Correlation::swamee_jain);
        return result.value + 1.0e-3 * result.d_reynolds + 1.0e-6 * result.d_relative_roughness;
    });
}
