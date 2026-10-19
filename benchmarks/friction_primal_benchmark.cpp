#include <friction/friction.hpp>

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
constexpr std::size_t sample_count = 8192;
constexpr std::size_t repetitions = 250;
struct Point { double re; double roughness; };

template <typename Function>
void run(const char* name, const std::vector<Point>& points, Function function) {
    double checksum = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
        for (const Point& point : points) checksum += function(point);
    }
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    const double rate = static_cast<double>(points.size() * repetitions) / seconds;
    std::cout << std::left << std::setw(32) << name << std::right << std::fixed
              << std::setprecision(2) << rate / 1.0e6 << " M eval/s"
              << " checksum=" << std::setprecision(10) << checksum << '\n';
}
} // namespace

int main() {
    std::vector<Point> points;
    points.reserve(sample_count);
    for (std::size_t i = 0; i < sample_count; ++i) {
        points.push_back({5000.0 + static_cast<double>((i * 7919U) % 100000000U),
                          1.0e-6 + static_cast<double>((i * 104729U) % 50000U) * 1.0e-6});
    }
    run("Colebrook all-regime", points, [](Point p) {
        return friction::darcy_friction_factor(p.re, p.roughness);
    });
    run("Swamee-Jain all-regime", points, [](Point p) {
        return friction::darcy_friction_factor(
            p.re, p.roughness, friction::Correlation::swamee_jain);
    });
    run("Colebrook raw", points, [](Point p) {
        return friction::colebrook_turbulent(p.re, p.roughness).darcy_friction_factor;
    });
    run("Swamee-Jain raw", points, [](Point p) {
        return friction::swamee_jain_turbulent(p.re, p.roughness);
    });
}
