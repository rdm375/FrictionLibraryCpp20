// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#include <friction/batch.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

constexpr std::size_t sample_count = 1U << 20;
constexpr unsigned repetitions = 3;

void report(const char* name, double seconds, std::size_t count, double checksum) {
    std::cout << std::left << std::setw(39) << name << std::right << std::setw(10)
              << std::setprecision(3) << (static_cast<double>(count) / seconds / 1.0e6)
              << " M eval/s  checksum=" << std::setprecision(12) << checksum << '\n';
}

template <typename Output, typename Function>
void run(const char* name,
         const std::vector<double>& reynolds,
         const std::vector<double>& roughness,
         std::vector<Output>& output,
         unsigned worker_count,
         Function function) {
    const auto start = std::chrono::steady_clock::now();
    if (worker_count <= 1) {
        for (unsigned repeat = 0; repeat < repetitions; ++repeat) {
            function(std::span<const double>(reynolds), std::span<const double>(roughness),
                     std::span<Output>(output));
        }
    } else {
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (unsigned worker = 0; worker < worker_count; ++worker) {
            const std::size_t begin = output.size() * worker / worker_count;
            const std::size_t end = output.size() * (worker + 1U) / worker_count;
            workers.emplace_back([&, begin, end] {
                for (unsigned repeat = 0; repeat < repetitions; ++repeat) {
                    function(std::span<const double>(reynolds).subspan(begin, end - begin),
                             std::span<const double>(roughness).subspan(begin, end - begin),
                             std::span<Output>(output).subspan(begin, end - begin));
                }
            });
        }
        for (auto& worker : workers) worker.join();
    }
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    double checksum = 0.0;
    if constexpr (std::is_same_v<Output, double>) {
        for (double value : output) checksum += value;
    } else {
        for (const auto& value : output) checksum += value.value +
            1.0e-3 * value.d_reynolds + 1.0e-6 * value.d_relative_roughness;
    }
    report(name, seconds, output.size() * repetitions, checksum);
}

} // namespace

int main() {
    std::vector<double> reynolds(sample_count);
    std::vector<double> roughness(sample_count);
    for (std::size_t i = 0; i < sample_count; ++i) {
        reynolds[i] = 5000.0 + static_cast<double>((i * 7919U) % 100000000U);
        roughness[i] = 1.0e-6 + static_cast<double>((i * 104729U) % 50000U) * 1.0e-6;
    }

    std::cout << "Batch size: " << sample_count << " independent turbulent-flow inputs, "
              << repetitions << " timed passes per worker count\n";
    for (unsigned workers : {1U, 2U, 4U, 8U}) {
        std::vector<double> factors(sample_count);
        std::vector<friction::analytic::FirstDerivatives> gradients(sample_count);
        const std::string suffix = " (" + std::to_string(workers) + " thread" +
            (workers == 1 ? ")" : "s)");
        const auto correlation = friction::Correlation::colebrook;
        run(("Colebrook primal" + suffix).c_str(), reynolds, roughness, factors,
            workers, [correlation](auto re, auto rr, auto out) {
                friction::batch::darcy_friction_factors(re, rr, out, correlation);
            });
        run(("Colebrook + analytic gradients" + suffix).c_str(), reynolds, roughness,
            gradients, workers, [correlation](auto re, auto rr, auto out) {
                friction::batch::darcy_friction_factors_with_derivatives(
                    re, rr, out, correlation);
            });
        const auto sj = friction::Correlation::swamee_jain;
        run(("Swamee-Jain primal" + suffix).c_str(), reynolds, roughness, factors,
            workers, [sj](auto re, auto rr, auto out) {
                friction::batch::darcy_friction_factors(re, rr, out, sj);
            });
        run(("Swamee-Jain + analytic gradients" + suffix).c_str(), reynolds, roughness,
            gradients, workers, [sj](auto re, auto rr, auto out) {
                friction::batch::darcy_friction_factors_with_derivatives(
                    re, rr, out, sj);
            });
    }
}
