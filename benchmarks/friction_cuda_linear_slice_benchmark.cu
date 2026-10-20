// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#include "friction/analytic_derivatives.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

// A fixed-roughness, one-dimensional piecewise-linear function of log(Re).
// Knots are selected adaptively from endpoint Taylor disagreement and direct
// midpoint checks. The GPU binary-searches the resulting nonuniform knots.
constexpr double re_min = 2300.0;
constexpr double re_max = 1.0e8;
constexpr double fixed_roughness = 6.971775e-4;
constexpr double log_re_min = 7.740664401917241;
constexpr double log_re_max = 18.420680743952367;
constexpr double log_re_span = log_re_max - log_re_min;
constexpr int maximum_refinement_depth = 24;
constexpr double value_rel_tol = 2.0e-7;
constexpr double dre_rel_tol = 2.0e-3;
constexpr double drr_rel_tol = 2.0e-6;
constexpr double value_abs_tol = 2.0e-12;
constexpr double dre_abs_tol = 2.0e-12;
constexpr double drr_abs_tol = 2.0e-10;

struct Result {
    double value{};
    double d_reynolds{};
    double d_roughness{};
};

struct Segment {
    double u0;
    double inverse_du;
    double f0;
    double delta_f;
    double dr0;
    double delta_dr;
};

struct Sample {
    double u;
    double re;
    Result value;
};

struct BuildStats {
    std::size_t sampled_midpoints{};
    std::size_t unresolved_intervals{};
    int deepest_refinement{};
};

struct SliceView {
    const Segment* segments;
    std::size_t segment_count;
};

__device__ __forceinline__ Result colebrook_gradient(double re, double rr) {
    constexpr double ln10 = 2.3025850929940456840179914546843642;
    constexpr double x1_scale = ln10 / 18.574;
    constexpr double x2_offset = -0.7793974884556818028;
    const double x1 = rr * re * x1_scale;
    const double x2 = log(re) + x2_offset;
    double z = x2 - 0.2;
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const double shifted = x1 + z;
        const double epsilon = (z + log(shifted) - x2) / (1.0 + shifted);
        const double numerator = 1.0 + shifted + 0.5 * epsilon;
        const double denominator = 1.0 + shifted + epsilon * (1.0 + epsilon / 3.0);
        z -= (numerator / denominator) * epsilon * shifted;
    }
    const double y = (2.0 / ln10) * z;
    const double f = 1.0 / (y * y);
    const double root = 1.0 / sqrt(f);
    const double arg = rr / 3.7 + 2.51 * root / re;
    const double residual_y = 1.0 + (2.0 / ln10) * (2.51 / re) / arg;
    const double y_re = ((2.0 / ln10) * 2.51 * root /
                         (re * re * arg)) / residual_y;
    const double y_rr = (-(2.0 / ln10) / (3.7 * arg)) / residual_y;
    const double df_dy = -2.0 / (root * root * root);
    const Result turbulent{f, df_dy * y_re, df_dy * y_rr};
    if (re < 2000.0) return {64.0 / re, -64.0 / (re * re), 0.0};
    if (re >= 2300.0) return turbulent;
    constexpr double width = 300.0;
    const double t = (re - 2000.0) / width;
    const double w = t * t * (3.0 - 2.0 * t);
    const double w_re = 6.0 * t * (1.0 - t) / width;
    const double laminar = 64.0 / re;
    const double laminar_re = -64.0 / (re * re);
    return {(1.0 - w) * laminar + w * turbulent.value,
            (1.0 - w) * laminar_re + w * turbulent.d_reynolds +
                w_re * (turbulent.value - laminar),
            w * turbulent.d_roughness};
}

__device__ __forceinline__ Result evaluate_slice(double re, double rr,
                                                  SliceView view) {
    if (re < re_min) return colebrook_gradient(re, rr);
    const double u = log(re);
    // Adaptive knots are nonuniform. Locate the cell with a bounded binary
    // search, then evaluate its linear polynomial and derivative channels.
    std::size_t lo = 0;
    std::size_t hi = view.segment_count - 1;
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo + 1) / 2;
        if (view.segments[mid].u0 <= u) lo = mid;
        else hi = mid - 1;
    }
    const Segment s = view.segments[lo];
    const double x = fmin(1.0, fmax(0.0, (u - s.u0) * s.inverse_du));
    const double value = s.f0 + s.delta_f * x;
    const double d_re = s.delta_f * s.inverse_du / re;
    const double d_rr = s.dr0 + s.delta_dr * x;
    return {value, d_re, d_rr};
}

Sample sample_at(double u) {
    const double re = std::exp(u);
    const auto d = friction::analytic::darcy_friction_factor(
        re, fixed_roughness, friction::Correlation::colebrook);
    return {u, re, {d.value, d.d_reynolds, d.d_relative_roughness}};
}

bool within_tolerance(double error, double absolute_tolerance,
                      double relative_tolerance, double reference) {
    return std::isfinite(error) && error <= absolute_tolerance +
        relative_tolerance * std::abs(reference);
}

bool interval_acceptable(const Sample& a, const Sample& b, const Sample& midpoint) {
    const double h = b.u - a.u;
    const double secant = (b.value.value - a.value.value) / h;
    const double linear_midpoint = 0.5 * (a.value.value + b.value.value);
    const double exact_midpoint = midpoint.value.value;
    const double left_taylor_midpoint = a.value.value +
        0.5 * h * a.re * a.value.d_reynolds;
    const double right_taylor_midpoint = b.value.value -
        0.5 * h * b.re * b.value.d_reynolds;

    // Endpoint Taylor predictions disagree with the linear interpolant when
    // curvature is significant. The direct midpoint sample validates that
    // indicator and the interpolated value.
    if (!within_tolerance(std::abs(left_taylor_midpoint - linear_midpoint),
                          value_abs_tol, value_rel_tol, exact_midpoint) ||
        !within_tolerance(std::abs(right_taylor_midpoint - linear_midpoint),
                          value_abs_tol, value_rel_tol, exact_midpoint) ||
        !within_tolerance(std::abs(exact_midpoint - linear_midpoint),
                          value_abs_tol, value_rel_tol, exact_midpoint))
        return false;

    // In u=log(Re), the linear interpolant's derivative is its secant slope.
    // Check the resulting df/dRe at each endpoint and at the midpoint.
    const double d_re_at_a = secant / a.re;
    const double d_re_at_b = secant / b.re;
    const double d_re_at_midpoint = secant / midpoint.re;
    if (!within_tolerance(std::abs(d_re_at_a - a.value.d_reynolds),
                          dre_abs_tol, dre_rel_tol, a.value.d_reynolds) ||
        !within_tolerance(std::abs(d_re_at_b - b.value.d_reynolds),
                          dre_abs_tol, dre_rel_tol, b.value.d_reynolds) ||
        !within_tolerance(std::abs(d_re_at_midpoint - midpoint.value.d_reynolds),
                          dre_abs_tol, dre_rel_tol, midpoint.value.d_reynolds))
        return false;

    const double roughness_midpoint = 0.5 *
        (a.value.d_roughness + b.value.d_roughness);
    return within_tolerance(std::abs(midpoint.value.d_roughness - roughness_midpoint),
                            drr_abs_tol, drr_rel_tol,
                            midpoint.value.d_roughness);
}

void refine_interval(const Sample& a, const Sample& b, int depth,
                     std::vector<Sample>& knots, BuildStats& stats) {
    stats.deepest_refinement = std::max(stats.deepest_refinement, depth);
    const Sample midpoint = sample_at(0.5 * (a.u + b.u));
    ++stats.sampled_midpoints;
    if (interval_acceptable(a, b, midpoint)) {
        knots.push_back(b);
        return;
    }
    if (depth >= maximum_refinement_depth) {
        ++stats.unresolved_intervals;
        knots.push_back(b);
        return;
    }
    refine_interval(a, midpoint, depth + 1, knots, stats);
    refine_interval(midpoint, b, depth + 1, knots, stats);
}

std::vector<Segment> build_slice(BuildStats& stats) {
    const Sample left = sample_at(log_re_min);
    const Sample right = sample_at(log_re_max);
    std::vector<Sample> knots;
    knots.reserve(4096);
    knots.push_back(left);
    refine_interval(left, right, 0, knots, stats);

    std::vector<Segment> segments;
    segments.reserve(knots.size() - 1);
    for (std::size_t i = 0; i + 1 < knots.size(); ++i) {
        const Sample& a = knots[i];
        const Sample& b = knots[i + 1];
        const double inverse_du = 1.0 / (b.u - a.u);
        segments.push_back({a.u, inverse_du, a.value.value,
                            b.value.value - a.value.value,
                            a.value.d_roughness,
                            b.value.d_roughness - a.value.d_roughness});
    }
    return segments;
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

Result host_direct(double re) {
    const auto x = friction::analytic::darcy_friction_factor(
        re, fixed_roughness, friction::Correlation::colebrook);
    return {x.value, x.d_reynolds, x.d_relative_roughness};
}

Result host_slice(double re, const std::vector<Segment>& segments) {
    if (re < re_min) return host_direct(re);
    const double u = std::log(re);
    const auto it = std::upper_bound(
        segments.begin(), segments.end(), u,
        [](double value, const Segment& segment) { return value < segment.u0; });
    const std::size_t i = it == segments.begin()
        ? 0 : static_cast<std::size_t>(it - segments.begin() - 1);
    const Segment& s = segments[i];
    const double x = std::clamp((u - s.u0) * s.inverse_du, 0.0, 1.0);
    return {s.f0 + s.delta_f * x,
            s.delta_f * s.inverse_du / re,
            s.dr0 + s.delta_dr * x};
}

__global__ void direct_kernel(const double* re, Result* out, std::size_t count,
                              double rr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = colebrook_gradient(re[i], rr);
}

__global__ void slice_kernel(const double* re, Result* out, std::size_t count,
                             double rr, SliceView view) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = evaluate_slice(re[i], rr, view);
}

template<class F>
double seconds(F&& f, int repetitions) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

double checksum(const std::vector<Result>& values) {
    double sum = 0.0;
    for (const auto& r : values)
        sum += r.value + 1.0e-3 * r.d_reynolds + 1.0e-6 * r.d_roughness;
    return sum;
}

} // namespace

int main() {
    constexpr std::size_t count = 1U << 20;
    constexpr int repetitions = 20;
    constexpr int threads = 256;

    int device = 0;
    cuda_check(cudaGetDevice(&device), "get CUDA device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "read CUDA device");
    std::cout << "GPU: " << properties.name << " (compute " << properties.major
              << '.' << properties.minor << ")\n"
              << "Batch: " << count << " Reynolds inputs; " << repetitions
              << " passes; fixed relative roughness=" << std::scientific
              << fixed_roughness << '\n'
              << "Slice: adaptive piecewise-linear function in log(Re); "
              << "Taylor disagreement + midpoint refinement\n";

    BuildStats build_stats{};
    const auto build_start = std::chrono::steady_clock::now();
    const std::vector<Segment> segments = build_slice(build_stats);
    const double build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - build_start).count();
    std::cout << "CPU sample + slice construction=" << std::fixed << std::setprecision(3)
              << build_ms << " ms; knots=" << (segments.size() + 1)
              << "; sampled midpoints=" << build_stats.sampled_midpoints
              << "; max depth=" << build_stats.deepest_refinement
              << "; unresolved=" << build_stats.unresolved_intervals
              << "; coefficient storage="
              << segments.size() * sizeof(Segment) / 1024.0 << " KiB\n";

    std::vector<double> re(count);
    std::vector<Result> reference(count), output(count);
    std::mt19937_64 rng(20261020);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (std::size_t i = 0; i < count; ++i) {
        re[i] = re_min * std::pow(re_max / re_min, unit(rng));
        reference[i] = host_direct(re[i]);
    }
    const double cpu_direct_s = seconds([&] {
        for (std::size_t i = 0; i < count; ++i) output[i] = host_direct(re[i]);
    }, repetitions);
    const double cpu_direct_checksum = checksum(output);
    const double cpu_slice_s = seconds([&] {
        for (std::size_t i = 0; i < count; ++i) output[i] = host_slice(re[i], segments);
    }, repetitions);
    const double cpu_slice_checksum = checksum(output);

    double *d_re = nullptr;
    Result* d_out = nullptr;
    Segment* d_segments = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_re), count * sizeof(double)), "allocate Re");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_out), count * sizeof(Result)), "allocate results");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_segments), segments.size() * sizeof(Segment)),
               "allocate slice coefficients");
    const auto upload_start = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpy(d_segments, segments.data(), segments.size() * sizeof(Segment),
                         cudaMemcpyHostToDevice), "upload slice coefficients");
    const double upload_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - upload_start).count();
    cuda_check(cudaMemcpy(d_re, re.data(), count * sizeof(double), cudaMemcpyHostToDevice),
               "upload Reynolds inputs");
    const dim3 block(threads);
    const dim3 grid(static_cast<unsigned>((count + threads - 1) / threads));
    const SliceView view{d_segments, segments.size()};
    direct_kernel<<<grid, block>>>(d_re, d_out, count, fixed_roughness);
    slice_kernel<<<grid, block>>>(d_re, d_out, count, fixed_roughness, view);
    cuda_check(cudaGetLastError(), "launch warmup kernels");
    cuda_check(cudaDeviceSynchronize(), "warmup synchronization");

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "create start event");
    cuda_check(cudaEventCreate(&stop), "create stop event");
    auto time_gpu = [&](bool use_slice) {
        cuda_check(cudaEventRecord(start), "record start event");
        for (int rep = 0; rep < repetitions; ++rep) {
            if (use_slice) slice_kernel<<<grid, block>>>(d_re, d_out, count, fixed_roughness, view);
            else direct_kernel<<<grid, block>>>(d_re, d_out, count, fixed_roughness);
        }
        cuda_check(cudaGetLastError(), "launch timed kernels");
        cuda_check(cudaEventRecord(stop), "record stop event");
        cuda_check(cudaEventSynchronize(stop), "wait for GPU");
        float ms = 0.0F;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "read GPU timing");
        cuda_check(cudaMemcpy(output.data(), d_out, count * sizeof(Result),
                              cudaMemcpyDeviceToHost), "download results");
        return static_cast<double>(ms) * 1.0e-3;
    };
    const double gpu_direct_s = time_gpu(false);
    const double direct_checksum = checksum(output);
    const double gpu_slice_s = time_gpu(true);
    const double slice_checksum = checksum(output);
    const double e2e_direct_s = seconds([&] {
        cuda_check(cudaMemcpy(d_re, re.data(), count * sizeof(double), cudaMemcpyHostToDevice),
                   "copy direct inputs");
        direct_kernel<<<grid, block>>>(d_re, d_out, count, fixed_roughness);
        cuda_check(cudaGetLastError(), "launch transfer-inclusive direct kernel");
        cuda_check(cudaMemcpy(output.data(), d_out, count * sizeof(Result), cudaMemcpyDeviceToHost),
                   "copy direct outputs");
    }, repetitions);
    const double e2e_slice_s = seconds([&] {
        cuda_check(cudaMemcpy(d_re, re.data(), count * sizeof(double), cudaMemcpyHostToDevice),
                   "copy slice inputs");
        slice_kernel<<<grid, block>>>(d_re, d_out, count, fixed_roughness, view);
        cuda_check(cudaGetLastError(), "launch transfer-inclusive slice kernel");
        cuda_check(cudaMemcpy(output.data(), d_out, count * sizeof(Result), cudaMemcpyDeviceToHost),
                   "copy slice outputs");
    }, repetitions);
    cuda_check(cudaDeviceSynchronize(), "finish transfer-inclusive runs");

    double max_abs[3]{}, max_rel[3]{};
    std::size_t failures[3]{};
    for (std::size_t i = 0; i < count; ++i) {
        const Result a = reference[i];
        const Result b = output[i];
        const double expected[] = {a.value, a.d_reynolds, a.d_roughness};
        const double actual[] = {b.value, b.d_reynolds, b.d_roughness};
        const double rel_tol[] = {value_rel_tol, dre_rel_tol, drr_rel_tol};
        const double abs_tol[] = {value_abs_tol, dre_abs_tol, drr_abs_tol};
        for (int k = 0; k < 3; ++k) {
            const double error = std::abs(expected[k] - actual[k]);
            max_abs[k] = std::max(max_abs[k], error);
            max_rel[k] = std::max(max_rel[k], error / std::max(std::abs(expected[k]), 1.0e-300));
            if (!std::isfinite(actual[k]) || error > abs_tol[k] + rel_tol[k] * std::abs(expected[k]))
                ++failures[k];
        }
    }
    const double evals = static_cast<double>(count) * repetitions;
    std::cout << std::fixed << std::setprecision(2)
              << "CPU direct=" << evals / cpu_direct_s / 1.0e6 << " M/s"
              << " CPU slice=" << evals / cpu_slice_s / 1.0e6 << " M/s"
              << " (slice/direct=" << cpu_direct_s / cpu_slice_s << "x)"
              << " checksums=" << std::setprecision(10) << cpu_direct_checksum
              << '/' << cpu_slice_checksum << '\n'
              << "GPU coefficient upload=" << std::setprecision(4) << upload_ms << " ms\n"
              << std::setprecision(2)
              << "Direct Colebrook gradients  GPU-kernel=" << evals / gpu_direct_s / 1.0e6
              << " M/s GPU+copies=" << evals / e2e_direct_s / 1.0e6
              << " M/s checksum=" << std::setprecision(10) << direct_checksum << '\n'
              << std::setprecision(2)
              << "Adaptive slice + gradients GPU-kernel=" << evals / gpu_slice_s / 1.0e6
              << " M/s GPU+copies=" << evals / e2e_slice_s / 1.0e6
              << " M/s checksum=" << std::setprecision(10) << slice_checksum << '\n'
              << "Slice error max_abs(value/Re/rr)=" << std::scientific
              << max_abs[0] << '/' << max_abs[1] << '/' << max_abs[2]
              << " max_rel=" << max_rel[0] << '/' << max_rel[1] << '/' << max_rel[2]
              << " failures=" << failures[0] << '/' << failures[1] << '/' << failures[2] << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_re);
    cudaFree(d_out);
    cudaFree(d_segments);
    return failures[0] || failures[1] || failures[2] ||
        build_stats.unresolved_intervals ? EXIT_FAILURE : EXIT_SUCCESS;
}
