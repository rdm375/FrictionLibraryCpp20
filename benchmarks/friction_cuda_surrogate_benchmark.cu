// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#include "friction/adaptive_tensor_surrogate.hpp"

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

using Tensor = friction::surrogate::AdaptiveColebrookTensor;
using Prepared = Tensor::PreparedRoughness;
using DeviceView = Prepared::DeviceView;
using Row = Prepared::Row;
using HostDerivatives = friction::analytic::FirstDerivatives;

struct Result {
    double value{};
    double d_reynolds{};
    double d_roughness{};
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
    const double y_re = ((2.0 / ln10) * 2.51 * root / (re * re * arg)) / residual_y;
    const double y_rr = (-(2.0 / ln10) / (3.7 * arg)) / residual_y;
    const double df_dy = -2.0 / (root * root * root);
    const Result turbulent{f, df_dy * y_re, df_dy * y_rr};

    // Match friction::analytic::darcy_friction_factor_unchecked exactly:
    // laminar below Re=2000, followed by a cubic smoothstep blend to the
    // turbulent Colebrook correlation over [2000, 2300].
    if (re < 2000.0) {
        return {64.0 / re, -64.0 / (re * re), 0.0};
    }
    if (re >= 2300.0) return turbulent;

    constexpr double width = 300.0;
    const double t = (re - 2000.0) / width;
    const double weight = t * t * (3.0 - 2.0 * t);
    const double weight_re = 6.0 * t * (1.0 - t) / width;
    const double laminar = 64.0 / re;
    const double laminar_re = -64.0 / (re * re);
    return {(1.0 - weight) * laminar + weight * turbulent.value,
            (1.0 - weight) * laminar_re + weight * turbulent.d_reynolds +
                weight_re * (turbulent.value - laminar),
            weight * turbulent.d_roughness};
}

__global__ void direct_kernel(const double* re, const double* rr, Result* out,
                              std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = colebrook_gradient(re[i], rr[i]);
}

__global__ void prepared_kernel(const double* re, DeviceView view, Result* out,
                                std::size_t count) {
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const double reynolds = re[k];
    const double u = (log10(reynolds) - view.log_re_min) * view.inverse_log_re_span;
    std::size_t bin = static_cast<std::size_t>(u * view.interval_bin_count);
    if (bin >= view.interval_bin_count) bin = view.interval_bin_count - 1;
    std::size_t i = view.interval_bins[bin];
    while (i + 1 < view.row_count && view.reynolds_nodes[i + 1] <= u) ++i;

    const double x = (u - view.reynolds_nodes[i]) * view.inverse_reynolds_widths[i];
    const Row& row = view.rows[i];
    const double g = ((row.value[3] * x + row.value[2]) * x + row.value[1]) * x + row.value[0];
    const double g_u = ((3.0 * row.value[3] * x + 2.0 * row.value[2]) * x + row.value[1]) *
                       view.inverse_reynolds_widths[i];
    const double g_v = ((row.d_roughness[3] * x + row.d_roughness[2]) * x +
                         row.d_roughness[1]) * x + row.d_roughness[0];
    const double inv_g = 1.0 / g;
    const double f = inv_g * inv_g;
    const double f_u = -2.0 * g_u * f * inv_g;
    const double f_v = -2.0 * g_v * f * inv_g;
    out[k] = {f, f_u * (view.inverse_re_derivative_scale / reynolds),
              f_v * view.roughness_derivative_scale};
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

Result host_direct(double re, double rr) {
    const HostDerivatives x = friction::analytic::darcy_friction_factor(
        re, rr, friction::Correlation::colebrook);
    return {x.value, x.d_reynolds, x.d_relative_roughness};
}

Result host_prepared(const Prepared& slice, double re) {
    const auto x = slice.evaluate(re).derivatives;
    return {x.value, x.d_reynolds, x.d_relative_roughness};
}

double checksum(const std::vector<Result>& values) {
    double sum = 0.0;
    for (const Result& x : values)
        sum += x.value + 1.0e-3 * x.d_reynolds + 1.0e-6 * x.d_roughness;
    return sum;
}

struct Errors {
    double max_absolute{};
    double max_relative{};
    std::size_t failures{};
};

Errors compare(const std::vector<Result>& expected, const std::vector<Result>& actual) {
    Errors e{};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double a[] = {expected[i].value, expected[i].d_reynolds, expected[i].d_roughness};
        const double b[] = {actual[i].value, actual[i].d_reynolds, actual[i].d_roughness};
        for (int j = 0; j < 3; ++j) {
            const double error = std::abs(a[j] - b[j]);
            const double scale = std::max(std::abs(a[j]), 1.0e-300);
            e.max_absolute = std::max(e.max_absolute, error);
            e.max_relative = std::max(e.max_relative, error / scale);
            if (!std::isfinite(b[j]) || error > 2.0e-13 + 2.0e-10 * scale) ++e.failures;
        }
    }
    return e;
}

template<class F>
double cpu_seconds(F&& f, int repetitions) {
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repetitions; ++rep) f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace

int main() {
    constexpr std::size_t count = 1U << 20;
    constexpr int repetitions = 20;
    constexpr double fixed_rr = 6.971775e-4;
    constexpr int threads = 256;
    const dim3 block(threads);
    const dim3 grid(static_cast<unsigned>((count + threads - 1) / threads));

    int device = 0;
    cuda_check(cudaGetDevice(&device), "get CUDA device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "read CUDA device");
    std::cout << "GPU: " << properties.name << " (compute " << properties.major << '.'
              << properties.minor << ")\n"
              << "Batch: " << count << " inputs; " << repetitions
              << " passes; fixed relative roughness=" << std::scientific << fixed_rr << '\n'
              << "Kernel timings keep the prepared slice and inputs resident on the GPU.\n"
              << "Transfer timings include input transfer, slice refresh when applicable, and output transfer.\n";

    friction::surrogate::AdaptiveTensorConfig config;
    Tensor table(config);
    const auto build_start = std::chrono::steady_clock::now();
    const auto report = table.build();
    const double build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - build_start).count();
    const auto prep_start = std::chrono::steady_clock::now();
    const Prepared slice = table.prepare_roughness(fixed_rr);
    const double prep_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prep_start).count();
    std::cout << "CPU table build=" << std::fixed << std::setprecision(3) << build_ms
              << " ms; roughness preparation=" << prep_ms << " ms; nodes="
              << report.grid_points << '\n';

    std::vector<double> re(count), rr(count, fixed_rr);
    std::vector<Result> direct_reference(count), prepared_reference(count), output(count);
    std::mt19937_64 rng(20261019);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (std::size_t i = 0; i < count; ++i) {
        const double u = unit(rng);
        re[i] = config.reynolds_min * std::pow(config.reynolds_max/config.reynolds_min, u);
        direct_reference[i] = host_direct(re[i], fixed_rr);
        prepared_reference[i] = host_prepared(slice, re[i]);
    }

    const double cpu_direct_s = cpu_seconds([&] {
        for (std::size_t i = 0; i < count; ++i) output[i] = host_direct(re[i], fixed_rr);
    }, repetitions);
    const double cpu_prepared_s = cpu_seconds([&] {
        for (std::size_t i = 0; i < count; ++i) output[i] = host_prepared(slice, re[i]);
    }, repetitions);

    double *d_re = nullptr, *d_rr = nullptr, *d_nodes = nullptr, *d_inv_widths = nullptr;
    std::size_t* d_bins = nullptr;
    Row* d_rows = nullptr;
    Result* d_out = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_re), count*sizeof(double)), "allocate Re buffer");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rr), count*sizeof(double)), "allocate roughness buffer");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_out), count*sizeof(Result)), "allocate output buffer");
    const DeviceView host_view = slice.device_view();
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_nodes), (host_view.row_count+1)*sizeof(double)), "allocate Re knots");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_inv_widths), host_view.row_count*sizeof(double)), "allocate inverse widths");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bins), host_view.interval_bin_count*sizeof(std::size_t)), "allocate interval bins");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rows), host_view.row_count*sizeof(Row)), "allocate slice rows");
    const auto axes_upload_start = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpy(d_nodes, host_view.reynolds_nodes, (host_view.row_count+1)*sizeof(double), cudaMemcpyHostToDevice), "copy Re knots");
    cuda_check(cudaMemcpy(d_inv_widths, host_view.inverse_reynolds_widths, host_view.row_count*sizeof(double), cudaMemcpyHostToDevice), "copy inverse widths");
    cuda_check(cudaMemcpy(d_bins, host_view.interval_bins, host_view.interval_bin_count*sizeof(std::size_t), cudaMemcpyHostToDevice), "copy interval bins");
    const double axes_upload_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now()-axes_upload_start).count();
    const auto slice_upload_start = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpy(d_rows, host_view.rows, host_view.row_count*sizeof(Row), cudaMemcpyHostToDevice), "copy prepared slice");
    const double slice_upload_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now()-slice_upload_start).count();
    cuda_check(cudaMemcpy(d_re, re.data(), count*sizeof(double), cudaMemcpyHostToDevice), "copy Reynolds inputs");
    cuda_check(cudaMemcpy(d_rr, rr.data(), count*sizeof(double), cudaMemcpyHostToDevice), "copy roughness inputs");
    std::cout << "GPU axes upload=" << std::fixed << std::setprecision(3)
              << axes_upload_ms << " ms; initial slice upload=" << slice_upload_ms << " ms\n";

    DeviceView device_view = host_view;
    device_view.reynolds_nodes = d_nodes;
    device_view.inverse_reynolds_widths = d_inv_widths;
    device_view.interval_bins = d_bins;
    device_view.rows = d_rows;
    direct_kernel<<<grid, block>>>(d_re, d_rr, d_out, count);
    prepared_kernel<<<grid, block>>>(d_re, device_view, d_out, count);
    cuda_check(cudaGetLastError(), "launch warmup kernels");
    cuda_check(cudaDeviceSynchronize(), "warmup synchronization");

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "create start event");
    cuda_check(cudaEventCreate(&stop), "create stop event");
    auto time_gpu = [&](bool prepared) {
        cuda_check(cudaEventRecord(start), "record start event");
        for (int rep = 0; rep < repetitions; ++rep) {
            if (prepared) prepared_kernel<<<grid, block>>>(d_re, device_view, d_out, count);
            else direct_kernel<<<grid, block>>>(d_re, d_rr, d_out, count);
        }
        cuda_check(cudaGetLastError(), "launch timed kernels");
        cuda_check(cudaEventRecord(stop), "record stop event");
        cuda_check(cudaEventSynchronize(stop), "wait for GPU kernels");
        float milliseconds = 0.0F;
        cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "read GPU elapsed time");
        cuda_check(cudaMemcpy(output.data(), d_out, count*sizeof(Result), cudaMemcpyDeviceToHost), "copy results");
        return static_cast<double>(milliseconds)*1.0e-3;
    };
    const double gpu_direct_s = time_gpu(false);
    const Errors direct_errors = compare(direct_reference, output);
    const double gpu_direct_checksum = checksum(output);
    const double gpu_prepared_s = time_gpu(true);
    const Errors prepared_errors = compare(prepared_reference, output);
    const double gpu_prepared_checksum = checksum(output);

    const double e2e_direct_s = cpu_seconds([&] {
        cuda_check(cudaMemcpy(d_re, re.data(), count*sizeof(double), cudaMemcpyHostToDevice), "copy Re inputs");
        cuda_check(cudaMemcpy(d_rr, rr.data(), count*sizeof(double), cudaMemcpyHostToDevice), "copy roughness inputs");
        direct_kernel<<<grid, block>>>(d_re, d_rr, d_out, count);
        cuda_check(cudaGetLastError(), "launch direct transfer-inclusive kernel");
        cuda_check(cudaMemcpy(output.data(), d_out, count*sizeof(Result), cudaMemcpyDeviceToHost), "copy direct output");
    }, repetitions);
    const double e2e_prepared_s = cpu_seconds([&] {
        cuda_check(cudaMemcpy(d_re, re.data(), count*sizeof(double), cudaMemcpyHostToDevice), "copy Re inputs");
        cuda_check(cudaMemcpy(d_rows, host_view.rows, host_view.row_count*sizeof(Row), cudaMemcpyHostToDevice), "refresh prepared slice");
        prepared_kernel<<<grid, block>>>(d_re, device_view, d_out, count);
        cuda_check(cudaGetLastError(), "launch prepared transfer-inclusive kernel");
        cuda_check(cudaMemcpy(output.data(), d_out, count*sizeof(Result), cudaMemcpyDeviceToHost), "copy prepared output");
    }, repetitions);
    cuda_check(cudaDeviceSynchronize(), "finish transfer-inclusive runs");

    auto print = [&](const char* name, double cpu_s, double kernel_s, double e2e_s,
                     const Errors& errors, double cpu_checksum, double gpu_checksum) {
        const double evaluated = static_cast<double>(count)*repetitions;
        std::cout << std::left << std::setw(24) << name
                  << " CPU=" << std::setw(8) << std::setprecision(2) << evaluated/cpu_s/1.0e6 << " M/s"
                  << " GPU-kernel=" << std::setw(8) << evaluated/kernel_s/1.0e6 << " M/s"
                  << " GPU+copies=" << std::setw(8) << evaluated/e2e_s/1.0e6 << " M/s"
                  << " max_abs=" << std::scientific << errors.max_absolute
                  << " max_rel=" << errors.max_relative
                  << " failures=" << errors.failures
                  << " checksum=" << std::fixed << std::setprecision(10)
                  << cpu_checksum << '/' << gpu_checksum << '\n';
    };

    std::vector<Result> direct_cpu_once(count), prepared_cpu_once(count);
    for (std::size_t i = 0; i < count; ++i) {
        direct_cpu_once[i] = host_direct(re[i], fixed_rr);
        prepared_cpu_once[i] = host_prepared(slice, re[i]);
    }
    print("Colebrook gradients", cpu_direct_s, gpu_direct_s, e2e_direct_s,
          direct_errors, checksum(direct_cpu_once), gpu_direct_checksum);
    print("Prepared tensor slice", cpu_prepared_s, gpu_prepared_s, e2e_prepared_s,
          prepared_errors, checksum(prepared_cpu_once), gpu_prepared_checksum);

    if (direct_errors.failures || prepared_errors.failures) return EXIT_FAILURE;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_re); cudaFree(d_rr); cudaFree(d_out); cudaFree(d_nodes);
    cudaFree(d_inv_widths); cudaFree(d_bins); cudaFree(d_rows);
    return EXIT_SUCCESS;
}
