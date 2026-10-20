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

// The sampled values are converted once to local bilinear polynomial
// coefficients. GPU evaluation calls the interpolant; it does not find a
// nearest sample or interpolate directly from the four nodal values.
constexpr std::size_t re_nodes = 1025;
constexpr std::size_t roughness_nodes = 1025;
constexpr std::size_t re_cells = re_nodes - 1;
constexpr std::size_t roughness_cells = roughness_nodes - 1;
constexpr double re_min = 2300.0;
constexpr double re_max = 1.0e8;
constexpr double roughness_max = 0.05;
constexpr double roughness_re_scale = 80.0;
constexpr double ln10 = 2.3025850929940456840179914546843642;
constexpr double log_re_min = 7.740664401917241;
constexpr double log_re_span = 10.680016342035124;
constexpr double log_rr_span = 11.042937835596494;

struct Result {
    double value{};
    double d_reynolds{};
    double d_roughness{};
};

struct Coefficients {
    // f(x,y) = a + b*x + c*y + d*x*y, x,y in [0,1] within a cell.
    double a{}, b{}, c{}, d{};
};

struct InterpolantView {
    const Coefficients* cells{};
};

struct ErrorSummary {
    double max_absolute[3]{};
    double max_relative[3]{};
    std::size_t failures[3]{};
};

double coordinate_re(double u) {
    return std::exp(log_re_min + log_re_span * u);
}

double coordinate_roughness(double re, double v) {
    return roughness_re_scale / re * std::expm1(log_rr_span * v);
}

// Sample the exact library wrapper, including the Re=2000..2300 blend.
double sample_colebrook(double re, double rr) {
    return friction::detail::darcy_factor_unchecked(
        re, rr, friction::Correlation::colebrook);
}

std::vector<Coefficients> construct_interpolant() {
    std::vector<double> samples(re_nodes * roughness_nodes);
    for (std::size_t i = 0; i < re_nodes; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(re_cells);
        const double re = coordinate_re(u);
        for (std::size_t j = 0; j < roughness_nodes; ++j) {
            const double v = static_cast<double>(j) / static_cast<double>(roughness_cells);
            samples[i * roughness_nodes + j] = sample_colebrook(
                re, coordinate_roughness(re, v));
        }
    }

    std::vector<Coefficients> cells(re_cells * roughness_cells);
    for (std::size_t i = 0; i < re_cells; ++i) {
        for (std::size_t j = 0; j < roughness_cells; ++j) {
            const double f00 = samples[i * roughness_nodes + j];
            const double f10 = samples[(i + 1) * roughness_nodes + j];
            const double f01 = samples[i * roughness_nodes + (j + 1)];
            const double f11 = samples[(i + 1) * roughness_nodes + (j + 1)];
            cells[i * roughness_cells + j] = {
                f00, f10 - f00, f01 - f00, f11 - f10 - f01 + f00};
        }
    }
    return cells;
}

__device__ __forceinline__ Result colebrook_gradient(double re, double rr) {
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

__device__ __forceinline__ Result evaluate_interpolant(
    double re, double rr, InterpolantView view) {
    // Keep the library's smooth transition closure exact; the interpolant
    // surface starts at its turbulent endpoint, Re=2300.
    if (re < re_min) return colebrook_gradient(re, rr);
    const double log_re = log(re);
    const double u = (log_re - log_re_min) / log_re_span;
    const double roughness_coordinate = rr * re / roughness_re_scale;
    const double v = log1p(roughness_coordinate) / log_rr_span;
    const double scaled_i = u * static_cast<double>(re_cells);
    const double scaled_j = v * static_cast<double>(roughness_cells);
    std::size_t i = static_cast<std::size_t>(scaled_i);
    std::size_t j = static_cast<std::size_t>(scaled_j);
    if (i >= re_cells) i = re_cells - 1;
    if (j >= roughness_cells) j = roughness_cells - 1;
    const double x = scaled_i - static_cast<double>(i);
    const double y = scaled_j - static_cast<double>(j);
    const Coefficients cell = view.cells[i * roughness_cells + j];

    const double value = cell.a + cell.b * x + cell.c * y + cell.d * (x * y);
    const double df_du = static_cast<double>(re_cells) * (cell.b + cell.d * y);
    const double df_dv = static_cast<double>(roughness_cells) * (cell.c + cell.d * x);
    const double dv_dre = (rr / (roughness_re_scale + rr * re)) / log_rr_span;
    const double dv_drr = (re / (roughness_re_scale + rr * re)) / log_rr_span;
    return {value,
            df_du / (log_re_span * re) + df_dv * dv_dre,
            df_dv * dv_drr};
}

__global__ void direct_kernel(const double* re, const double* rr, Result* out,
                              std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = colebrook_gradient(re[i], rr[i]);
}

__global__ void interpolant_kernel(const double* re, const double* rr,
                                   InterpolantView view, Result* out,
                                   std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = evaluate_interpolant(re[i], rr[i], view);
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

Result host_direct(double re, double rr) {
    const auto x = friction::analytic::darcy_friction_factor(
        re, rr, friction::Correlation::colebrook);
    return {x.value, x.d_reynolds, x.d_relative_roughness};
}

double checksum(const std::vector<Result>& x) {
    double sum = 0.0;
    for (const auto& r : x)
        sum += r.value + 1.0e-3 * r.d_reynolds + 1.0e-6 * r.d_roughness;
    return sum;
}

ErrorSummary compare(const std::vector<Result>& expected,
                     const std::vector<Result>& actual) {
    ErrorSummary e{};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double a[] = {expected[i].value, expected[i].d_reynolds,
                            expected[i].d_roughness};
        const double b[] = {actual[i].value, actual[i].d_reynolds,
                            actual[i].d_roughness};
        for (int k = 0; k < 3; ++k) {
            const double error = std::abs(a[k] - b[k]);
            const double scale = std::max(std::abs(a[k]), 1.0e-300);
            e.max_absolute[k] = std::max(e.max_absolute[k], error);
            e.max_relative[k] = std::max(e.max_relative[k], error / scale);
            // The direct GPU implementation is expected to match the CPU
            // closely. The bilinear approximation has its own reported
            // engineering tolerances below.
            if (!std::isfinite(b[k]) || error > 2.0e-13 + 2.0e-10 * scale)
                ++e.failures[k];
        }
    }
    return e;
}

template<class F>
double seconds(F&& f, int repetitions) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

} // namespace

int main() {
    constexpr std::size_t count = 1U << 20;
    constexpr int repetitions = 20;
    constexpr int threads = 256;
    constexpr double value_rel_tol = 5.0e-5;
    constexpr double derivative_rel_tol = 1.0e-2;
    constexpr double value_abs_tol = 2.0e-12;
    constexpr double reynolds_abs_tol = 2.0e-7;
    constexpr double roughness_abs_tol = 2.0e-12;
    const dim3 block(threads);
    const dim3 grid(static_cast<unsigned>((count + threads - 1) / threads));

    int device = 0;
    cuda_check(cudaGetDevice(&device), "get CUDA device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "read CUDA device");
    std::cout << "GPU: " << properties.name << " (compute " << properties.major
              << '.' << properties.minor << ")\n"
              << "Batch: " << count << " independent (Re, relative roughness) inputs; "
              << repetitions << " passes\n"
              << "Interpolant: " << re_nodes << " x " << roughness_nodes
              << " samples; bilinear function in log(Re), log1p(Re*rr/80) coordinates; "
              << "exact transition branch below Re=2300\n";

    const auto build_start = std::chrono::steady_clock::now();
    std::vector<Coefficients> coefficients = construct_interpolant();
    const double build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - build_start).count();
    std::cout << "CPU sample + coefficient construction=" << std::fixed
              << std::setprecision(3) << build_ms << " ms; coefficient storage="
              << (coefficients.size() * sizeof(Coefficients)) / (1024.0 * 1024.0)
              << " MiB\n";

    std::vector<double> re(count), rr(count);
    std::vector<Result> direct_reference(count), output(count);
    std::mt19937_64 rng(20261020);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (std::size_t i = 0; i < count; ++i) {
        re[i] = 2000.0 * std::pow(re_max / 2000.0, unit(rng));
        const double v_max = std::log1p(roughness_max * re[i] /
                                        roughness_re_scale);
        rr[i] = coordinate_roughness(
            re[i], unit(rng) * v_max / log_rr_span);
        direct_reference[i] = host_direct(re[i], rr[i]);
    }

    const double cpu_direct_s = seconds([&] {
        for (std::size_t i = 0; i < count; ++i)
            output[i] = host_direct(re[i], rr[i]);
    }, repetitions);
    const double cpu_direct_checksum = checksum(output);
    std::cout << "CPU direct Colebrook=" << std::fixed << std::setprecision(2)
              << (static_cast<double>(count) * repetitions / cpu_direct_s / 1.0e6)
              << " M/s checksum=" << std::setprecision(10) << cpu_direct_checksum << '\n';

    double *d_re = nullptr, *d_rr = nullptr;
    Result* d_out = nullptr;
    Coefficients* d_coefficients = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_re), count * sizeof(double)), "allocate Re");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rr), count * sizeof(double)), "allocate roughness");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_out), count * sizeof(Result)), "allocate outputs");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_coefficients),
                          coefficients.size() * sizeof(Coefficients)), "allocate interpolant coefficients");

    const auto upload_start = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpy(d_coefficients, coefficients.data(),
                         coefficients.size() * sizeof(Coefficients),
                         cudaMemcpyHostToDevice), "upload interpolant coefficients");
    const double upload_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - upload_start).count();
    cuda_check(cudaMemcpy(d_re, re.data(), count * sizeof(double), cudaMemcpyHostToDevice), "upload Re inputs");
    cuda_check(cudaMemcpy(d_rr, rr.data(), count * sizeof(double), cudaMemcpyHostToDevice), "upload roughness inputs");
    std::cout << "GPU interpolant upload=" << upload_ms << " ms\n"
              << "Accuracy targets: value rel=" << std::scientific << value_rel_tol
              << "; derivative rel=" << derivative_rel_tol
              << " (absolute floors value/Re/rr=" << value_abs_tol << '/'
              << reynolds_abs_tol << '/' << roughness_abs_tol << ")\n";

    const InterpolantView view{d_coefficients};
    direct_kernel<<<grid, block>>>(d_re, d_rr, d_out, count);
    interpolant_kernel<<<grid, block>>>(d_re, d_rr, view, d_out, count);
    cuda_check(cudaGetLastError(), "launch warmup kernels");
    cuda_check(cudaDeviceSynchronize(), "warmup synchronization");

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "create start event");
    cuda_check(cudaEventCreate(&stop), "create stop event");
    auto time_gpu = [&](bool use_interpolant) {
        cuda_check(cudaEventRecord(start), "record start");
        for (int rep = 0; rep < repetitions; ++rep) {
            if (use_interpolant)
                interpolant_kernel<<<grid, block>>>(d_re, d_rr, view, d_out, count);
            else
                direct_kernel<<<grid, block>>>(d_re, d_rr, d_out, count);
        }
        cuda_check(cudaGetLastError(), "launch timed kernels");
        cuda_check(cudaEventRecord(stop), "record stop");
        cuda_check(cudaEventSynchronize(stop), "wait for GPU");
        float ms = 0.0F;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "read GPU timing");
        cuda_check(cudaMemcpy(output.data(), d_out, count * sizeof(Result),
                              cudaMemcpyDeviceToHost), "download results");
        return static_cast<double>(ms) * 1.0e-3;
    };

    const double direct_gpu_s = time_gpu(false);
    const ErrorSummary direct_error = compare(direct_reference, output);
    const double direct_checksum = checksum(output);
    const double interp_gpu_s = time_gpu(true);
    const ErrorSummary interp_error = compare(direct_reference, output);
    const double interp_checksum = checksum(output);

    const double direct_e2e_s = seconds([&] {
        cuda_check(cudaMemcpy(d_re, re.data(), count * sizeof(double), cudaMemcpyHostToDevice), "copy direct Re");
        cuda_check(cudaMemcpy(d_rr, rr.data(), count * sizeof(double), cudaMemcpyHostToDevice), "copy direct roughness");
        direct_kernel<<<grid, block>>>(d_re, d_rr, d_out, count);
        cuda_check(cudaGetLastError(), "launch direct transfer-inclusive kernel");
        cuda_check(cudaMemcpy(output.data(), d_out, count * sizeof(Result), cudaMemcpyDeviceToHost), "copy direct output");
    }, repetitions);
    const double interp_e2e_s = seconds([&] {
        cuda_check(cudaMemcpy(d_re, re.data(), count * sizeof(double), cudaMemcpyHostToDevice), "copy interpolant Re");
        cuda_check(cudaMemcpy(d_rr, rr.data(), count * sizeof(double), cudaMemcpyHostToDevice), "copy interpolant roughness");
        interpolant_kernel<<<grid, block>>>(d_re, d_rr, view, d_out, count);
        cuda_check(cudaGetLastError(), "launch interpolant transfer-inclusive kernel");
        cuda_check(cudaMemcpy(output.data(), d_out, count * sizeof(Result), cudaMemcpyDeviceToHost), "copy interpolant output");
    }, repetitions);
    cuda_check(cudaDeviceSynchronize(), "finish transfer-inclusive runs");

    const double evals = static_cast<double>(count) * repetitions;
    std::size_t interpolant_failures[3]{};
    for (std::size_t i = 0; i < count; ++i) {
        const double a[] = {direct_reference[i].value, direct_reference[i].d_reynolds,
                            direct_reference[i].d_roughness};
        const double b[] = {output[i].value, output[i].d_reynolds,
                            output[i].d_roughness};
        for (int k = 0; k < 3; ++k) {
            const double rel_tol = k == 0 ? value_rel_tol : derivative_rel_tol;
            const double abs_tol = k == 0 ? value_abs_tol
                : (k == 1 ? reynolds_abs_tol : roughness_abs_tol);
            if (!std::isfinite(b[k]) || std::abs(a[k] - b[k]) >
                abs_tol + rel_tol * std::abs(a[k])) ++interpolant_failures[k];
        }
    }
    auto print = [&](const char* label, double gpu_s, double e2e_s,
                     const ErrorSummary& error, double gpu_checksum, bool direct) {
        std::cout << std::left << std::setw(28) << label
                  << " GPU-kernel=" << std::setw(8) << evals / gpu_s / 1.0e6 << " M/s"
                  << " GPU+copies=" << std::setw(8) << evals / e2e_s / 1.0e6 << " M/s"
                  << " max_rel(value/Re/rr)=" << std::scientific
                  << error.max_relative[0] << '/' << error.max_relative[1] << '/'
                  << error.max_relative[2]
                  << " max_abs=" << error.max_absolute[0] << '/' << error.max_absolute[1]
                  << '/' << error.max_absolute[2]
                  << " checksum=" << std::fixed << std::setprecision(10)
                  << gpu_checksum;
        if (direct) {
            std::cout << " failures=" << error.failures[0] + error.failures[1] + error.failures[2];
        } else {
            std::cout << " accuracy_failures(value/Re/rr)=" << interpolant_failures[0] << '/'
                      << interpolant_failures[1] << '/' << interpolant_failures[2];
        }
        std::cout << '\n';
    };

    print("Direct Colebrook", direct_gpu_s, direct_e2e_s,
          direct_error, direct_checksum, true);
    print("Bilinear interpolant function", interp_gpu_s, interp_e2e_s,
          interp_error, interp_checksum, false);
    std::cout << "Accuracy failures (value/Re/rr)=" << interpolant_failures[0] << '/'
              << interpolant_failures[1] << '/' << interpolant_failures[2] << '\n';
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_re); cudaFree(d_rr); cudaFree(d_out); cudaFree(d_coefficients);
    return direct_error.failures[0] || direct_error.failures[1] ||
           direct_error.failures[2] || interpolant_failures[0] ||
           interpolant_failures[1] || interpolant_failures[2]
        ? EXIT_FAILURE : EXIT_SUCCESS;
}
