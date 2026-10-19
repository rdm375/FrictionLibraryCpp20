#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t maximum_count = 1U << 20;
constexpr double ln10 = 2.3025850929940456840179914546843642;

struct Result {
    double value;
    double d_reynolds;
    double d_roughness;
};

__host__ __device__ double colebrook(double re, double rr) {
    constexpr double x1_scale = ln10 / 18.574;
    constexpr double x2_offset = -0.7793974884556818028;
    const double x1 = rr * re * x1_scale;
    const double x2 = log(re) + x2_offset;
    double z = x2 - 0.2;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const double shifted = x1 + z;
        const double epsilon = (z + log(shifted) - x2) / (1.0 + shifted);
        const double numerator = 1.0 + shifted + 0.5 * epsilon;
        const double denominator = 1.0 + shifted + epsilon * (1.0 + epsilon / 3.0);
        z -= (numerator / denominator) * epsilon * shifted;
    }
    const double y = (2.0 / ln10) * z;
    return 1.0 / (y * y);
}

__host__ __device__ Result colebrook_with_derivatives(double re, double rr) {
    const double value = colebrook(re, rr);
    const double y = 1.0 / sqrt(value);
    const double arg = rr / 3.7 + 2.51 * y / re;
    constexpr double two_over_ln10 = 2.0 / ln10;
    const double residual_y = 1.0 + two_over_ln10 * (2.51 / re) / arg;
    const double y_re = (two_over_ln10 * 2.51 * y / (re * re * arg)) / residual_y;
    const double y_rr = -(two_over_ln10 / (3.7 * arg)) / residual_y;
    const double df_dy = -2.0 / (y * y * y);
    return {value, df_dy * y_re, df_dy * y_rr};
}

__host__ __device__ double swamee_jain(double re, double rr) {
#ifdef __CUDA_ARCH__
    const double re_power = exp2(0.9 * log2(re));
#else
    const double re_power = pow(re, 0.9);
#endif
    const double arg = rr / 3.7 + 5.74 / re_power;
    const double l = log10(arg);
    return 0.25 / (l * l);
}

__host__ __device__ Result swamee_jain_with_derivatives(double re, double rr) {
#ifdef __CUDA_ARCH__
    const double re_power = exp2(0.9 * log2(re));
#else
    const double re_power = pow(re, 0.9);
#endif
    const double re_term = 5.74 / re_power;
    const double arg = rr / 3.7 + re_term;
    const double l = log10(arg);
    const double value = 0.25 / (l * l);
    const double common = -0.5 / (ln10 * arg * l * l * l);
    return {value, common * (-0.9 * re_term / re), common / 3.7};
}

__global__ void evaluate_kernel(const double* re, const double* rr, Result* output,
                                std::size_t count, int mode) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double r = re[i], rough = rr[i];
    Result result{};
    switch (mode) {
        case 0: result.value = colebrook(r, rough); break;
        case 1: result = colebrook_with_derivatives(r, rough); break;
        case 2: result.value = swamee_jain(r, rough); break;
        default: result = swamee_jain_with_derivatives(r, rough); break;
    }
    output[i] = result;
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

double checksum(const std::vector<Result>& values) {
    double sum = 0.0;
    for (const auto& x : values)
        sum += x.value + 1.0e-3 * x.d_reynolds + 1.0e-6 * x.d_roughness;
    return sum;
}

struct ErrorSummary {
    double max_absolute{};
    double max_relative{};
    std::size_t failures{};
};

ErrorSummary compare_results(const std::vector<Result>& reference,
                             const std::vector<Result>& candidate) {
    constexpr double absolute_tolerance = 2.0e-14;
    constexpr double relative_tolerance = 2.0e-11;
    ErrorSummary summary{};
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double expected[] = {reference[i].value, reference[i].d_reynolds,
                                   reference[i].d_roughness};
        const double actual[] = {candidate[i].value, candidate[i].d_reynolds,
                                 candidate[i].d_roughness};
        for (int component = 0; component < 3; ++component) {
            const double error = std::abs(actual[component] - expected[component]);
            const double scale = std::max(std::abs(expected[component]), 1.0e-300);
            summary.max_absolute = std::max(summary.max_absolute, error);
            summary.max_relative = std::max(summary.max_relative, error / scale);
            if (!std::isfinite(actual[component]) ||
                error > absolute_tolerance + relative_tolerance * scale)
                ++summary.failures;
        }
    }
    return summary;
}

int repetitions_for(std::size_t count) {
    constexpr std::size_t target_evaluations = 1U << 22;
    const std::size_t reps = (target_evaluations + count - 1) / count;
    return static_cast<int>(std::clamp<std::size_t>(reps, 5, 2000));
}

using HostFunction = Result (*)(double, double);
Result host_colebrook(double re, double rr) { return {colebrook(re, rr), 0.0, 0.0}; }
Result host_colebrook_grad(double re, double rr) { return colebrook_with_derivatives(re, rr); }
Result host_swamee(double re, double rr) { return {swamee_jain(re, rr), 0.0, 0.0}; }
Result host_swamee_grad(double re, double rr) { return swamee_jain_with_derivatives(re, rr); }

void benchmark(const char* label, int mode, HostFunction host_fn,
               const std::vector<double>& re, const std::vector<double>& rr,
               double* d_re, double* d_rr, Result* d_out,
               std::vector<Result>& output, std::size_t count) {
    const int repetitions = repetitions_for(count);
    const dim3 block(256);
    const dim3 grid(static_cast<unsigned>((count + block.x - 1) / block.x));

    const auto cpu_start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repetitions; ++rep)
        for (std::size_t i = 0; i < count; ++i)
            output[i] = host_fn(re[i], rr[i]);
    const auto cpu_stop = std::chrono::steady_clock::now();
    const double cpu_seconds = std::chrono::duration<double>(cpu_stop - cpu_start).count();
    const std::vector<Result> cpu_reference = output;

    cuda_check(cudaMemcpy(d_re, re.data(), count * sizeof(double), cudaMemcpyHostToDevice), "copy Re to GPU");
    cuda_check(cudaMemcpy(d_rr, rr.data(), count * sizeof(double), cudaMemcpyHostToDevice), "copy roughness to GPU");
    evaluate_kernel<<<grid, block>>>(d_re, d_rr, d_out, count, mode);
    cuda_check(cudaGetLastError(), "launch warmup kernel");
    cuda_check(cudaDeviceSynchronize(), "warmup synchronization");

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "create CUDA start event");
    cuda_check(cudaEventCreate(&stop), "create CUDA stop event");
    cuda_check(cudaEventRecord(start), "record CUDA start event");
    for (int rep = 0; rep < repetitions; ++rep)
        evaluate_kernel<<<grid, block>>>(d_re, d_rr, d_out, count, mode);
    cuda_check(cudaGetLastError(), "launch timed kernel");
    cuda_check(cudaEventRecord(stop), "record CUDA stop event");
    cuda_check(cudaEventSynchronize(stop), "wait for timed kernels");
    float kernel_ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&kernel_ms, start, stop), "read CUDA event time");
    cuda_check(cudaMemcpy(output.data(), d_out, count * sizeof(Result), cudaMemcpyDeviceToHost), "copy output from GPU");
    const ErrorSummary error = compare_results(cpu_reference, output);

    // End-to-end throughput includes both input copies and output copy for every pass.
    const auto e2e_start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repetitions; ++rep) {
        cuda_check(cudaMemcpy(d_re, re.data(), count * sizeof(double), cudaMemcpyHostToDevice), "copy Re to GPU");
        cuda_check(cudaMemcpy(d_rr, rr.data(), count * sizeof(double), cudaMemcpyHostToDevice), "copy roughness to GPU");
        evaluate_kernel<<<grid, block>>>(d_re, d_rr, d_out, count, mode);
        cuda_check(cudaGetLastError(), "launch end-to-end kernel");
        cuda_check(cudaMemcpy(output.data(), d_out, count * sizeof(Result), cudaMemcpyDeviceToHost), "copy output from GPU");
    }
    cuda_check(cudaDeviceSynchronize(), "end-to-end synchronization");
    const auto e2e_stop = std::chrono::steady_clock::now();
    const double e2e_seconds = std::chrono::duration<double>(e2e_stop - e2e_start).count();

    const double cpu_us = cpu_seconds * 1.0e6 / repetitions;
    const double gpu_kernel_us = static_cast<double>(kernel_ms) * 1000.0 / repetitions;
    const double gpu_e2e_us = e2e_seconds * 1.0e6 / repetitions;
    const double cpu_sum = checksum(cpu_reference);
    const double gpu_sum = checksum(output);
    std::cout << std::left << std::setw(25) << label << " n=" << std::setw(8) << count
              << " reps=" << std::setw(5) << repetitions
              << " CPU=" << std::setw(9) << std::setprecision(3) << cpu_us << " us"
              << " GPU-kernel=" << std::setw(9) << gpu_kernel_us << " us"
              << " GPU+copies=" << std::setw(9) << gpu_e2e_us << " us"
              << " E2E speedup=" << std::setw(6) << cpu_us / gpu_e2e_us << "x"
              << " max|err|=" << std::scientific << error.max_absolute
              << " max rel=" << error.max_relative
              << " failures=" << error.failures
              << " checksums=" << std::fixed << std::setprecision(10)
              << cpu_sum << '/' << gpu_sum << '\n';
    if (error.failures != 0) {
        std::cerr << "CPU/GPU elementwise validation failed for " << label
                  << " at batch size " << count << '\n';
        std::exit(EXIT_FAILURE);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

} // namespace

int main() {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "select current CUDA device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "read CUDA device properties");
    std::cout << "GPU: " << properties.name << " (compute " << properties.major << '.'
              << properties.minor << ")\n"
              << "Batch-size sweep: 1, 8, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576.\n"
              << "Timings are per batch; GPU+copies includes both inputs and the output copy.\n"
              << "Elementwise CPU/GPU checks use abs_tol=2e-14 and rel_tol=2e-11.\n";

    double *d_re = nullptr, *d_rr = nullptr;
    Result* d_out = nullptr;
    cuda_check(cudaMalloc(&d_re, maximum_count * sizeof(double)), "allocate GPU Re buffer");
    cuda_check(cudaMalloc(&d_rr, maximum_count * sizeof(double)), "allocate GPU roughness buffer");
    cuda_check(cudaMalloc(&d_out, maximum_count * sizeof(Result)), "allocate GPU output buffer");

    constexpr std::size_t sizes[] = {
        1, 8, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, maximum_count};
    for (std::size_t count : sizes) {
        std::vector<double> re(count), rr(count);
        std::vector<Result> output(count);
        for (std::size_t i = 0; i < count; ++i) {
            re[i] = 5000.0 + static_cast<double>((i * 7919U) % 100000000U);
            rr[i] = 1.0e-6 + static_cast<double>((i * 104729U) % 50000U) * 1.0e-6;
        }
        benchmark("Colebrook primal", 0, host_colebrook, re, rr, d_re, d_rr, d_out, output, count);
        benchmark("Colebrook + gradients", 1, host_colebrook_grad, re, rr, d_re, d_rr, d_out, output, count);
        benchmark("Swamee-Jain primal", 2, host_swamee, re, rr, d_re, d_rr, d_out, output, count);
        benchmark("Swamee-Jain + gradients", 3, host_swamee_grad, re, rr, d_re, d_rr, d_out, output, count);
    }

    cudaFree(d_re);
    cudaFree(d_rr);
    cudaFree(d_out);
    return 0;
}
