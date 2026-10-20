// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#include "friction/adaptive_bilinear_tensor_surrogate.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Config = friction::surrogate::AdaptiveTensorConfig;
using Table = friction::surrogate::AdaptiveBilinearColebrookTensor;
using BicubicTable = friction::surrogate::AdaptiveColebrookTensor;
constexpr std::size_t count = 1U << 20;
constexpr int repetitions = 12;
constexpr int block_size = 256;
constexpr double ln10 = 2.3025850929940456840179914546843642;

struct Point { double re, rr; };
struct Result { double value, d_re, d_rr; };
struct Cell { double c[4]; };
struct CubicCell { double c[16]; };
struct Error {
    double value{}, dre{}, drr{};
    std::size_t fail_value{}, fail_dre{}, fail_drr{};
};

void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::cerr << where << ": " << cudaGetErrorString(e) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

__device__ int interval(const double* x, int n, double q) {
    if (q >= x[n - 1]) return n - 2;
    int lo = 0, hi = n - 1;
    while (lo + 1 < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (x[mid] <= q) lo = mid; else hi = mid;
    }
    return lo;
}

__device__ Result colebrook(double re, double rr) {
    const double x1 = rr * re * (ln10 / 18.574);
    const double x2 = log(re) - 0.7793974884556818028;
    double z = x2 - 0.2;
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const double q = x1 + z;
        const double e = (z + log(q) - x2) / (1.0 + q);
        z -= ((1.0 + q + 0.5 * e) / (1.0 + q + e * (1.0 + e / 3.0))) * e * q;
    }
    const double y = (2.0 / ln10) * z;
    const double f = 1.0 / (y * y), root = 1.0 / sqrt(f);
    const double a = rr / 3.7 + 2.51 * root / re;
    const double ry = 1.0 + (2.0 / ln10) * (2.51 / re) / a;
    const double yre = ((2.0 / ln10) * 2.51 * root / (re * re * a)) / ry;
    const double yrr = (-(2.0 / ln10) / (3.7 * a)) / ry;
    const double dfdy = -2.0 / (root * root * root);
    return {f, dfdy * yre, dfdy * yrr};
}

__device__ Result interpolate(double re, double rr, const double* u_axis,
                               const double* v_axis, int nu, int nv,
                               const Cell* cells, double u_span,
                               double v_span, double rr_scale) {
    const double u = log10(re / 2000.0) / u_span;
    const double v = log1p(rr / rr_scale) / v_span;
    const int i = interval(u_axis, nu, u), j = interval(v_axis, nv, v);
    const double hu = u_axis[i + 1] - u_axis[i];
    const double hv = v_axis[j + 1] - v_axis[j];
    const double x = (u - u_axis[i]) / hu;
    const double y = (v - v_axis[j]) / hv;
    const Cell& cell = cells[static_cast<std::size_t>(i) * (nv - 1) + j];
    const double g = cell.c[0] + cell.c[1] * x + cell.c[2] * y + cell.c[3] * x * y;
    const double gu = (cell.c[1] + cell.c[3] * y) / hu;
    const double gv = (cell.c[2] + cell.c[3] * x) / hv;
    const double f = 1.0 / (g * g), factor = -2.0 / (g * g * g);
    return {f, factor * gu / (re * ln10 * u_span),
            factor * gv / ((rr_scale + rr) * v_span)};
}

__device__ Result interpolate_bicubic(double re, double rr, const double* u_axis,
                                       const double* v_axis, int nu, int nv,
                                       const CubicCell* cells, double u_span,
                                       double v_span, double rr_scale) {
    const double u = log10(re / 2000.0) / u_span;
    const double v = log1p(rr / rr_scale) / v_span;
    const int i = interval(u_axis, nu, u), j = interval(v_axis, nv, v);
    const double hu = u_axis[i + 1] - u_axis[i];
    const double hv = v_axis[j + 1] - v_axis[j];
    const double x = (u - u_axis[i]) / hu;
    const double y = (v - v_axis[j]) / hv;
    const CubicCell& cell = cells[static_cast<std::size_t>(i) * (nv - 1) + j];
    double row[4], row_dv[4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const double c0 = cell.c[a * 4], c1 = cell.c[a * 4 + 1];
        const double c2 = cell.c[a * 4 + 2], c3 = cell.c[a * 4 + 3];
        row[a] = ((c3 * y + c2) * y + c1) * y + c0;
        row_dv[a] = ((3.0 * c3 * y + 2.0 * c2) * y + c1) / hv;
    }
    const double g = ((row[3] * x + row[2]) * x + row[1]) * x + row[0];
    const double gu = ((3.0 * row[3] * x + 2.0 * row[2]) * x + row[1]) / hu;
    const double gv = ((row_dv[3] * x + row_dv[2]) * x + row_dv[1]) * x + row_dv[0];
    const double f = 1.0 / (g * g), factor = -2.0 / (g * g * g);
    return {f, factor * gu / (re * ln10 * u_span),
            factor * gv / ((rr_scale + rr) * v_span)};
}

__global__ void direct_kernel(const Point* points, Result* out, std::size_t n) {
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < n) out[k] = colebrook(points[k].re, points[k].rr);
}

__global__ void bilinear_kernel(const Point* points, Result* out, std::size_t n,
                                const double* u, const double* v, int nu, int nv,
                                const Cell* cells, double u_span, double v_span,
                                double rr_scale) {
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < n) out[k] = interpolate(points[k].re, points[k].rr, u, v, nu, nv,
                                    cells, u_span, v_span, rr_scale);
}

__global__ void bicubic_kernel(const Point* points, Result* out, std::size_t n,
                               const double* u, const double* v, int nu, int nv,
                               const CubicCell* cells, double u_span,
                               double v_span, double rr_scale) {
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < n) out[k] = interpolate_bicubic(points[k].re, points[k].rr, u, v, nu, nv,
                                             cells, u_span, v_span, rr_scale);
}

Error compare(const std::vector<Result>& exact, const std::vector<Result>& approx,
              const std::vector<Point>& points) {
    constexpr double tol[3] = {2.0e-4, 2.0e-2, 2.0e-3};
    Error e{};
    for (std::size_t k = 0; k < count; ++k) {
        const double scale = std::max(std::abs(exact[k].value), 1.0e-300);
        const double ev = std::abs(approx[k].value - exact[k].value) / scale;
        const double er = points[k].re * std::abs(approx[k].d_re - exact[k].d_re) / scale;
        const double err = (1.0e-5 + points[k].rr) *
            std::abs(approx[k].d_rr - exact[k].d_rr) / scale;
        e.value = std::max(e.value, ev);
        e.dre = std::max(e.dre, er);
        e.drr = std::max(e.drr, err);
        e.fail_value += !std::isfinite(ev) || ev > tol[0];
        e.fail_dre += !std::isfinite(er) || er > tol[1];
        e.fail_drr += !std::isfinite(err) || err > tol[2];
    }
    return e;
}

template<class Launch>
double time_kernel(Launch launch, const char* label) {
    launch(); check(cudaGetLastError(), label); check(cudaDeviceSynchronize(), label);
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");
    check(cudaEventRecord(start), "event start");
    for (int k = 0; k < repetitions; ++k) launch();
    check(cudaGetLastError(), label);
    check(cudaEventRecord(stop), "event stop");
    check(cudaEventSynchronize(stop), "event wait");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return static_cast<double>(count) * repetitions / (ms * 1.0e-3) / 1.0e6;
}

template<class Launch>
double time_e2e(Launch launch, Point* d_points,
                const std::vector<Point>& points, Result* d_out,
                std::vector<Result>& output, const char* label) {
    const auto start = Clock::now();
    for (int k = 0; k < repetitions; ++k) {
        check(cudaMemcpy(d_points, points.data(), count * sizeof(Point),
                         cudaMemcpyHostToDevice), label);
        launch(); check(cudaGetLastError(), label);
        check(cudaMemcpy(output.data(), d_out, count * sizeof(Result),
                         cudaMemcpyDeviceToHost), label);
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return static_cast<double>(count) * repetitions / seconds / 1.0e6;
}

double checksum(const std::vector<Result>& values) {
    double sum = 0.0;
    for (const auto& x : values) sum += x.value + 1.0e-3 * x.d_re + 1.0e-6 * x.d_rr;
    return sum;
}
} // namespace

int main() {
    Config cfg;
    Table table(cfg);
    const auto build_start = Clock::now();
    const auto report = table.build();
    const double bilinear_build_ms = std::chrono::duration<double, std::milli>(Clock::now() - build_start).count();
    const int nu = static_cast<int>(table.u_nodes().size());
    const int nv = static_cast<int>(table.v_nodes().size());
    std::vector<Cell> cells((nu - 1) * (nv - 1));
    for (std::size_t i = 0; i < cells.size(); ++i)
        for (int k = 0; k < 4; ++k)
            cells[i].c[k] = table.cell_polynomials()[i].coefficients[k];

    BicubicTable bicubic_table(cfg);
    const auto bicubic_build_start = Clock::now();
    const auto bicubic_report = bicubic_table.build();
    const double bicubic_build_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - bicubic_build_start).count();
    const int cubic_nu = static_cast<int>(bicubic_table.u_nodes().size());
    const int cubic_nv = static_cast<int>(bicubic_table.v_nodes().size());
    std::vector<CubicCell> cubic_cells((cubic_nu - 1) * (cubic_nv - 1));
    for (std::size_t i = 0; i < cubic_cells.size(); ++i)
        for (int k = 0; k < 16; ++k)
            cubic_cells[i].c[k] = bicubic_table.cell_polynomials()[i].coefficients[k];

    const double u_span = std::log10(cfg.reynolds_max / cfg.reynolds_min);
    const double v_span = std::log1p(cfg.relative_roughness_max / cfg.roughness_scale);
    std::vector<Point> points(count);
    std::vector<Result> exact(count), output(count);
    std::mt19937_64 rng(20261020);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (std::size_t k = 0; k < count; ++k) {
        const double u = unit(rng), v = unit(rng);
        points[k] = {cfg.reynolds_min * std::pow(cfg.reynolds_max / cfg.reynolds_min, u),
                     cfg.roughness_scale * std::expm1(v * v_span)};
        const auto d = friction::analytic::darcy_friction_factor(
            points[k].re, points[k].rr, friction::Correlation::colebrook);
        exact[k] = {d.value, d.d_reynolds, d.d_relative_roughness};
    }

    int device = 0; check(cudaGetDevice(&device), "get device");
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, device), "get properties");
    std::cout << "GPU: " << prop.name << " (compute " << prop.major << '.' << prop.minor << ")\n"
              << "Adaptive bilinear and bicubic tensor comparison; inputs=" << count
              << "; repetitions=" << repetitions << "\n"
              << "Bilinear grid=" << nu << 'x' << nv << " (" << report.grid_points
              << " nodes), rounds=" << report.refinement_rounds << ", build="
              << std::fixed << std::setprecision(2) << bilinear_build_ms << " ms, coefficients="
              << (cells.size() * sizeof(Cell) / 1048576.0) << " MiB; bicubic grid="
              << cubic_nu << 'x' << cubic_nv << " (" << bicubic_report.grid_points
              << " nodes), build=" << bicubic_build_ms << " ms, coefficients="
              << (cubic_cells.size() * sizeof(CubicCell) / 1048576.0) << " MiB\n"
              << "Scaled error limits (value, Re slope, roughness slope)=2e-4, 2e-2, 2e-3\n";

    Point* d_points = nullptr; Result* d_out = nullptr;
    double *d_u = nullptr, *d_v = nullptr, *d_cubic_u = nullptr, *d_cubic_v = nullptr;
    Cell* d_cells = nullptr; CubicCell* d_cubic_cells = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&d_points), count * sizeof(Point)), "alloc points");
    check(cudaMalloc(reinterpret_cast<void**>(&d_out), count * sizeof(Result)), "alloc output");
    check(cudaMalloc(reinterpret_cast<void**>(&d_u), nu * sizeof(double)), "alloc u");
    check(cudaMalloc(reinterpret_cast<void**>(&d_v), nv * sizeof(double)), "alloc v");
    check(cudaMalloc(reinterpret_cast<void**>(&d_cells), cells.size() * sizeof(Cell)), "alloc cells");
    check(cudaMalloc(reinterpret_cast<void**>(&d_cubic_u), cubic_nu * sizeof(double)), "alloc bicubic u");
    check(cudaMalloc(reinterpret_cast<void**>(&d_cubic_v), cubic_nv * sizeof(double)), "alloc bicubic v");
    check(cudaMalloc(reinterpret_cast<void**>(&d_cubic_cells),
                     cubic_cells.size() * sizeof(CubicCell)), "alloc bicubic cells");
    const auto upload_start = Clock::now();
    check(cudaMemcpy(d_u, table.u_nodes().data(), nu * sizeof(double), cudaMemcpyHostToDevice), "upload u");
    check(cudaMemcpy(d_v, table.v_nodes().data(), nv * sizeof(double), cudaMemcpyHostToDevice), "upload v");
    check(cudaMemcpy(d_cells, cells.data(), cells.size() * sizeof(Cell), cudaMemcpyHostToDevice), "upload cells");
    check(cudaMemcpy(d_cubic_u, bicubic_table.u_nodes().data(), cubic_nu * sizeof(double), cudaMemcpyHostToDevice), "upload bicubic u");
    check(cudaMemcpy(d_cubic_v, bicubic_table.v_nodes().data(), cubic_nv * sizeof(double), cudaMemcpyHostToDevice), "upload bicubic v");
    check(cudaMemcpy(d_cubic_cells, cubic_cells.data(), cubic_cells.size() * sizeof(CubicCell), cudaMemcpyHostToDevice), "upload bicubic cells");
    check(cudaMemcpy(d_points, points.data(), count * sizeof(Point), cudaMemcpyHostToDevice), "upload points");
    const double upload_ms = std::chrono::duration<double, std::milli>(Clock::now() - upload_start).count();
    std::cout << "Initial upload=" << upload_ms
              << " ms (separate setup cost; E2E below includes point and result copies).\n";

    const dim3 block(block_size), grid(static_cast<unsigned>((count + block_size - 1) / block_size));
    const auto direct_launch = [&] { direct_kernel<<<grid, block>>>(d_points, d_out, count); };
    const auto bilinear_launch = [&] {
        bilinear_kernel<<<grid, block>>>(d_points, d_out, count, d_u, d_v, nu, nv,
                                         d_cells, u_span, v_span, cfg.roughness_scale);
    };
    const auto bicubic_launch = [&] {
        bicubic_kernel<<<grid, block>>>(d_points, d_out, count, d_cubic_u, d_cubic_v,
            cubic_nu, cubic_nv, d_cubic_cells, u_span, v_span, cfg.roughness_scale);
    };
    const double direct_mps = time_kernel(direct_launch, "direct kernel");
    const double direct_e2e_mps = time_e2e(direct_launch, d_points, points, d_out,
                                            output, "direct E2E transfer");
    std::cout << "Direct Colebrook         kernel=" << std::fixed << std::setprecision(1)
              << direct_mps << " M/s E2E=" << direct_e2e_mps << " M/s\n";
    const double bilinear_mps = time_kernel(bilinear_launch, "bilinear kernel");
    check(cudaMemcpy(output.data(), d_out, count * sizeof(Result), cudaMemcpyDeviceToHost), "copy output");
    const Error bilinear_error = compare(exact, output, points);
    const double bilinear_e2e_mps = time_e2e(bilinear_launch, d_points, points, d_out,
                                              output, "bilinear E2E transfer");
    std::cout << "Adaptive bilinear       kernel=" << std::fixed << std::setprecision(1)
              << bilinear_mps << " M/s E2E=" << bilinear_e2e_mps << " M/s"
              << " scaled max errors=" << std::scientific << bilinear_error.value << '/'
              << bilinear_error.dre << '/' << bilinear_error.drr
              << " failures=" << bilinear_error.fail_value << '/' << bilinear_error.fail_dre
              << '/' << bilinear_error.fail_drr
              << " checksum=" << std::fixed << std::setprecision(9) << checksum(output) << '\n';

    const double bicubic_mps = time_kernel(bicubic_launch, "bicubic kernel");
    check(cudaMemcpy(output.data(), d_out, count * sizeof(Result), cudaMemcpyDeviceToHost), "copy bicubic output");
    const Error bicubic_error = compare(exact, output, points);
    const double bicubic_e2e_mps = time_e2e(bicubic_launch, d_points, points, d_out,
                                             output, "bicubic E2E transfer");
    std::cout << "Adaptive bicubic        kernel=" << std::fixed << std::setprecision(1)
              << bicubic_mps << " M/s E2E=" << bicubic_e2e_mps << " M/s"
              << " scaled max errors=" << std::scientific << bicubic_error.value << '/'
              << bicubic_error.dre << '/' << bicubic_error.drr
              << " failures=" << bicubic_error.fail_value << '/' << bicubic_error.fail_dre
              << '/' << bicubic_error.fail_drr
              << " checksum=" << std::fixed << std::setprecision(9) << checksum(output) << '\n';

    cudaFree(d_points); cudaFree(d_out); cudaFree(d_u); cudaFree(d_v); cudaFree(d_cells);
    cudaFree(d_cubic_u); cudaFree(d_cubic_v); cudaFree(d_cubic_cells);
    return EXIT_SUCCESS;
}
