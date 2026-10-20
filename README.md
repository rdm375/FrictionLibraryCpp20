# Darcy friction factor library (C++20)

A header-only C++20 library for Darcy-Weisbach pipe friction factors. It currently includes:

- Clamond's two-update solver for the implicit Colebrook-White correlation, based on [arXiv:0810.5564](https://arxiv.org/abs/0810.5564).
- The explicit Swamee-Jain turbulent-flow approximation.
- An all-regime wrapper with laminar flow and a smooth transition blend.

All roughness inputs are relative roughness, \(k_s/D\), and results are Darcy friction factors (not Fanning factors).

## Correlations and API

```cpp
#include <friction/batch.hpp>
#include <friction/friction.hpp>

#include <vector>

const double f_cw = friction::colebrook_turbulent(700000.0, 0.01)
                        .darcy_friction_factor;
const double f_sj = friction::swamee_jain_turbulent(700000.0, 0.01);

const double f_all_regimes = friction::darcy_friction_factor(
    700000.0, 0.01, friction::Correlation::swamee_jain);

const auto gradient = friction::analytic::darcy_friction_factor(
    700000.0, 0.01, friction::Correlation::colebrook);
// gradient.value, gradient.d_reynolds, gradient.d_relative_roughness

std::vector<double> re{700000.0, 1.0e6};
std::vector<double> roughness{0.01, 0.001};
std::vector<friction::analytic::FirstDerivatives> gradients(re.size());
friction::batch::darcy_friction_factors_with_derivatives(
    re, roughness, gradients, friction::Correlation::colebrook);
```

`colebrook_turbulent` returns `ColebrookResult`, including the Darcy factor and iteration count. It accepts finite `Re >= 2000`, which also supports the transition blend. `swamee_jain_turbulent` returns a `double` and requires finite `Re >= 5000`. Swamee-Jain is commonly applied over approximately `5000 <= Re <= 1e8`; users should treat results outside a correlation's engineering range as extrapolation.

The Colebrook implementation solves the implicit Colebrook-White equation. `colebrook_turbulent` uses Clamond's specialized two-update method. The additional `colebrook_fixed_point` and `colebrook_newton` functions solve that same equation in (y=1/\sqrt{f_D}), iterate to a double-precision stopping criterion, and return the iteration count. They provide conventional iterative alternatives and useful cross-checks; use `colebrook_solver_benchmark` to compare their throughput and convergence on the target CPU.

`darcy_friction_factor` validates finite positive Reynolds number and finite nonnegative relative roughness, then applies the selected correlation as follows:

| Reynolds number | Colebrook selection | Swamee-Jain selection |
|---|---|---|
| `Re < 2000` | `64/Re` | `64/Re` |
| `2000 <= Re < 2300` | smoothstep blend to Colebrook | smoothstep blend to the Swamee-Jain formula |
| `2300 <= Re < 5000` | Colebrook | smoothstep blend to Swamee-Jain |
| `Re >= 5000` | Colebrook | Swamee-Jain |

The transition interval is an explicit engineering convention, not a universal transitional-flow law. A cubic smoothstep \(w(t)=3t^2-2t^3\) blends friction factors as \((1-w)f_{laminar}+wf_{turbulent}\). It gives a continuous first derivative at each blend endpoint. The Swamee-Jain formula is evaluated algebraically inside its blend below `Re=5000`; this is a transition interpolation convention, while the standalone turbulent API enforces the correlation's conventional lower bound.

## Use from another CMake project

After installing this project, a consumer can link the exported header-only target:

```cmake
find_package(friction 1 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE friction::friction)
```

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"
cmake --build build
ctest --test-dir build --output-on-failure
./build/friction_example
```

Tests compare Clamond against an independent bisection solution, verify Swamee-Jain's equation and its agreement with Colebrook over a representative range, exercise both transition branches and endpoints, and check invalid inputs.

`friction::analytic::darcy_friction_factor` returns the value and first derivatives with respect to Reynolds number and relative roughness. Its Colebrook derivatives use implicit differentiation of the Colebrook equation; the Swamee-Jain derivatives are differentiated in closed form. With `FRICTION_ENABLE_ENZYME_AD=ON`, the Enzyme test also cross-checks those derivatives against LLVM-generated reverse-mode derivatives.

Include `<friction/batch.hpp>` for the batch APIs. They accept equal-length `std::span`s for Reynolds number, relative roughness, and output; all inputs are validated before any outputs are written. The batch functions are serial and allocation-free. For multicore work, split the input and output spans into disjoint ranges and call them from separate threads. Output spans must not overlap either input span.

## Derivatives with Enzyme and LLVM

The first-order quantities needed by pipe-model Jacobians, tangent sensitivities, discrete adjoints, and Gauss-Newton updates are

\[
\frac{\partial f_D}{\partial Re},\qquad
\frac{\partial f_D}{\partial (k_s/D)}.
\]

The optional Enzyme target generates both partials in one reverse-mode pass for the scalar friction output. It covers the smooth turbulent Colebrook and Swamee-Jain kernels and the all-regime wrapper, including its laminar and blended branches. The wrapper's regime thresholds are piecewise; away from those thresholds the derivatives are ordinary derivatives, and the cubic blend makes the first derivative continuous at the joins. Exact second derivatives or Hessian-vector products are not included yet; they become useful if we move from Gauss-Newton to full Newton curvature.

Enzyme differentiates optimized LLVM/Clang code through its compiler plugin; this project uses the Clang plugin on the translation unit that contains the inline friction kernels. The benchmark and Enzyme derivative library compile at `-O3 -march=native` for the local CPU. For your own executable, use the same release flags if you want the compiler to target your CPU rather than a generic processor. To enable Enzyme, use the Clang version that matches the plugin and pass its plugin path:

```sh
cmake -S . -B build-enzyme -DCMAKE_BUILD_TYPE=Release \\
  -DCMAKE_CXX_COMPILER=clang++-21 \\
  -DFRICTION_ENABLE_ENZYME_AD=ON \\
  -DFRICTION_ENZYME_PLUGIN=/path/to/ClangEnzyme-21.so
cmake --build build-enzyme -j
ctest --test-dir build-enzyme --output-on-failure
./build-enzyme/friction_cpu_benchmark
```

The derivative validation compares Enzyme results with central finite differences. The CPU benchmark reports primal and primal-plus-gradient throughput for both correlations. The normal build also includes `friction_primal_benchmark`:

```sh
./build/friction_primal_benchmark
```

The batch benchmark compares one-thread and 2/4/8-thread throughput for primal and analytic-gradient batches:

```sh
./build/friction_batch_benchmark
```

Compare the implicit-equation solvers with:

```sh
./build/colebrook_solver_benchmark
```

This benchmark uses the same turbulent input grid for Clamond, fixed point, Newton, Swamee-Jain, and the two analytic-gradient APIs. Gradient rows return the primal value and both partial derivatives from one function call.

The solver benchmark includes CPU-specific A/B candidates for the Colebrook gradient algebra and for computing `Re^0.9` with `exp2(0.9*log2(Re))`. On the Dell i7-6920HQ run, those candidates were slower than the production formulas: 10.54 vs. 11.35 M Colebrook gradient evaluations/s, 18.67 vs. 19.55 M Swamee-Jain evaluations/s, and 16.83 vs. 17.93 M Swamee-Jain gradient evaluations/s. Production therefore uses the conventional implicit derivative and `std::pow`; the benchmark retains the alternatives for testing other compilers and CPUs. The CUDA kernel keeps the `exp2`/`log2` implementation on device, while its CPU reference uses `pow`.

## Optional CUDA throughput experiment

The CUDA benchmark compares a CPU reference with one-thread-per-input GPU kernels for Colebrook and Swamee-Jain, both primal-only and primal plus the two analytic first derivatives. It sweeps batches from 1 through 1,048,576 inputs and prints per-batch CPU, GPU kernel-only, and GPU transfer-inclusive latency. The transfer-inclusive result includes copies of both inputs and the output. It also compares every value and derivative against the CPU result, reports maximum absolute and relative error, and exits with failure if any component exceeds the stated tolerances. The CUDA target is opt-in and does not change the normal library build:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \\
  -DCMAKE_CUDA_COMPILER=nvcc \\
  -DFRICTION_BUILD_CUDA_BENCHMARK=ON \\
  -DCMAKE_CUDA_ARCHITECTURES=52
cmake --build build-cuda --target friction_cuda_benchmark -j
./build-cuda/friction_cuda_benchmark
```

To compare fixed-roughness Colebrook gradients with the prepared tensor slice on the GPU, build and run the separate 1M-input benchmark:

```sh
cmake --build build-cuda --target friction_cuda_surrogate_benchmark -j
./build-cuda/friction_cuda_surrogate_benchmark
```

For the fixed-roughness one-dimensional slice experiment, build and run:

```sh
cmake --build build-cuda --target friction_cuda_linear_slice_benchmark -j
./build-cuda/friction_cuda_linear_slice_benchmark
```

This benchmark adaptively builds a one-dimensional piecewise-linear Colebrook
slice in `log(Re)`. It uses disagreement between the endpoint Taylor predictions
and the interval interpolant, with direct midpoint checks for friction factor,
`df/dRe`, and `df/d(k_s/D)`, to decide which intervals to split. The GPU uses a
binary search over the resulting nonuniform knots, then evaluates the local
linear function and its derivative channels. It reports sample/build time and
coefficient upload separately from CPU/GPU evaluation, and checks all three
outputs against direct Colebrook calculations. The slice is for one fixed
relative roughness; rebuild it when that roughness changes. The roughness
derivative channel supports local sensitivity at the prepared roughness.

It reports CPU, resident GPU-kernel, and transfer-inclusive throughput. Transfer-inclusive prepared-slice timing includes uploading the slice for each roughness evaluation; table construction, roughness preparation, and initial axis upload are reported separately. The direct GPU Colebrook path matches the library's laminar branch and smooth transition from Re=2000 to 2300, so the direct and surrogate checks compare the same all-regime function. The executable checks GPU values and both derivatives against the CPU implementation.

Set `CMAKE_CUDA_ARCHITECTURES` to the compute capability supported by the installed toolkit and GPU (for example, `52` for a device reporting compute capability 5.2). The program prints the detected GPU and capability. It uses ordinary IEEE math calls rather than CUDA fast-math flags. The elementwise comparison allows an absolute error of `2e-14` plus a relative error of `2e-11`; small last-bit differences can result from CPU and GPU math libraries.

It uses a single large input batch and reports throughput including thread startup for the multithreaded cases. Results depend on CPU, compiler, and math-library vectorization; the batch interface enables easy chunking but does not promise SIMD or a particular scaling factor.

## Experimental adaptive tensor surrogate

`friction::surrogate::AdaptiveColebrookTensor` is an experimental two-input surrogate over Reynolds number and relative roughness. It interpolates `1/sqrt(f)` in normalized `log(Re)` and `log1p(relative_roughness / scale)` coordinates, returns the friction factor and both first derivatives, and falls back to the direct analytic correlation outside its configured domain. Sampling the all-regime Colebrook wrapper from `Re_min >= 2000` includes the current transition blend; inputs below the table range use the exact wrapper.

For workloads that evaluate many Reynolds numbers at one pipe roughness, prepare a roughness slice once and reuse it:

```cpp
const auto roughness = table.prepare_roughness(relative_roughness);
const auto result = roughness.evaluate(reynolds).derivatives;
```

The prepared evaluator caches the roughness coordinate and its derivative scale, then reduces each bicubic patch to a one-dimensional cubic in Reynolds number. It returns the same value and both physical derivatives as `evaluate(re, rr)`. Keep the source table alive and do not rebuild it while a prepared slice is in use.

For a full two-input GPU interpolant, build and run:

```sh
cmake --build build-cuda --target friction_cuda_bilinear_benchmark -j
./build-cuda/friction_cuda_bilinear_benchmark
```

This experiment samples the turbulent Colebrook surface on a `1025 x 1025` grid in `(log(Re), log1p(Re*rr/80))` coordinates, then converts the samples into cell-local bilinear polynomial coefficients. The roughness coordinate follows the changing roughness sensitivity as Reynolds number varies. The GPU evaluates this piecewise bilinear interpolant function and its two physical derivatives. It uses the exact library wrapper below `Re=2300` to retain the transition blend, then interpolates for `Re >= 2300`. Sample/coefficient construction and coefficient upload are timed separately from resident kernel and transfer-inclusive evaluation. Accuracy is checked against direct GPU Colebrook using per-output absolute-plus-relative tolerances, and the executable returns failure if any target is missed.

The builder begins with a 3-by-3 tensor grid, uses corner Taylor-model disagreement plus direct cell-center checks to mark intervals for refinement, and constructs bicubic Hermite patches. Mixed derivatives at nodes are estimated from the analytic first derivatives. The table uses adaptively placed axis knots but currently stores the full Cartesian tensor product; it is a validated baseline for a future sparse hierarchical representation. Grid generation is CPU-based. Its refinement indicator is not a certified error bound, so validate values and sensitivities on independent points in the operating domain.

Run `friction_surrogate_benchmark` to measure one-time table construction and roughness-slice preparation separately from evaluation. It reports medians of five passes for a seeded random `(Re, rr)` batch, the same Reynolds samples at fixed roughness, and the prepared roughness slice. The default grid has 94,556 nodes and uses about 15.78 MiB. Results are CPU-specific; use the benchmark on the target machine. The table currently serves as a roughness-calibration and interpolation experiment; the full tensor storage is the current baseline, with a genuinely sparse hierarchical representation and batched GPU generation as follow-up work.
