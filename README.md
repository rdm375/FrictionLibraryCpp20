# Darcy friction factor library (C++20)

A header-only C++20 library for Darcy-Weisbach pipe friction factors. It currently includes:

- Clamond's two-update solver for the implicit Colebrook-White correlation, based on [arXiv:0810.5564](https://arxiv.org/abs/0810.5564).
- The explicit Swamee-Jain turbulent-flow approximation.
- An all-regime wrapper with laminar flow and a smooth transition blend.

All roughness inputs are relative roughness, \(k_s/D\), and results are Darcy friction factors (not Fanning factors).

Start with [QUICKSTART.md](QUICKSTART.md) for build, test, install, packaging, and downstream-consumer commands.

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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/friction_example
```

Tests compare Clamond against an independent bisection solution, verify Swamee-Jain's equation and its agreement with Colebrook over a representative range, exercise both transition branches and endpoints, and check invalid inputs.

`friction::analytic::darcy_friction_factor` returns the value and first derivatives with respect to Reynolds number and relative roughness. Its Colebrook derivatives use implicit differentiation of the Colebrook equation; the Swamee-Jain derivatives are differentiated in closed form. With `FRICTION_ENABLE_ENZYME_AD=ON`, the Enzyme test also cross-checks those derivatives against LLVM-generated reverse-mode derivatives.

Include `<friction/batch.hpp>` for the batch APIs. They accept equal-length `std::span`s for Reynolds number, relative roughness, and output; input values are validated before output is written. The batch functions are serial and allocation-free. They reject output spans that overlap either input span. For multicore work, split the input and output spans into disjoint ranges and call them from separate threads.

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

The solver benchmark compares the production simplified Colebrook derivative algebra and exp2(0.9*log2(Re)) evaluation against direct implicit-derivative and std::pow baselines. The faster formulation depends on compiler and CPU; build and run the benchmark on the deployment target before choosing a path. Results are not portable performance guarantees. The CUDA kernel also uses exp2/log2; CPU/GPU results are checked against tolerances by the CUDA benchmark.

## License and commercial use

This project is licensed under **GNU Affero General Public License version 3 only (AGPL-3.0-only)**. See [LICENSE](LICENSE) for the complete license text. Copyright: © 2026 FrictionLibraryCpp20 contributors.

AGPL copyleft may require source code for covered modified or combined works to be made available under the license terms, including the network-use provision in section 13. The exact effect depends on how a product uses and distributes this header-only library.

AGPL does **not** require royalties, a revenue share, or payment to the copyright holder. Organizations that need to keep a covered use proprietary can seek a separate commercial license from the copyright holder. That requires a separate agreement; this repository does not currently offer commercial terms or promise a particular fee. A commercial license can be granted only by the owner(s) of the relevant rights, so confirm ownership and contributor rights before offering one.

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

Set `CMAKE_CUDA_ARCHITECTURES` to the compute capability supported by the installed toolkit and GPU (for example, `52` for a device reporting compute capability 5.2). The program prints the detected GPU and capability. It uses ordinary IEEE math calls rather than CUDA fast-math flags. The elementwise comparison allows an absolute error of `2e-14` plus a relative error of `2e-11`; small last-bit differences can result from CPU and GPU math libraries.

It uses a single large input batch and reports throughput including thread startup for the multithreaded cases. Results depend on CPU, compiler, and math-library vectorization; the batch interface enables easy chunking but does not promise SIMD or a particular scaling factor.
