# FrictionLibraryCpp20

[![CI](https://github.com/rdm375/FrictionLibraryCpp20/actions/workflows/ci.yml/badge.svg)](https://github.com/rdm375/FrictionLibraryCpp20/actions/workflows/ci.yml)

A header-only C++20 library for Darcy–Weisbach pipe friction factors. It provides implementations for turbulent flow, an explicit laminar and transition wrapper, analytic first derivatives, and allocation-free batch evaluation.

## Features

- **Colebrook–White:** Clamond's specialized two-update solver, based on [Clamond (2009)](https://arxiv.org/abs/0810.5564), plus fixed-point and Newton solvers for comparison.
- **Swamee–Jain:** explicit turbulent-flow approximation.
- **All-regime API:** laminar `64/Re` behavior and a cubic smoothstep transition into the selected turbulent correlation.
- **Derivatives:** closed-form first derivatives with respect to Reynolds number and relative roughness, plus optional reverse-mode checks using Enzyme and LLVM.
- **Batch API:** validated, allocation-free `std::span` interfaces for scalar and derivative batches.
- **Experiments:** optional CPU and CUDA benchmarks, including adaptive one- and two-dimensional tensor interpolants. These are exploratory tools; their measured limits are documented separately.

Inputs use relative roughness `k_s/D`. Returned factors are Darcy factors, not Fanning factors.

## Quick start

Requirements: CMake 3.20 or later and a C++20 compiler. The library has no third-party runtime dependencies.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/friction_example
```

See [QUICKSTART.md](QUICKSTART.md) for installation, downstream use, CPack, and optional benchmark instructions.
See [CHANGELOG.md](CHANGELOG.md) for changes planned for the next release.

## Example

```cpp
#include <friction/analytic_derivatives.hpp>
#include <friction/friction.hpp>

int main() {
    const double re = 700'000.0;
    const double relative_roughness = 0.01;

    const double f = friction::darcy_friction_factor(
        re, relative_roughness, friction::Correlation::colebrook);
    const auto d = friction::analytic::darcy_friction_factor(
        re, relative_roughness, friction::Correlation::colebrook);

    // f and d.value are equal; d contains both first derivatives.
    return (f > 0.0 && d.d_reynolds < 0.0) ? 0 : 1;
}
```

The scalar derivative result contains `value`, `d_reynolds`, and `d_relative_roughness`. Include `<friction/batch.hpp>` for the span-based batch APIs. Inputs must have equal lengths and valid values; batch functions validate the complete input before writing outputs. Output spans must not overlap input spans. Calls are serial and allocation-free; applications can parallelize by passing disjoint spans to separate threads.

## Correlation ranges and transition convention

The turbulent Colebrook solver accepts finite `Re >= 2000`; the standalone Swamee–Jain function accepts finite `Re >= 5000`. Swamee–Jain is commonly used over approximately `5000 <= Re <= 1e8`; using it outside that range is extrapolation.

The all-regime wrapper applies `64/Re` below `Re=2000`, then uses a cubic smoothstep to blend into the selected turbulent correlation. The blend ends at `Re=2300` for Colebrook and `Re=5000` for Swamee–Jain. This is an explicit engineering convention, not a universal transitional-flow model. The smoothstep makes the friction factor and its first derivative continuous at the blend endpoints.

## Optional Enzyme derivatives

The default derivative API uses analytic formulas. To build the optional Enzyme-generated derivative implementation, use Clang and the matching ClangEnzyme plugin:

```sh
cmake -S . -B build-enzyme -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++-21 \
  -DFRICTION_ENABLE_ENZYME_AD=ON \
  -DFRICTION_ENZYME_PLUGIN=/path/to/ClangEnzyme-21.so
cmake --build build-enzyme --parallel
ctest --test-dir build-enzyme --output-on-failure
```

The test compares generated derivatives against the analytic results. In a project configured and installed with Enzyme enabled, link the optional `friction::friction_enzyme_ad` target to use the generated functions declared in `<friction/enzyme_derivatives.hpp>`. The Enzyme build is optional and requires a compatible plugin; it is not needed to use the library.

## Solver methods and performance

The library includes several ways to obtain a turbulent Darcy friction factor:

- **Clamond Colebrook solver (`colebrook_turbulent`):** a specialized two-update solution to the implicit Colebrook–White equation, based on [Clamond (2009)](https://arxiv.org/abs/0810.5564). This is the library's default direct Colebrook method.
- **Fixed-point Colebrook (`colebrook_fixed_point`):** repeatedly substitutes the current estimate into the Colebrook equation until convergence. It is straightforward and useful as an independent reference, but took more iterations and ran slower in the measured CPU comparison.
- **Newton Colebrook (`colebrook_newton`):** applies Newton's method to the Colebrook residual in the variable `1/sqrt(f)`. It converged in fewer iterations than fixed point in the recorded run, but remained slower than Clamond.
- **Swamee–Jain (`swamee_jain_turbulent`):** an explicit turbulent-flow correlation. It avoids solving the implicit Colebrook equation and was the fastest direct primal correlation in the recorded CPU results. It is an approximation, not another solver for the Colebrook equation.
- **Adaptive bilinear and bicubic tensor interpolants:** prebuilt approximations over a configured Reynolds-number/relative-roughness domain. They trade setup time and coefficient storage for faster repeated GPU evaluation. Their accuracy, especially for derivatives, must be checked for the intended use.

The direct analytic derivative API returns the factor and its first derivatives with respect to Reynolds number and relative roughness. The derivative rows below time those three outputs together; the iterative solver rows time the primal factor only.

### Generating and publishing performance results

Run the suite on the machine whose results you want to report:

```sh
python3 scripts/run_performance_suite.py --cuda --cuda-arch 52 --runs 7 --gpu-runs 3
```

Omit `--cuda` on systems without the CUDA Toolkit and a supported NVIDIA GPU. The script writes a full report and raw logs into the selected build directory (default: `build-performance/`) and replaces the generated-results block below in this README. Commit the README update only after reviewing the machine, compiler, test results, and accuracy checks in the report. To insert a report you already generated without rerunning benchmarks, use `python3 scripts/run_performance_suite.py --insert-results build-performance-v2/performance-results.md` (an absolute path also works).

### Latest benchmark run

<!-- BEGIN GENERATED PERFORMANCE RESULTS -->

Generated: Linux-7.0.0-38-generic-x86_64-with-glibc2.43
CPU: Intel(R) Core(TM) i7-6920HQ CPU @ 2.90GHz
C++ compiler: c++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0
CPU benchmark repetitions: 7 process runs; CPU rows show median (min–max) throughput.
CPU compiler flags: CMake Release plus `-O3 -march=native` for benchmark targets.
Tests:

```text
Test project /nvme/Sync/FrictionLibraryCpp20/build-performance-v2
    Start 1: friction
1/6 Test #1: friction .............................   Passed    0.00 sec
    Start 2: analytic_derivatives
2/6 Test #2: analytic_derivatives .................   Passed    0.00 sec
    Start 3: batch
3/6 Test #3: batch ................................   Passed    0.00 sec
    Start 4: colebrook_methods
4/6 Test #4: colebrook_methods ....................   Passed    0.00 sec
    Start 5: adaptive_tensor_surrogate
5/6 Test #5: adaptive_tensor_surrogate ............   Passed    0.23 sec
    Start 6: adaptive_bilinear_tensor_surrogate
6/6 Test #6: adaptive_bilinear_tensor_surrogate ...   Passed    0.23 sec

100% tests passed, 0 tests failed out of 6

Total Test time (real) =   0.47 sec
```

### CPU direct methods

Primal rows are friction-factor solves/evaluations; derivative rows compute the primal plus both first derivatives.

| Method | CPU M/s (median; range) | Outputs |
|---|---|---|
| Swamee–Jain | 19.71 (19.58–19.83) | Primal explicit correlation |
| Swamee–Jain + derivatives | 17.91 (17.75–18.02) | Primal + `df/dRe` + `df/d(k_s/D)` |
| Clamond Colebrook | 13.91 (13.83–14.15) | Primal; 2 updates |
| Newton Colebrook | 13.10 (13.01–13.18) | Primal; converged solve |
| Clamond Colebrook + derivatives | 11.73 (11.60–11.77) | Primal + `df/dRe` + `df/d(k_s/D)` |
| Fixed-point Colebrook | 7.09 (7.08–7.12) | Primal; converged solve |

All direct CPU methods use the same deterministic 8,192-point turbulent input set and benchmark executable. It performs 250 passes per method. Solver results also report iteration counts and checksums in the raw logs.

### CPU adaptive 2D interpolants

All three methods evaluate the same 1,048,576 deterministic `(Re, k_s/D)` inputs and return the factor and both derivatives. Each executable run reports the median of five passes; the table below reports the median (range) across process runs.

| Method | CPU M eval/s (median; range) |
|---|---|
| Direct Colebrook + gradients | 11.69 (11.35–11.97) |
| Adaptive bicubic + gradients | 8.69 (6.68–9.99) |
| Adaptive bilinear + gradients | 4.85 (4.83–4.88) |

Adaptive bilinear tensor: 94556 nodes (307 x 308), 6.08 MiB, 12 refinement rounds, build=0.215 s
Adaptive tensor table: 94556 nodes (307 x 308), 15.78 MiB, 12 refinement rounds, build=0.207 s, final indicator=0.000

### GPU direct methods

GPU: Quadro M3000M (compute 5.2)
Each row is the 1,048,576-element case from the batch-size sweep, summarized across 3 process runs. CPU and GPU timings are from the same benchmark executable. Kernel timing keeps inputs resident; `GPU + copies` transfers both inputs and output for each pass.

| Method | CPU M/s | GPU kernel M/s | GPU + copies M/s |
|---|---|---|---|
| Swamee–Jain, primal | 19.28 (19.09–19.39) | 346.70 (346.67–346.84) | 136.24 (132.55–137.72) |
| Swamee–Jain + derivatives | 18.74 (17.99–18.81) | 269.76 (269.74–269.80) | 123.71 (123.55–124.22) |
| Clamond Colebrook, primal | 13.89 (13.88–13.91) | 198.31 (198.27–198.32) | 106.57 (106.51–106.62) |
| Colebrook Newton, primal | 12.50 (12.41–12.55) | 196.35 (196.34–196.49) | 105.25 (104.98–105.71) |
| Colebrook fixed point, primal | 7.14 (7.14–7.16) | 147.75 (147.74–147.81) | 90.00 (89.96–90.13) |
| Clamond Colebrook + derivatives | 11.68 (11.62–11.72) | 129.00 (129.00–129.03) | 82.73 (82.37–82.75) |

Newton and fixed-point GPU kernels are direct ports of the library's converged residual iterations. Their GPU primal results are checked elementwise against the production CPU implementations. Derivatives are not currently benchmarked for these iterative variants.

### GPU adaptive 2D interpolants

GPU: Quadro M3000M (compute 5.2)
The adaptive CUDA benchmark uses a shared input batch and compares the direct Colebrook reference with bilinear and bicubic interpolation over 3 process runs. Rates are medians (min–max); accuracy and setup details are recorded in the raw logs.

| Method | GPU kernel M/s | GPU + copies M/s | Max scaled errors (value/Re/roughness) | Failed checks |
|---|---|---|---|---|
| Adaptive bilinear | 172.00 (166.60–172.00) | 99.00 (95.40–99.00) | 1.7e-04/2.6e-01/2.6e-02 | 0/9425/11393 |
| Adaptive bicubic | 134.20 (134.10–134.20) | 85.20 (82.30–85.30) | 6.2e-06/1.8e-02/1.7e-05 | 0/0/0 |
| Direct Colebrook | 131.80 (111.40–131.90) | 83.10 (79.20–84.10) | reference | 0 |

Scaled error limits (value, Re slope, roughness slope)=2e-4, 2e-2, 2e-3
Bilinear grid=307x308 (94556 nodes), rounds=12, build=254.96 ms, coefficients=2.87 MiB; bicubic grid=307x308 (94556 nodes), build=232.67 ms, coefficients=11.47 MiB
Initial upload=3.56 ms (separate setup cost; E2E below includes point and result copies).

Raw CUDA logs include error maxima, tolerance failures, checksums, and every batch size:

Raw logs: `build-performance-v2/performance-raw/`

<!-- END GENERATED PERFORMANCE RESULTS -->

## License

The project is licensed under **GNU Affero General Public License version 3 only (AGPL-3.0-only)**; see [LICENSE](LICENSE). If you need different commercial terms, contact the rights holders to discuss a separate written agreement. This repository does not grant a commercial license or specify a fee or royalty schedule; any such terms must be agreed separately by the relevant rights holders. See [COPYRIGHT](COPYRIGHT) for the project-level rights notice.
