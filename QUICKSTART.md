# Quickstart

FrictionLibraryCpp20 is a header-only C++20 library. It requires CMake 3.20 or later and a C++20 compiler (GCC 10+, Clang 12+, or a compatible compiler). The library itself has no third-party dependencies. It is licensed under AGPL-3.0-only; see [LICENSE](LICENSE) and [License](README.md#license) for details.

## Build, test, and run the example

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/friction_example
```

Benchmarks are disabled by default. To build the CPU benchmarks:

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DFRICTION_BUILD_BENCHMARKS=ON
cmake --build build-bench --parallel
./build-bench/friction_primal_benchmark
./build-bench/friction_batch_benchmark
./build-bench/colebrook_solver_benchmark
./build-bench/friction_surrogate_benchmark
```

Benchmark results depend on the CPU, compiler, build flags, and system load. They describe the tested machine and are not performance guarantees.

To run the comparison suite and generate a Markdown report with CPU results, run the tests, and save raw benchmark logs:

```sh
python3 scripts/run_performance_suite.py --runs 7
```

On a machine with the CUDA Toolkit and an NVIDIA GPU, include direct CUDA solvers and adaptive 2D interpolation:

```sh
python3 scripts/run_performance_suite.py --cuda --cuda-arch 52 \
  --runs 7 --gpu-runs 3
```

Change `--cuda-arch` to match the installed GPU. The script writes `build-performance/performance-results.md` and keeps each run's complete output under `build-performance/performance-raw/`. The CUDA direct comparison includes Clamond, Swamee–Jain, fixed-point, and Newton primal evaluations; the adaptive CUDA comparison measures direct Colebrook, bilinear, and bicubic methods on the same batch. The fixed-point and Newton GPU kernels are checked against their CPU library implementations before their timings are reported.

## Install and test a downstream consumer

Install into a local staging directory:

```sh
cmake -S . -B build-package -DCMAKE_BUILD_TYPE=Release \
  -DFRICTION_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --install build-package --prefix "$PWD/stage"
cmake -S tests/consumer -B build-consumer \
  -DCMAKE_PREFIX_PATH="$PWD/stage"
cmake --build build-consumer --parallel
ctest --test-dir build-consumer --output-on-failure
```

A consumer links the exported header-only target:

```cmake
find_package(friction 1 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE friction::friction)
```

## Build a relocatable CPack package

CPack creates a relocatable tarball containing the installed headers, CMake package files, and license:

```sh
cmake -S . -B build-package -DCMAKE_BUILD_TYPE=Release \
  -DFRICTION_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --install build-package --prefix "$PWD/stage"
cpack --config build-package/CPackConfig.cmake -G TGZ
```

The tarball is written under `build-package/`. Extract it under a chosen prefix and set `CMAKE_PREFIX_PATH` to that prefix when configuring the consumer project.

## Optional CUDA benchmark

The CUDA benchmarks require the NVIDIA CUDA Toolkit and a compatible GPU. Set the architecture to match the installed GPU and toolkit; `52` targets the Quadro M3000M used for the recorded experiments.

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DFRICTION_BUILD_CUDA_BENCHMARK=ON \
  -DCMAKE_CUDA_ARCHITECTURES=52
cmake --build build-cuda --target friction_cuda_benchmark --parallel
./build-cuda/friction_cuda_benchmark
```

The program compares CPU and GPU values and derivatives and reports kernel-only and transfer-inclusive timing. Small batches are generally dominated by launch and transfer overhead. Adaptive interpolation targets and their accuracy limits are documented in [the experiment notes](docs/experiments/README.md).
