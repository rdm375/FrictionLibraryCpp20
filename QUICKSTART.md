# Quickstart

FrictionLibraryCpp20 is a header-only C++20 library. You need CMake 3.20 or
newer and a C++20 compiler (GCC 10+, Clang 12+, or a compatible compiler).
No third-party dependency is needed for the library itself. The project is licensed under AGPL-3.0-only; see [LICENSE](LICENSE) and the README section on commercial use.

## Build, test, and run the example

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/friction_example
```

The quickstart keeps benchmarks out of the normal build. To build and run the
CPU benchmarks:

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DFRICTION_BUILD_BENCHMARKS=ON
cmake --build build-bench --parallel
./build-bench/friction_primal_benchmark
./build-bench/friction_batch_benchmark
./build-bench/colebrook_solver_benchmark
```

Benchmark numbers depend on the CPU, compiler, flags, and system load. They are
measurements for the selected machine, not performance guarantees.

## Use the installed CMake package

Install to a local staging directory:

```sh
cmake -S . -B build-package -DCMAKE_BUILD_TYPE=Release \
  -DFRICTION_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build build-package --parallel
cmake --install build-package --prefix "$PWD/stage"
```

In a consumer project, use:

```cmake
find_package(friction 1 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE friction::friction)
```

Configure that project with the stage prefix, for example:

```sh
cmake -S consumer -B consumer/build \
  -DCMAKE_PREFIX_PATH="$PWD/stage"
cmake --build consumer/build --parallel
```

## Create a binary package

CPack creates a relocatable tarball containing the installed headers and CMake
package files:

```sh
cmake -S . -B build-package -DCMAKE_BUILD_TYPE=Release \
  -DFRICTION_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build build-package --parallel
cpack --config build-package/CPackConfig.cmake -G TGZ
```

The archive is written to the repository root and includes the AGPL license text. Extract it under a chosen prefix, then set `CMAKE_PREFIX_PATH` to that prefix in the consumer build.

## Optional CUDA benchmark

This target requires the NVIDIA CUDA Toolkit and a compatible GPU. Set the
architecture to match the device; `52` is for the Quadro M3000M:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DFRICTION_BUILD_CUDA_BENCHMARK=ON \
  -DCMAKE_CUDA_ARCHITECTURES=52
cmake --build build-cuda --target friction_cuda_benchmark --parallel
./build-cuda/friction_cuda_benchmark
```

The program compares CPU and GPU outputs and reports kernel-only and
transfer-inclusive timing. Small batches are generally dominated by GPU
launch and transfer overhead.
