Adaptive bilinear tensor experiment
===================================

This is a separate experimental implementation. It leaves the existing
adaptive bicubic tensor and its benchmark unchanged. The new CUDA benchmark
compares direct Colebrook, adaptive bilinear, and adaptive bicubic on one
deterministic input batch, with kernel and input/output-copy timings for each.

The bilinear table samples g = 1/sqrt(f) on a tensor grid in the same
normalized log(Re), log1p(relative roughness / scale) coordinates. It refines
the axes using direct cell-center defects and disagreement between corner
Taylor models. Each cell stores four coefficients. Evaluation returns f and
the derivatives implied by the bilinear polynomial.

Files are laid out relative to the project root. Extract them there, then run:

  python3 apply_adaptive_bilinear_cmake.py
  cmake -S . -B build-bilinear -DCMAKE_BUILD_TYPE=Release
  cmake --build build-bilinear -j"$(nproc)"
  ctest --test-dir build-bilinear --output-on-failure

The CPU test was compiled directly with g++ 13 using C++20 and passed. At the
default 2e-4 build indicator, its 37x29 held-out test reported maximum relative
value error 9.75e-5, scaled d/dlog(Re) error 0.176, and scaled roughness-slope
error 0.00665. The derivative errors exceed the tighter bicubic benchmark
limits (2e-2 and 2e-3); that is an expected limitation of first-order
interpolation and is reported rather than hidden.

To build and run on a CUDA host with an sm_52 GPU:

  cmake -S . -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DFRICTION_BUILD_CUDA_BENCHMARK=ON \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=52
  cmake --build build-cuda --target friction_cuda_adaptive_bilinear_benchmark -j"$(nproc)"
  ./build-cuda/friction_cuda_adaptive_bilinear_benchmark

The CUDA source was not compiled or run in the artifact environment (nvcc was
unavailable). It reports direct Colebrook and both interpolants' kernel and
input/output-copy throughput, table build and upload times, memory use, and
value/derivative errors on a deterministic million-point batch. The
bilinear benchmark reports derivative tolerance failures when present; it does
not silently fall back to direct evaluation inside the table domain.
