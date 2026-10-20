FrictionLibraryCpp20 adaptive 2D CUDA tensor benchmark

Overlay files:
  CMakeLists.txt
  include/friction/adaptive_tensor_surrogate.hpp
  benchmarks/friction_cuda_adaptive_tensor_benchmark.cu

The benchmark compares the existing Taylor-disagreement adaptive tensor grid
against a uniformly sampled tensor grid with the same number of nodes on each
axis. Both use the same bicubic Hermite cell polynomial, transformed Re and
roughness coordinates, and derivative calculation. Build and coefficient
construction are reported separately from runtime throughput. Runtime reports
GPU-resident kernel throughput and end-to-end throughput including point input
and result output copies. It also reports held-out error in friction factor
and scaled Re/roughness derivatives.

From the repository root, configure, build, and run:

  cmake -S . -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DFRICTION_BUILD_CUDA_BENCHMARK=ON \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=52
  cmake --build build-cuda --target friction_cuda_adaptive_tensor_benchmark -j"$(nproc)"
  ./build-cuda/friction_cuda_adaptive_tensor_benchmark

Change CUDA_ARCHITECTURES to match your GPU/toolkit if necessary. The adaptive
indicator is a refinement heuristic, not a formal error certificate. The
benchmark prints sampled-point errors and tolerance failures for both grids.
