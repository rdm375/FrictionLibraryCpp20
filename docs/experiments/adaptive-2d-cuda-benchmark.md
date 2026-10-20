# Adaptive 2D CUDA tensor benchmark

The benchmark compares the Taylor-disagreement adaptive tensor grid against a uniform tensor grid with the same number of nodes on each axis. Both use bicubic Hermite cell polynomials, the same transformed Reynolds and relative-roughness coordinates, and the same derivative calculation. It reports grid construction separately from GPU-resident and end-to-end evaluation, and validates on held-out points.

Build and run from the repository root:

```sh
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DFRICTION_BUILD_CUDA_BENCHMARK=ON \
  -DCMAKE_CUDA_ARCHITECTURES=52
cmake --build build-cuda --target friction_cuda_adaptive_tensor_benchmark --parallel
./build-cuda/friction_cuda_adaptive_tensor_benchmark
```

Use a CUDA architecture supported by your GPU and installed toolkit. The refinement indicator is a heuristic, not a formal error certificate. The run reports scaled errors and the number of held-out checks that exceed its stated limits. See the [experiment overview](README.md) for comparison results and interpretation.
