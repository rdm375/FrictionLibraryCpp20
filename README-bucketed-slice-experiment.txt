FrictionLibraryCpp20 CUDA bucketed-slice experiment

Overlay contents:
  benchmarks/friction_cuda_bucketed_slice_benchmark.cu

From the root of the FrictionLibraryCpp20 repository, build and run it with:

  mkdir -p build-cuda
  nvcc -std=c++17 -O3 -arch=sm_52 -Iinclude \
    benchmarks/friction_cuda_bucketed_slice_benchmark.cu \
    -o build-cuda/friction_cuda_bucketed_slice_benchmark
  ./build-cuda/friction_cuda_bucketed_slice_benchmark

Use an -arch value supported by your GPU/toolkit if it differs from sm_52.
The program compares direct Colebrook evaluation, a uniform linear slice, a
single globally adaptive slice, and adaptive slices grouped into coarse Re
buckets. It benchmarks random log-uniform and Re-sorted input streams. Kernel
rates keep data resident on the GPU; E2E rates include copying Re inputs and
results, but exclude sorting and one-time coefficient uploads.

This is a standalone benchmark source overlay; it does not change CMakeLists.txt.
