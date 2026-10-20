# CUDA bucketed-slice experiment

This standalone benchmark compares direct Colebrook evaluation, a uniform linear slice, a global adaptive slice, and adaptive slices grouped into coarse Reynolds-number buckets. It measures random log-uniform and Reynolds-sorted input streams. Kernel rates keep data resident on the GPU; end-to-end rates include Reynolds inputs and result copies, but exclude sorting and one-time coefficient uploads.

The current source file is `benchmarks/friction_cuda_bucketed_slice_benchmark.cu`. Build it with the CUDA compiler and an architecture supported by your GPU:

```sh
mkdir -p build-cuda
nvcc -std=c++17 -O3 -arch=sm_52 -Iinclude \
  benchmarks/friction_cuda_bucketed_slice_benchmark.cu \
  -o build-cuda/friction_cuda_bucketed_slice_benchmark
./build-cuda/friction_cuda_bucketed_slice_benchmark
```

The recorded M3000M run did not show a throughput advantage for bucketed adaptive sampling over uniform sampling. See the [experiment overview](README.md) for the measured rates and limits.
