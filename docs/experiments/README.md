# Experimental interpolation benchmarks

These benchmarks evaluate whether precomputed Colebrook interpolants help large, repeated workloads. They are experiments, not certified approximation bounds. Their results depend on the sampled domain, input distribution, hardware, compiler, and error criteria. Validate an interpolant against the direct correlation on the operating domain before using it in a simulation or optimization.

## Adaptive two-input tensor comparison

`AdaptiveColebrookTensor` constructs a dense tensor grid with adaptively refined axes and bicubic Hermite patches. `AdaptiveBilinearColebrookTensor` uses the same sampled tensor-grid/refinement approach with four coefficients per cell. Both interpolate `g = 1/sqrt(f)` in transformed Reynolds and relative-roughness coordinates, then recover the factor and physical derivatives from the patch. The build indicator is heuristic, not a proof of a global error bound. Adaptive axis placement does not make the dense tensor sparse.

Build and run the same-batch CUDA comparison:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DFRICTION_BUILD_CUDA_BENCHMARK=ON \
  -DCMAKE_CUDA_ARCHITECTURES=52
cmake --build build-cuda --target friction_cuda_adaptive_bilinear_benchmark --parallel
./build-cuda/friction_cuda_adaptive_bilinear_benchmark
```

The benchmark measures resident GPU kernel throughput and end-to-end throughput including input and output copies. It reports table construction and initial upload separately; those setup costs are not included in evaluation rates. It checks friction factor and both derivatives on the same deterministic 1,048,576-point batch. Recorded derivative checks compare patch derivatives with the direct Colebrook derivatives.

### Recorded Quadro M3000M result

One run used a Quadro M3000M (compute 5.2), 1,048,576 inputs, and 12 repetitions. The grid had 94,556 nodes in both cases. Scaled-error limits were `2e-4` for value, `2e-2` for the Reynolds slope, and `2e-3` for the relative-roughness slope.

| Method | Resident kernel | End-to-end | Coefficients | Max scaled errors (value / Re slope / roughness slope) | Failed checks |
|---|---:|---:|---:|---:|---:|
| Direct Colebrook | 105.8 M/s | 77.5 M/s | — | Reference | 0 |
| Adaptive bilinear | 166.3 M/s | 97.5 M/s | 2.87 MiB | `1.7e-4 / 2.6e-1 / 2.6e-2` | `0 / 9,425 / 11,393` |
| Adaptive bicubic | 133.9 M/s | 85.1 M/s | 11.47 MiB | `6.2e-6 / 1.8e-2 / 1.7e-5` | `0 / 0 / 0` |

The bilinear interpolant was faster and used less memory in this run. Its friction-factor values met the selected limit, but some derivative values exceeded their stricter limits. Do not use its patch derivatives where those derivative tolerances are required. The bicubic interpolant met all three benchmark limits for this sample; that result does not certify accuracy for other points or domains. The adaptive and uniform bicubic benchmark also found that, at the same 307-by-308 axis sizes, adaptive placement reduced held-out scaled errors but did not make the coefficient array sparse.

The table build took about 232 ms for bilinear and 229 ms for bicubic in this run; initial GPU upload was about 3.60 ms. Rebuilding the interpolant for a single evaluation is not useful. Its potential benefit is repeated evaluation after setup, particularly when the table remains resident on the GPU.

## Fixed-roughness one-dimensional slices

The fixed-roughness slice benchmark compares direct Colebrook derivatives with a precomputed piecewise-linear function in `log(Re)`. A uniform 8,193-sample slice reached 528.9 M evaluations/s resident on the M3000M and 196.1 M/s end-to-end, versus 111.2 M/s and 87.7 M/s for direct GPU Colebrook in the same run. The adaptive version used 2,999 knots; it reached 444.9 M/s resident and 176.2 M/s end-to-end. On the CPU, that adaptive slice was slower than direct evaluation in the reported run, showing that the GPU result does not automatically transfer to CPU performance.

The one-dimensional slice is tied to one relative roughness. Rebuild it when that roughness changes. Its roughness derivative is a local sensitivity channel for the fixed slice. Detailed CUDA build instructions are in [the adaptive 2D benchmark note](adaptive-2d-cuda-benchmark.md).

## Bucketed slice experiment

The bucketed experiment groups adaptive one-dimensional knots into coarse Reynolds-number buckets and compares random log-uniform with Reynolds-sorted input. On the recorded M3000M run, a uniform slice reached 716.8 M/s resident and 209.4 M/s end-to-end for random inputs; the global adaptive slice reached 525.3 M/s and 188.9 M/s; the bucketed adaptive slice reached 561.9 M/s and 192.8 M/s. Sorted inputs improved all three. Bucketed adaptive placement did not outperform the uniform slice in this experiment. See [the bucketed benchmark note](bucketed-slice-experiment.md) for the standalone command and measurement scope.

## Other benchmark targets

With `FRICTION_BUILD_CUDA_BENCHMARK=ON`, CMake also provides:

- `friction_cuda_benchmark`: direct Colebrook and Swamee–Jain primal/derivative throughput over a batch-size sweep.
- `friction_cuda_surrogate_benchmark`: direct Colebrook versus a prepared bicubic slice from the full adaptive tensor.
- `friction_cuda_bilinear_benchmark`: a separate sampled bilinear function benchmark using a different transformed coordinate system.
- `friction_cuda_linear_slice_benchmark`: uniform and adaptive fixed-roughness linear slice comparison.
- `friction_cuda_adaptive_tensor_benchmark`: adaptive versus uniform bicubic tensor node placement.
- `friction_cuda_adaptive_bilinear_benchmark`: adaptive bilinear versus adaptive bicubic, shown above.

The CPU table builder is for experimentation; CUDA support currently benchmarks evaluation, not table construction. Transfer-inclusive rates include only the copies described by each executable and exclude one-time setup unless the executable explicitly says otherwise.
