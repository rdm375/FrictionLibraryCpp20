# Changelog

## 1.3.0 (unreleased)

- Added adaptive two-input bilinear and bicubic Colebrook interpolant experiments with analytic value and derivative evaluation.
- Added CPU and optional CUDA benchmarks for direct correlations, batch evaluation, and interpolation strategies.
- Clarified that interpolants are experimental, use heuristic refinement indicators, and require application-specific validation.
- Restored CPack packaging and made benchmark targets opt-in for normal library builds.
- Expanded tests for solver methods, derivatives, batches, and adaptive interpolation.
