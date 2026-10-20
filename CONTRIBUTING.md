# Contributing

Thanks for helping improve FrictionLibraryCpp20. Changes to numerical methods should include a reproducible test or benchmark that explains the supported input range and checks results against an independent reference where practical.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For CPU benchmarks, configure with `-DFRICTION_BUILD_BENCHMARKS=ON`. CUDA experiments require a compatible NVIDIA toolkit and GPU and are not part of the default test suite.

## Pull requests

- Describe the numerical or API change and its intended input range.
- Include tests for valid cases, boundaries, and invalid inputs when relevant.
- Run `git diff --check` and the CMake test suite.
- Treat benchmark numbers as machine-specific measurements; include hardware, compiler, flags, input distribution, and tolerances when reporting them.
- Keep the public API and documentation clear about the difference between direct correlations and experimental surrogates.

Contributions to this project are made under AGPL-3.0-only. Please include the appropriate copyright and SPDX notices in new source files and ensure you have the rights to contribute the work.
