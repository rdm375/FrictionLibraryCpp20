#!/usr/bin/env python3
"""Register the adaptive bilinear CPU test and optional CUDA benchmark."""
from pathlib import Path

path = Path("CMakeLists.txt")
text = path.read_text()

cuda_anchor = """    set_target_properties(friction_cuda_adaptive_tensor_benchmark PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_EXTENSIONS NO)"""
cuda_block = """
    add_executable(friction_cuda_adaptive_bilinear_benchmark
        benchmarks/friction_cuda_adaptive_bilinear_benchmark.cu)
    target_include_directories(friction_cuda_adaptive_bilinear_benchmark
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(friction_cuda_adaptive_bilinear_benchmark PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_EXTENSIONS NO)"""
test_anchor = "    add_test(NAME adaptive_tensor_surrogate COMMAND test_adaptive_tensor_surrogate)"
test_block = """

    add_executable(test_adaptive_bilinear_tensor_surrogate
        tests/test_adaptive_bilinear_tensor_surrogate.cpp)
    target_link_libraries(test_adaptive_bilinear_tensor_surrogate PRIVATE friction::friction)
    add_test(NAME adaptive_bilinear_tensor_surrogate
        COMMAND test_adaptive_bilinear_tensor_surrogate)"""

need_cuda = "friction_cuda_adaptive_bilinear_benchmark" not in text
need_test = "test_adaptive_bilinear_tensor_surrogate" not in text
if need_cuda and cuda_anchor not in text:
    raise SystemExit("Could not find the adaptive CUDA benchmark CMake block; no changes made.")
if need_test and test_anchor not in text:
    raise SystemExit("Could not find the adaptive tensor CTest entry; no changes made.")

if need_cuda:
    text = text.replace(cuda_anchor, cuda_anchor + cuda_block, 1)
if need_test:
    text = text.replace(test_anchor, test_anchor + test_block, 1)
path.write_text(text)
print("CMake registered the adaptive bilinear test and CUDA benchmark.")
