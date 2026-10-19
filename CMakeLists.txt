cmake_minimum_required(VERSION 3.20)
project(friction_factor_cpp20 VERSION 1.2.0 DESCRIPTION "Header-only C++20 Darcy friction factor library" LICENSE "AGPL-3.0-only" LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(friction INTERFACE)
add_library(friction::friction ALIAS friction)
target_include_directories(friction INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(friction INTERFACE cxx_std_20)

option(FRICTION_BUILD_EXAMPLES "Build usage examples" ON)
if(FRICTION_BUILD_EXAMPLES)
    add_executable(friction_example examples/friction_example.cpp)
    target_link_libraries(friction_example PRIVATE friction::friction)
endif()

option(FRICTION_BUILD_BENCHMARKS "Build CPU throughput benchmarks" OFF)
if(FRICTION_BUILD_BENCHMARKS)
    add_executable(friction_primal_benchmark benchmarks/friction_primal_benchmark.cpp)
    target_link_libraries(friction_primal_benchmark PRIVATE friction::friction)
    find_package(Threads REQUIRED)
    add_executable(friction_batch_benchmark benchmarks/friction_batch_benchmark.cpp)
    target_link_libraries(friction_batch_benchmark PRIVATE friction::friction Threads::Threads)
    add_executable(colebrook_solver_benchmark benchmarks/colebrook_solver_benchmark.cpp)
    target_link_libraries(colebrook_solver_benchmark PRIVATE friction::friction)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(friction_primal_benchmark PRIVATE -O3 -march=native)
        target_compile_options(friction_batch_benchmark PRIVATE -O3 -march=native)
        target_compile_options(colebrook_solver_benchmark PRIVATE -O3 -march=native)
    endif()
endif()

option(FRICTION_BUILD_CUDA_BENCHMARK "Build optional CUDA vs CPU benchmark" OFF)
if(FRICTION_BUILD_CUDA_BENCHMARK)
    enable_language(CUDA)
    add_executable(friction_cuda_benchmark benchmarks/friction_cuda_benchmark.cu)
    set_target_properties(friction_cuda_benchmark PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_EXTENSIONS NO)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test_friction tests/test_friction.cpp)
    target_link_libraries(test_friction PRIVATE friction::friction)
    add_test(NAME friction COMMAND test_friction)

    add_executable(test_analytic_derivatives tests/test_analytic_derivatives.cpp)
    target_link_libraries(test_analytic_derivatives PRIVATE friction::friction)
    add_test(NAME analytic_derivatives COMMAND test_analytic_derivatives)

    add_executable(test_batch tests/test_batch.cpp)
    target_link_libraries(test_batch PRIVATE friction::friction)
    add_test(NAME batch COMMAND test_batch)

    add_executable(test_colebrook_methods tests/test_colebrook_methods.cpp)
    target_link_libraries(test_colebrook_methods PRIVATE friction::friction)
    add_test(NAME colebrook_methods COMMAND test_colebrook_methods)
endif()

configure_package_config_file(
    cmake/frictionConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/frictionConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/friction)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/frictionConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(TARGETS friction EXPORT frictionTargets)
install(DIRECTORY include/friction DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/FrictionLibraryCpp20)
install(EXPORT frictionTargets
    FILE frictionTargets.cmake
    NAMESPACE friction::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/friction)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/frictionConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/frictionConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/friction)

set(CPACK_PACKAGE_NAME "FrictionLibraryCpp20")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Header-only C++20 Darcy friction factor library")
set(CPACK_PACKAGE_LICENSE "AGPL-3.0-only")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_GENERATOR "TGZ")
include(CPack)

option(FRICTION_ENABLE_ENZYME_AD "Build Enzyme-generated derivative support" OFF)
set(FRICTION_ENZYME_PLUGIN "" CACHE FILEPATH "Path to the ClangEnzyme LLVM plugin matching the selected Clang")

if(FRICTION_ENABLE_ENZYME_AD)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "FRICTION_ENABLE_ENZYME_AD requires Clang built against the matching LLVM version")
    endif()
    if(NOT EXISTS "${FRICTION_ENZYME_PLUGIN}")
        message(FATAL_ERROR "Set FRICTION_ENZYME_PLUGIN to the matching ClangEnzyme-<LLVM_VERSION>.so")
    endif()

    add_library(friction_enzyme_ad STATIC src/enzyme_derivatives.cpp)
    target_link_libraries(friction_enzyme_ad PUBLIC friction::friction)
    target_compile_options(friction_enzyme_ad PRIVATE
        -O3 -march=native "-fplugin=${FRICTION_ENZYME_PLUGIN}")

    if(BUILD_TESTING)
        add_executable(test_enzyme_derivatives tests/test_enzyme_derivatives.cpp)
        target_link_libraries(test_enzyme_derivatives PRIVATE friction_enzyme_ad)
        add_test(NAME enzyme_derivatives COMMAND test_enzyme_derivatives)
    endif()

    add_executable(friction_cpu_benchmark benchmarks/friction_cpu_benchmark.cpp)
    target_link_libraries(friction_cpu_benchmark PRIVATE friction_enzyme_ad)
    target_compile_options(friction_cpu_benchmark PRIVATE -O3 -march=native)
endif()
