# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Richard Myers

cmake_minimum_required(VERSION 3.20)
project(friction_package_consumer LANGUAGES CXX)

find_package(friction 1 CONFIG REQUIRED)
add_executable(friction_package_consumer main.cpp)
target_link_libraries(friction_package_consumer PRIVATE friction::friction)

enable_testing()
add_test(NAME installed_package_consumer COMMAND friction_package_consumer)
