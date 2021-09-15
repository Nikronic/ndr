cmake_minimum_required(VERSION 3.1)
project(VoxelFEM)

if (${CMAKE_VERSION} VERSION_GREATER "3.13")
    cmake_policy(SET CMP0076 NEW)
endif()

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile flags (used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Make sure warnings/errors are still colorized when using Ninja for building.
add_definitions(-fdiagnostics-color=always)

# For python bindings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#Include MeshFem (Eigen will be included by MeshFEM)
if (NOT TARGET MeshFEM)
    option(MESHFEM_ENABLE_BENCHMARKING "" ON)
    option(MESHFEM_VECTORIZE "" ON) # Request vectorization
    add_subdirectory(${THIRD_PARTY_DIR}/MeshFEM)
endif()

# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

add_library(VoxelFEM INTERFACE)
target_sources(VoxelFEM INTERFACE
    TensorProductBasisPolynomial.hh
    TensorProductPolynomialInterpolant.hh
    TensorProductQuadrature.hh
    TensorProductSimulator.hh
    TensorProjection.hh
    TPPeriodicHomogenization.hh
)
target_link_libraries(VoxelFEM INTERFACE MeshFEM)
target_compile_options(VoxelFEM INTERFACE -g -march=native -ffast-math)

add_subdirectory(tests)
add_subdirectory(python_bindings)
