
################################################################################
# Independent binaries
################################################################################
add_executable(        homogenizedElasticityTensor homogenizedElasticityTensor.cc)
target_link_libraries( homogenizedElasticityTensor MeshFEM)
target_compile_options(homogenizedElasticityTensor PRIVATE -Wno-unused-parameter -Wno-unknown-pragmas)
set_target_properties( homogenizedElasticityTensor PROPERTIES CXX_STANDARD 14)

find_package(Knitro QUIET)
if (${KNITRO_FOUND})
    add_executable(        Topology_Optimization Topology_Optimization.cc)
    target_link_libraries( Topology_Optimization MeshFEM  ${KNITRO_LIBRARIES})
    target_include_directories(Topology_Optimization SYSTEM PUBLIC ${KNITRO_INCLUDE_DIRS})
    target_compile_options(Topology_Optimization PRIVATE -Wno-unused-parameter -Wno-unknown-pragmas)
    set_target_properties( Topology_Optimization PROPERTIES CXX_STANDARD 14)
endif()

add_executable(        test_TensorProductSimulator test_TensorProductSimulator.cc)
target_link_libraries( test_TensorProductSimulator MeshFEM)
target_compile_options(test_TensorProductSimulator PRIVATE -Wno-unused-parameter -Wno-unknown-pragmas)
set_target_properties( test_TensorProductSimulator PROPERTIES CXX_STANDARD 14)

add_executable(        test_TopologyOptimizationProblem test_TopologyOptimizationProblem.cc)
target_link_libraries( test_TopologyOptimizationProblem MeshFEM)
target_compile_options(test_TopologyOptimizationProblem PRIVATE -Wno-unused-parameter -Wno-unknown-pragmas)
set_target_properties( test_TopologyOptimizationProblem PROPERTIES CXX_STANDARD 14)
