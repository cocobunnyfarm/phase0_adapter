cmake_minimum_required(VERSION 3.14) # Minimum for modern CUDA support
project(phase0_adapter)

# ... (options)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Use CUDAToolkit (the modern way)
find_package(CUDAToolkit REQUIRED)

add_executable(simple_adapter_test src/simple_adapter_test.cpp)
ament_target_dependencies(simple_adapter_test rclcpp std_msgs)

# Phase 1: GPU Test
add_executable(simple_gpu_adapter_test src/simple_gpu_adapter_test.cpp)
target_link_libraries(simple_gpu_adapter_test 
  PRIVATE 
    CUDA::cudart
)
ament_target_dependencies(simple_gpu_adapter_test rclcpp std_msgs)

install(TARGETS 
  simple_adapter_test
  simple_gpu_adapter_test
  DESTINATION lib/${PROJECT_NAME})

ament_package()