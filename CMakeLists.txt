cmake_minimum_required(VERSION 3.14)
project(phase0_adapter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 1. Manual Link for CUDA (This needs keywords like PRIVATE)


add_executable(simple_gpu_adapter_test src/simple_gpu_adapter_test.cpp)

target_link_libraries(simple_gpu_adapter_test 
  # PRIVATE 
    CUDA::cudart
)
ament_target_dependencies(simple_gpu_adapter_test 
  rclcpp 
  std_msgs
)


add_executable(simple_gpu_adapter_test_improved src/simple_gpu_adapter_test_improved.cpp)

target_link_libraries(simple_gpu_adapter_test_improved 
  # PRIVATE 
    CUDA::cudart
)

ament_target_dependencies(simple_gpu_adapter_test_improved 
  rclcpp 
  std_msgs
)


add_executable(adapter_sync_pinned_counter
 src/adapter_sync_pinned_counter.cpp)

target_link_libraries(adapter_sync_pinned_counter 
  # PRIVATE 
    CUDA::cudart
)

ament_target_dependencies(adapter_sync_pinned_counter 
  rclcpp 
  std_msgs
)
# 2. ROS Helper for Includes and Linking (This fixes the missing .hpp error)
# This macro handles the include paths for rclcpp and std_msgs automatically.

install(TARGETS simple_gpu_adapter_test simple_gpu_adapter_test_improved adapter_sync_pinned_counter DESTINATION lib/${PROJECT_NAME})
ament_package()