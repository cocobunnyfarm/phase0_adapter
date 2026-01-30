cmake_minimum_required(VERSION 3.14)
project(phase0_adapter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED) # Needed for components
find_package(std_msgs REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(include)

# --- Publisher Component ---
add_library(gpu_pub_component SHARED src/gpu_pub_component.cpp)
target_link_libraries(gpu_pub_component 
  PRIVATE 
    CUDA::cudart
    rclcpp::rclcpp
    rclcpp_components::component_manager # Explicitly link the component manager
    std_msgs::std_msgs__rosidl_typesupport_cpp
)
# We still need this to find the .hpp files!
target_include_directories(gpu_pub_component PRIVATE include)
rclcpp_components_register_nodes(gpu_pub_component "phase0_adapter::GpuPublisher")

# --- Do the same for the Subscriber ---
add_library(gpu_sub_component SHARED src/gpu_sub_component.cpp)
target_link_libraries(gpu_sub_component 
  PRIVATE 
    CUDA::cudart
    rclcpp::rclcpp
    rclcpp_components::component_manager
    std_msgs::std_msgs__rosidl_typesupport_cpp
)
target_include_directories(gpu_sub_component PRIVATE include)
rclcpp_components_register_nodes(gpu_sub_component "phase0_adapter::GpuSubscriber")

install(TARGETS gpu_pub_component gpu_sub_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()


