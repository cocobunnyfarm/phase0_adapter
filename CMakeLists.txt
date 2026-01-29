cmake_minimum_required(VERSION 3.8)
project(phase0_adapter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(simple_adapter_test src/simple_adapter_test.cpp)
ament_target_dependencies(simple_adapter_test rclcpp std_msgs)

install(TARGETS simple_adapter_test
  DESTINATION lib/${PROJECT_NAME})

ament_package()