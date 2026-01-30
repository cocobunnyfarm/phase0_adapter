#pragma once

#include <cuda_runtime.h>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

// The custom structure must be in the header so both .so files see it
struct GpuBuffer {
  float* dev_ptr;
};

namespace rclcpp
{
template<>
struct TypeAdapter<GpuBuffer, std_msgs::msg::Float32>
{
  using is_specialized = std::true_type;
  using custom_type = GpuBuffer;
  using ros_message_type = std_msgs::msg::Float32;

  static void convert_to_ros_message(const custom_type & source, ros_message_type & dest)
  {
    float host_val;
    cudaMemcpy(&host_val, source.dev_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    dest.data = host_val;
  }

  static void convert_to_custom(const ros_message_type & source, custom_type & dest)
  {
    cudaMalloc(&dest.dev_ptr, sizeof(float));
    cudaMemcpy(dest.dev_ptr, &source.data, sizeof(float), cudaMemcpyHostToDevice);
  }
};

// 2. The Identity Specialization (Crucial for Component syntax)
template<>
struct TypeAdapter<GpuBuffer> : public TypeAdapter<GpuBuffer, std_msgs::msg::Float32> {};

} // namespace rclcpp