/***
 * Improvements:
 * 
 * 1) Use Pinned Memory
 * cudaHostAlloc(&host_ptr, size, cudaHostAllocDefault);
 * 
 * 
 * cudaError_t cudaMemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    cudaMemcpyKind kind,
    cudaStream_t stream
  );
 * 
 */

#include <memory>
#include <chrono>
#include <cuda_runtime.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

using namespace std::chrono_literals;

// 1. DATA STRUCTURE
struct GpuBuffer {
  float* dev_ptr;
};

// 2. THE ADAPTER (Must be fully defined before the Node)
namespace rclcpp {
template<>
struct TypeAdapter<GpuBuffer, std_msgs::msg::Float32> {
  using is_specialized = std::true_type;
  using custom_type = GpuBuffer;
  using ros_message_type = std_msgs::msg::Float32;

  static void convert_to_ros_message(const custom_type& source, ros_message_type& dest) {
    float host_val;
    cudaMemcpy(&host_val, source.dev_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    dest.data = host_val;
  }

  static void convert_to_custom(const ros_message_type& source, custom_type& dest) {
    cudaMalloc(&dest.dev_ptr, sizeof(float));
    cudaMemcpy(dest.dev_ptr, &source.data, sizeof(float), cudaMemcpyHostToDevice);
  }
};

// This alias is the secret sauce that tells the Publisher GpuBuffer is okay.
template<>
struct TypeAdapter<GpuBuffer> : public TypeAdapter<GpuBuffer, std_msgs::msg::Float32> {};

} // namespace rclcpp

// 3. THE NODE
class GpuBridgeNode : public rclcpp::Node {
public:
  GpuBridgeNode() : Node("gpu_bridge_node", 
    rclcpp::NodeOptions().use_intra_process_comms(true)) 
  {
    msg = std::make_unique<GpuBuffer>();
    cudaMalloc(&msg->dev_ptr, sizeof(float));


    // Now the Publisher knows GpuBuffer has a TypeAdapter!
    pub_ = this->create_publisher<GpuBuffer>("gpu_topic", 10);

    sub_ = this->create_subscription<GpuBuffer>("gpu_topic", 10,
      [this](const GpuBuffer& msg) {
        float host_val;
        cudaMemcpy(&host_val, msg.dev_ptr, sizeof(float), cudaMemcpyDeviceToHost);
        RCLCPP_INFO(this->get_logger(), "SUB: Got %f from GPU addr %p", host_val, (void*)msg.dev_ptr);
        // cudaFree(msg.dev_ptr);
      });

    timer_ = this->create_wall_timer(1s, [this]() {
      
      float val = 3.14159f;
      cudaMemcpy(msg->dev_ptr, &val, sizeof(float), cudaMemcpyHostToDevice);
      
      RCLCPP_INFO(this->get_logger(), "PUB: Sending GPU addr %p", (void*)msg->dev_ptr);
      raw_device_ptr_ = msg->dev_ptr;
      pub_->publish(std::move(msg));
      msg = std::make_unique<GpuBuffer>();
      msg->dev_ptr = raw_device_ptr_;
    });
  }

private:
  rclcpp::Publisher<GpuBuffer>::SharedPtr pub_;
  rclcpp::Subscription<GpuBuffer>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<GpuBuffer> msg;
  float* raw_device_ptr_;
  
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GpuBridgeNode>());
  rclcpp::shutdown();
  return 0;
}