

#include <memory>
#include <chrono>
#include <cuda_runtime.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

using namespace std::chrono_literals;

struct GpuBuffer {
  float* dev_ptr;
};

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

class GpuBridgeNode : public rclcpp::Node {
public:
  GpuBridgeNode() : Node("gpu_bridge_node", 
    rclcpp::NodeOptions().use_intra_process_comms(true)) {
    pub_ = this->create_publisher<GpuBuffer>("gpu_topic", 10);
    sub_ = this->create_subscription<GpuBuffer>("gpu_topic", 10,
      [this](const GpuBuffer& msg) {
        float host_val;
        cudaMemcpy(&host_val, msg.dev_ptr, sizeof(float), cudaMemcpyDeviceToHost);
        RCLCPP_INFO(this->get_logger(), "SUB: Got %f from GPU addr %p", host_val, (void*)msg.dev_ptr);
        // cudaFree(msg.dev_ptr);
      });

    timer_ = this->create_wall_timer(1s, [this]() {
      auto msg = std::make_unique<GpuBuffer>();
      float val = 3.14159f;
      host_pinned_mem_ptr_[0] = val;
      cudaMemcpyAsync(msg->dev_ptr, &host_pinned_mem_ptr_[0], sizeof(float), cudaMemcpyHostToDevice, stream_);
      
      RCLCPP_INFO(this->get_logger(), "PUB: Sending GPU addr %p", (void*)msg->dev_ptr);
      pub_->publish(std::move(msg));
    });
  }

  auto Initialize() -> void {
    cudaStreamCreate(&stream_);
    host_pinned_mem_size_ = 2014 * sizeof(float);
    cudaError_t status = cudaHostAlloc((void**)&host_pinned_mem_ptr_, sizeof(float), cudaHostAllocDefault);
    if (status != cudaSuccess) {
        std::cerr << "할당 실패: " << cudaGetErrorString(status) << std::endl;
        return -1;
    }


  }

private:
  rclcpp::Publisher<GpuBuffer>::SharedPtr pub_;
  rclcpp::Subscription<GpuBuffer>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<GpuBuffer> msg;

  float* host_pinned_mem_ptr_;
  float* device_ptr_;
  size_t host_pinned_mem_size_;
  cudaStream_t stream_;
  
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GpuBridgeNode>());
  rclcpp::shutdown();
  return 0;
}