#include "phase0_adapter/gpu_types.hpp" // Our shared header
#include "rclcpp_components/register_node_macro.hpp"

namespace phase0_adapter
{
// We inherit from rclcpp::Node just like before
class GpuPublisher : public rclcpp::Node
{
public:
  // Components MUST take NodeOptions in the constructor
  explicit GpuPublisher(const rclcpp::NodeOptions & options)
  : Node("gpu_publisher", options)
  {
    pub_ = this->create_publisher<GpuBuffer>("gpu_topic", 10);
    
    timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&GpuPublisher::on_timer, this)
    );
  }

private:
  void on_timer()
  {
    auto msg = std::make_unique<GpuBuffer>();
    cudaMalloc(&msg->dev_ptr, sizeof(float));
    
    float val = 3.14159f;
    cudaMemcpy(msg->dev_ptr, &val, sizeof(float), cudaMemcpyHostToDevice);
    
    RCLCPP_INFO(this->get_logger(), "PUB: Sent GPU Pointer %p", (void*)msg->dev_ptr);
    pub_->publish(std::move(msg));
  }

  rclcpp::Publisher<GpuBuffer>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};
} // namespace phase0_adapter

// This line is the "Magic": it tells ROS 2 this class can be loaded into a container
RCLCPP_COMPONENTS_REGISTER_NODE(phase0_adapter::GpuPublisher)