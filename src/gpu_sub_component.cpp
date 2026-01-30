#include "phase0_adapter/gpu_types.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace phase0_adapter
{
class GpuSubscriber : public rclcpp::Node
{
public:
  explicit GpuSubscriber(const rclcpp::NodeOptions & options)
  : Node("gpu_subscriber", options)
  {
    sub_ = this->create_subscription<GpuBuffer>(
      "gpu_topic", 10,
      [this](const GpuBuffer & msg) {
        float host_val;
        cudaMemcpy(&host_val, msg.dev_ptr, sizeof(float), cudaMemcpyDeviceToHost);
        RCLCPP_INFO(this->get_logger(), "SUB: Got %f from %p", host_val, (void*)msg.dev_ptr);
        cudaFree(msg.dev_ptr);
      });
  }

private:
  rclcpp::Subscription<GpuBuffer>::SharedPtr sub_;
};
}

RCLCPP_COMPONENTS_REGISTER_NODE(phase0_adapter::GpuSubscriber)