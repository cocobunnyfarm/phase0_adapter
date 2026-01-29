#include <memory>
#include <string>
#include <chrono>

// 1. ROS Messages first
#include "std_msgs/msg/string.hpp"
#include "rclcpp/rclcpp.hpp"

// ==========================================
// THE CUSTOM TYPE
// ==========================================
struct LocalString {
  std::string text;
  int magic_number;
};

// ==========================================
// THE ADAPTER (Manual Registration)
// ==========================================
// We put this inside the rclcpp namespace explicitly.
// This is what the Publisher looks for.
namespace rclcpp
{
template<>
struct TypeAdapter<LocalString, std_msgs::msg::String>
{
  using is_specialized = std::true_type;
  using custom_type = LocalString;
  using ros_message_type = std_msgs::msg::String;

  static void convert_to_ros_message(
    const custom_type & source,
    ros_message_type & destination)
  {
    destination.data = source.text;
  }

  static void convert_to_custom(
    const ros_message_type & source,
    custom_type & destination)
  {
    destination.text = source.data;
    destination.magic_number = 0;
  }
};

// HUMBLE FIX: We also need to specialize the single-argument version
// so the Publisher<LocalString> knows which ROS message to use.
template<>
struct TypeAdapter<LocalString> : public TypeAdapter<LocalString, std_msgs::msg::String> {};

}  // namespace rclcpp

// ==========================================
// THE NODE
// ==========================================
class AdapterNode : public rclcpp::Node
{
public:
  AdapterNode() : Node("adapter_node", rclcpp::NodeOptions().use_intra_process_comms(true))
  {
    // Humble's create_publisher needs the explicit types here to be safe
    pub_ = this->create_publisher<LocalString>("chatter", 10);

    sub_ = this->create_subscription<LocalString>(
      "chatter",
      10,
      [this](const LocalString & msg) {
        RCLCPP_INFO(this->get_logger(), 
          "Received: '%s' | Magic: %d", 
          msg.text.c_str(), msg.magic_number);
      }
    );

    timer_ = this->create_wall_timer(
        std::chrono::seconds(1),
        [this]() {
        auto msg = std::make_unique<LocalString>(); // Use unique_ptr!
        msg->text = "Zero Copy Success";
        msg->magic_number = 42; 
        
        RCLCPP_INFO(this->get_logger(), "Publishing pointer...");
        pub_->publish(std::move(msg)); // Move the pointer into ROS
        }
    );

    // timer_ = this->create_wall_timer(
    //   std::chrono::seconds(1),
    //   [this]() {
    //     LocalString msg;
    //     msg.text = "Phase 0 Success";
    //     msg.magic_number = 42; 
    //     pub_->publish(msg);
    //   }
    // );
  }

private:
  rclcpp::Publisher<LocalString>::SharedPtr pub_;
  rclcpp::Subscription<LocalString>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AdapterNode>());
  rclcpp::shutdown();
  return 0;
}