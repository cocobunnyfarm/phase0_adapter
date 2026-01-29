#include <cuda_runtime.h>

struct GpuBuffer {
    float* dev_ptr; // Pointer to memory on the GPU (VRAM)
};

// --- Inside the TypeAdapter specialization ---
// We convert GpuBuffer to std_msgs::msg::Float32 (as a fallback)
static void convert_to_ros_message(const GpuBuffer& source, std_msgs::msg::Float32& dest) {
    float host_val;
    // Download from GPU to CPU so standard ROS nodes can see it
    cudaMemcpy(&host_val, source.dev_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    dest.data = host_val;
}

// --- Inside the Publisher Timer ---
auto msg = std::make_unique<GpuBuffer>();
cudaMalloc(&msg->dev_ptr, sizeof(float)); // Allocate in VRAM

float pi = 3.14159f;
cudaMemcpy(msg->dev_ptr, &pi, sizeof(float), cudaMemcpyHostToDevice); // Write to GPU

RCLCPP_INFO(this->get_logger(), "Sent GPU Address: %p", (void*)msg->dev_ptr);
pub_->publish(std::move(msg));

// --- Inside the Subscriber Callback ---
float val;
// Read directly from the GPU address we received!
cudaMemcpy(&val, msg->dev_ptr, sizeof(float), cudaMemcpyDeviceToHost);
RCLCPP_INFO(this->get_logger(), "Received from GPU: %f", val);
cudaFree(msg->dev_ptr); // Clean up for now (we'll automate this in Phase 2)