from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name='gpu_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container', # The pre-built ROS 2 process
        composable_node_descriptions=[
            # Load the Publisher
            ComposableNode(
                package='phase0_adapter',
                plugin='phase0_adapter::GpuPublisher',
                name='publisher',
                extra_arguments=[{'use_intra_process_comms': True}] # THE KEY SETTING
            ),
            # Load the Subscriber
            ComposableNode(
                package='phase0_adapter',
                plugin='phase0_adapter::GpuSubscriber',
                name='subscriber',
                extra_arguments=[{'use_intra_process_comms': True}] # THE KEY SETTING
            ),
        ],
        output='screen',
    )

    return LaunchDescription([container])