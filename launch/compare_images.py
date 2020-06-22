from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='opencv_demos',
            node_executable='compare_images',
            node_name='compare_images',
            remappings=[
                ('in_image_base_topic', '/airsim_node/drone_1/front_center_custom/Scene')
            ],
            output="screen"
        )
    ])
