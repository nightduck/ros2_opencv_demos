from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='opencv_demos',
            node_executable='object_tracking',
            node_name='object_tracking',
            remappings=[
                ('in_image_base_topic', '/airsim_node/drone_1/front_center_custom/Scene'),
                ('out_image_base_topic', '/object_tracking/out_image_base_topic'),
            ]
        )
    ])
