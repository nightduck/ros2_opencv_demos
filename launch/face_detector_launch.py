from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='opencv_demos',
            node_executable='face_detector',
            node_name='face_detector',
            remappings=[
                ('in_image_base_topic', '/airsim_node/drone_1/front_center_custom/Scene'),
                ('out_image_base_topic', '/face_detector/out_image_base_topic'),
            ]
        )
    ])