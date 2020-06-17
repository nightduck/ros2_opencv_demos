# ros2_opencv_demos
A collection of ROS2 nodes that use OpenCV. Intended as an artificial workload for RT research

This is meant to be a single project in a larger ROS2 workspace.

# How to use

    cd ros2_ws/src
    git clone https://github.com/nightduck/ros2_opencv_demos.git opencv_demos
    cd ../..
    colcon build
    source install/local_setup.bash
    
 The launch files in the `launch/` folder have mappings you can edit to set your input and output topics
 (currently set to the front camera in the Airsim drone).
 
 These launch files can be run with:
 
     ros2 launch src/opencv_demos/launch/object_tracking_launch.py
