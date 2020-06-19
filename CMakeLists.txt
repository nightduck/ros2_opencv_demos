cmake_minimum_required(VERSION 3.5)
project(opencv_demos)

SET(CMAKE_C_COMPILER /usr/bin/gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/g++)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV 4 REQUIRED
  COMPONENTS
  opencv_core
  opencv_imgproc
  opencv_highgui
  opencv_objdetect
  opencv_videoio
  opencv_video
  opencv_imgcodecs
  opencv_cudaarithm
  opencv_cudafilters
  CONFIG
)
find_package(cv_bridge REQUIRED)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(face_detector src/face_detector.cpp)
ament_target_dependencies(face_detector rclcpp image_transport sensor_msgs cv_bridge)

add_executable(object_tracking src/object_tracking.cpp)
ament_target_dependencies(object_tracking rclcpp image_transport sensor_msgs cv_bridge)

add_executable(compare_images src/compare_images.cpp)
ament_target_dependencies(compare_images rclcpp image_transport sensor_msgs cv_bridge)

install(TARGETS
        face_detector
        compare_images
        object_tracking
        DESTINATION lib/${PROJECT_NAME})

ament_package()
