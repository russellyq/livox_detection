cmake_minimum_required(VERSION 2.8.3)
project(livox_detection)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_conversions
  pcl_ros
  roslib
)

find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)

catkin_package(
 CATKIN_DEPENDS roscpp rospy std_msgs
 DEPENDS PCL
)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

add_executable(livox_rotate_translate_repub src/livox_rotate_translate_repub.cpp)
target_link_libraries(livox_rotate_translate_repub ${catkin_LIBRARIES} ${PCL_LIBRARIES} /usr/lib/x86_64-linux-gnu/libyaml-cpp.so)
