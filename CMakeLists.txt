cmake_minimum_required(VERSION 2.8.3)
project(aubo_arm_usr)
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++0x") 
find_package(catkin REQUIRED COMPONENTS
  apriltag_ros
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  dh_hand_driver
  moveit_msgs
  moveit_ros_perception
  moveit_ros_planning_interface
  trajectory_msgs
)
 
catkin_package(
    
 INCLUDE_DIRS include
 LIBRARIES aubo_arm_usr
 CATKIN_DEPENDS apriltag_ros geometry_msgs roscpp rospy sensor_msgs std_msgs tf moveit_core moveit_ros_planning_interface
 DEPENDS system_lib
)
 
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# add_executable(moveit_random_demo src/moveit_study/moveit_random_demo.cpp)
# target_link_libraries(moveit_random_demo ${catkin_LIBRARIES})

add_executable(moveit_fk_demo src/moveit_study/moveit_fk_demo.cpp)
target_link_libraries(moveit_fk_demo ${catkin_LIBRARIES})

add_executable(moveit_ik_demo src/moveit_study/moveit_ik_demo.cpp)
target_link_libraries(moveit_ik_demo ${catkin_LIBRARIES})

add_executable(moveit_cartesian_demo src/moveit_study/moveit_cartesian_demo.cpp)
target_link_libraries(moveit_cartesian_demo ${catkin_LIBRARIES})


install(PROGRAMS 
   scripts/moveit_fk_demo.py
   scripts/moveit_ik_demo.py
   scripts/moveit_cartesian_demo.py 
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)