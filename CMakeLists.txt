cmake_minimum_required(VERSION 2.8.3)
project(aubo_arm_usr)
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++0x") 
find_package(catkin REQUIRED COMPONENTS
  apriltag_ros
  geometry_msgs
  message_generation
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  dh_hand_driver
  moveit_msgs
  moveit_ros_perception 
  trajectory_msgs 
  interactive_markers 
  moveit_core
  moveit_visual_tools
  moveit_ros_planning
  moveit_ros_planning_interface
  pluginlib
  geometric_shapes
  pcl_ros
  pcl_conversions
  rosbag
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs 
  ur_kinematics
  find_cup_ros
)

find_package(Boost REQUIRED system filesystem date_time thread)

add_service_files(
  FILES
  armshakehand.srv
  armshakehead.srv
  graspobj.srv
  graspcube.srv
  graspcup.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package( 
 INCLUDE_DIRS 
#  include
 LIBRARIES 
 aubo_arm_usr
#  CATKIN_DEPENDS 
 apriltag_ros geometry_msgs roscpp rospy sensor_msgs std_msgs tf moveit_core moveit_ros_planning_interface message_runtime
#  DEPENDS system_lib
)
 
include_directories(
  include ${catkin_INCLUDE_DIRS}
  include
  ${catkin_INCLUDE_DIRS}
)
 
add_executable(grasp_april_cube_steadily_demo src/grasp_april_cube_steadily.cpp)
target_link_libraries(grasp_april_cube_steadily_demo ${catkin_LIBRARIES})

 
add_executable(grasp_april_cube_once src/grasp_april_cube_once.cpp)
target_link_libraries(grasp_april_cube_once ${catkin_LIBRARIES})


add_executable(aubo_3D_struct src/aubo_3D_struct.cpp)
target_link_libraries(aubo_3D_struct ${catkin_LIBRARIES})
 
add_executable(aubo_motion src/aubo_motion.cpp)
target_link_libraries(aubo_motion ${catkin_LIBRARIES})

add_executable(grasp_cup src/grasp_cup.cpp)
target_link_libraries(grasp_cup ${catkin_LIBRARIES})
  
catkin_install_python(PROGRAMS
   scripts/moveit_fk_demo.py
   scripts/moveit_ik_demo.py
   scripts/moveit_cartesian_demo.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
) 