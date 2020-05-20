
/* Author: wang prirong 2020-5-20 */
#include <ros/ros.h>
#include <ros/console.h>

#include <cv_bridge/cv_bridge.h> 
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include <memory>
#include <string> 
#include <vector>
#include <map> 
#include <thread>

#include <tf/transform_broadcaster.h>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>

#include <moveit_visual_tools/moveit_visual_tools.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <pcl_conversions/pcl_conversions.h> 
#include <pcl/point_types.h> 
#include <pcl/PCLPointCloud2.h> 
#include <pcl/conversions.h> 
#include <pcl_ros/transforms.h>  
#include <pcl/visualization/cloud_viewer.h>
// #include <pcl/visualization/cloud_viewer.h> 

#include "std_msgs/Int8.h"
#include "apriltag_ros/AprilTagDetection.h"
#include "apriltag_ros/AprilTagDetectionArray.h" 
#include "aubo_arm_usr/armshakehead.h"
#include "aubo_arm_usr/armshakehand.h"

using namespace Eigen; 

#define PI 3.1415926


tf::StampedTransform Trans_Camera_in_wrist3Link;
tf::StampedTransform Trans_wrist3Link_in_Camera;
tf::StampedTransform Trans_QRcode_in_Camera; 
tf::StampedTransform Trans_Camera_in_World; // 这里我们认为世界是机械臂基座中心
tf::StampedTransform Trans_base_link_in_World;
tf::StampedTransform Trans_Camera_in_base_link;
double T_Tag[16] = {0};   // 相机光心相对于机械臂基座的变换

// 机械臂需要到达的四个位姿
const int take_photo_pose_num = 2;
std::vector<double> joint_place_saft = {79.57/57.3, -2.13/57.3, -53.65/57.3, 17.71/57.3, -93.67/57.3, -5.97/57.3};

std::vector<double> shake_head_start = {85.3201/57.3, 34.793/57.3, 68.918/57.3, -52.9561/57.3, -107/57.3, 4.243/57.3};
std::vector<double> shake_head_end = {85.3201/57.3, 34.793/57.3, 68.918/57.3, -52.9561/57.3, -79.725/57.3, 4.243/57.3};
std::vector<std::vector<double>> joint_place_all_pose;


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "aubo_arm_motion");
    ros::NodeHandle node_handle;
  
    tf::TransformListener listener; 

    geometry_msgs::Pose pose_current;  
    geometry_msgs::Pose pose_base_link_current;
 
    
    ros::AsyncSpinner spinner(1);
    spinner.start();
     
     // ---------------------------------
    // moveit相关模块初始化
    // ---------------------------------
    static const std::string PLANNING_GROUP = "manipulator_i5";

    moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP);

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    // Raw pointers are frequently used to refer to the planning group for improved performance.
    const robot_state::JointModelGroup* joint_model_group =
        move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP);

    const std::vector<std::string>& joint_names = joint_model_group->getVariableNames();

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    
    robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel(); 
    ROS_INFO_STREAM("model_frame is:" << kinematic_model->getModelFrame().c_str());

    robot_state::RobotStatePtr kinematic_state(new robot_state::RobotState(kinematic_model)); 
    kinematic_state->setToDefaultValues();

    const Eigen::Affine3d &end_effector_state = kinematic_state->getGlobalLinkTransform("wrist3_Link");
    ROS_INFO_STREAM("Translation: " << end_effector_state.translation());
    ROS_INFO_STREAM("Rotation: " << end_effector_state.rotation());
     
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;

    // ---------------------------------------
    // MoveItVisualTools 可视化，控制moveit一步步地运行
    // ---------------------------------------    
    namespace rvt = rviz_visual_tools;
    moveit_visual_tools::MoveItVisualTools visual_tools("shoulder_link");   
    visual_tools.deleteAllMarkers();
    visual_tools.loadRemoteControl();
    // RViz provides many types of markers, in this demo we will use text, cylinders, and spheres
    Eigen::Affine3d text_pose = Eigen::Affine3d::Identity();
    text_pose.translation().z() = 1.0;
    visual_tools.publishText(text_pose, "MoveGroupInterface Demo", rvt::WHITE, rvt::XLARGE);
    visual_tools.trigger();

    ROS_INFO_NAMED("tutorial", "Reference frame: %s", move_group.getPlanningFrame().c_str());
    ROS_INFO_NAMED("tutorial", "End effector link: %s", move_group.getEndEffectorLink().c_str());
 
    move_group.setMaxVelocityScalingFactor(0.35);
    move_group.setMaxAccelerationScalingFactor(0.1);

    
 
    move_group.setJointValueTarget(shake_head_start);     
    move_group.plan(my_plan);
    move_group.execute(my_plan);
    int shake_num = 3;
    move_group.setMaxVelocityScalingFactor(0.75);
    move_group.setMaxAccelerationScalingFactor(0.3);
    // ---------------------------------------
    // 执行摇摆程序
    // ---------------------------------------    
    for(int i = 0; i < shake_num; i++)
    {     
        move_group.setJointValueTarget(shake_head_start);     
        move_group.plan(my_plan);
        move_group.execute(my_plan);
        move_group.setJointValueTarget(shake_head_end);     
        move_group.plan(my_plan);
        move_group.execute(my_plan);
    }
  
    // 运动完毕后,清除约束   
    move_group.clearPathConstraints();

    // visual_tools.prompt("Press 'next',  all things done!!! \n");

    ros::shutdown();
    return 0;  
}
