

/* Author: wang prirong 2019-8-24 */
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

using namespace Eigen; 

#define PI 3.1415926

void handle_points_in_camera(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);
bool pose_to_mat(geometry_msgs::Pose pose, double* T);

// 坐标齐次变换变量
// geometry_msgs::Pose  Pose_QRcode_in_Camera;
// tf::StampedTransform Transform_target;
// tf::StampedTransform Trans_Gripper_in_wrist3Link;
// tf::StampedTransform Trans_wrist3Link_in_Gripper;
tf::StampedTransform Trans_Camera_in_wrist3Link;
tf::StampedTransform Trans_wrist3Link_in_Camera;
tf::StampedTransform Trans_QRcode_in_Camera; 
tf::StampedTransform Trans_Camera_in_World; // 这里我们认为世界是机械臂基座中心
tf::StampedTransform Trans_base_link_in_World;
tf::StampedTransform Trans_Camera_in_base_link;
double T_Tag[16] = {0};   // 相机光心相对于机械臂基座的变换
// 扫描拼接的总点云
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_base_arm(new pcl::PointCloud<pcl::PointXYZRGB>); 

// 机械臂需要到达的四个位姿
const int take_photo_pose_num = 2;
std::vector<double> joint_place_saft = {79.57/57.3, -2.13/57.3, -53.65/57.3, 17.71/57.3, -93.67/57.3, -5.97/57.3};
std::vector<double> joint_place_pose1 = {117.08/57.3, 5.63/57.3, -63.71/57.3, 17.29/57.3, -113.88/57.3, 26.49/57.3};
std::vector<double> joint_place_pose2 = {79.57/57.3, -2.13/57.3, -53.65/57.3, 17.71/57.3, -93.67/57.3, -5.97/57.3};
std::vector<double> joint_place_pose3 = {91.28/57.3, -39.54/57.3, -115.20/57.3, -8.71/57.3, -94.51/57.3, 2.72/57.3};
std::vector<double> joint_place_pose4 = {47.89/57.3, 27.94/57.3, -52.70/57.3, 26.80/57.3, -75.88/57.3, -52.95/57.3}; 
std::vector<std::vector<double>> joint_place_all_pose;      

bool get_cloud_flag = true;

// 相机内参
double fx = 571.8319822931295;
double cx = 436.7418038692657;
double fy = 568.5477678366276;
double cy = 259.9872860050479;
double depthScale = 1000.0;

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "aubo_3d_struct");
    ros::NodeHandle node_handle;
 
    // 初始化变量
    // Transform_target.setOrigin(tf::Vector3(0, 0, 0));
    // Transform_target.setRotation(tf::Quaternion(0, 0, 0, 1));
    
    tf::TransformListener listener; 

    geometry_msgs::Pose pose_current;  
    geometry_msgs::Pose pose_base_link_current;

    // ---------------------------------
    // 话题订阅与发布
    // ---------------------------------
    message_filters::Subscriber<sensor_msgs::Image>rgb_sub(node_handle, "/kinect2/qhd/image_color_rect", 1);
    message_filters::Subscriber<sensor_msgs::Image>depth_sub(node_handle,"/kinect2/qhd/image_depth_rect",1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub); 
    sync.registerCallback(boost::bind(&handle_points_in_camera,_1,_2)); 
    
    ros::AsyncSpinner spinner(1);
    spinner.start();
    // ---------------------------------------
    // 固连坐标系变换关系求解
    // ---------------------------------------
    tf::Transform ttf;
    bool tf_OK = false; 
    while (tf_OK == false)
    {
        try{
            listener.waitForTransform("wrist3_Link", "kinect2_rgb_optical_frame", ros::Time(0), ros::Duration(1.0));
            listener.lookupTransform("wrist3_Link", "kinect2_rgb_optical_frame", ros::Time(0), Trans_Camera_in_wrist3Link);
            tf_OK = true;
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
            ros::Duration(1.0).sleep();
        }
        ros::Duration(1.0).sleep();
        ROS_INFO("get_Trans_Camera_in_wrist3Link");
    }

    ttf = Trans_Camera_in_wrist3Link.inverse();
    Trans_wrist3Link_in_Camera.setData(ttf);
    
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
 
    move_group.setMaxVelocityScalingFactor(0.15);
    move_group.setMaxAccelerationScalingFactor(0.1);

    // ---------------------------------------
    // 记录四个位置
    // ---------------------------------------    
    joint_place_all_pose.push_back(joint_place_pose1);
    joint_place_all_pose.push_back(joint_place_pose2); 
    // joint_place_all_pose.push_back(joint_place_pose3);
    // joint_place_all_pose.push_back(joint_place_pose4);  

    // --------------------------------------- 
    // 首先到达拍照安全过渡位
    // --------------------------------------- 
    move_group.setJointValueTarget(joint_place_saft);     
    move_group.plan(my_plan);
    move_group.execute(my_plan);
    
    // ---------------------------------------
    // 移动四个位置拍照
    // ---------------------------------------    
    for(int i = 0; i < take_photo_pose_num; i++)
    {    
        ROS_INFO_STREAM("Move endeffort to pose " << i+1);
        move_group.setJointValueTarget(joint_place_all_pose[i]);     
        move_group.plan(my_plan);
        move_group.execute(my_plan); 

        // ---------------------------------------
        // 获取相机相对于基座的位姿
        // ---------------------------------------  
        pose_base_link_current = move_group.getCurrentPose("base_link").pose;
        Trans_base_link_in_World.setOrigin(tf::Vector3(pose_base_link_current.position.x, pose_base_link_current.position.y, pose_base_link_current.position.z));
        Trans_base_link_in_World.setRotation(tf::Quaternion(pose_base_link_current.orientation.x, pose_base_link_current.orientation.y, 
                                                        pose_base_link_current.orientation.z, pose_base_link_current.orientation.w));
        pose_current = move_group.getCurrentPose("kinect2_rgb_optical_frame").pose;
        Trans_Camera_in_World.setOrigin(tf::Vector3(pose_current.position.x, pose_current.position.y, pose_current.position.z));
        Trans_Camera_in_World.setRotation(tf::Quaternion(pose_current.orientation.x, pose_current.orientation.y, 
                                                        pose_current.orientation.z, pose_current.orientation.w));
        Trans_Camera_in_base_link.mult(Trans_base_link_in_World.inverse(), Trans_Camera_in_World);
        
        geometry_msgs::Pose pose_temp;
        pose_temp.position.x = Trans_Camera_in_base_link.getOrigin().x();
        pose_temp.position.y = Trans_Camera_in_base_link.getOrigin().y();
        pose_temp.position.z = Trans_Camera_in_base_link.getOrigin().z();
        pose_temp.orientation.x = Trans_Camera_in_base_link.getRotation().x();
        pose_temp.orientation.y = Trans_Camera_in_base_link.getRotation().y();
        pose_temp.orientation.z = Trans_Camera_in_base_link.getRotation().z();
        pose_temp.orientation.w = Trans_Camera_in_base_link.getRotation().w();
        pose_to_mat(pose_temp, T_Tag);
        // Eigen::Quaterniond q( Trans_Camera_in_base_link.getRotation().w(),Trans_Camera_in_base_link.getRotation().x(), Trans_Camera_in_base_link.getRotation().y(), Trans_Camera_in_base_link.getRotation().z() );
        // Eigen::Isometry3d T(q);
        // T.pretranslate( Eigen::Vector3d( Trans_Camera_in_base_link.getOrigin().x(), Trans_Camera_in_base_link.getOrigin().y(), Trans_Camera_in_base_link.getOrigin().z()));
        // T_points_base = T; 
        ROS_INFO_STREAM("Get points and Processing ...");
        get_cloud_flag = false;
        while(!get_cloud_flag);  
    }
 
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
	viewer.showCloud(cloud_base_arm);
	while (!viewer.wasStopped())
	{

	}

    // 运动完毕后,清除约束   
    move_group.clearPathConstraints();

    visual_tools.prompt("Press 'next',  all things done!!! \n");

    ros::shutdown();
    return 0;  
}


void handle_points_in_camera(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD)
{   
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    if(!get_cloud_flag)
    {

        ROS_INFO_STREAM("handle points in camera");  
        cv::Mat color = cv_ptrRGB->image;
        cv::Mat depth = cv_ptrD->image;
 
  
        for ( int v=0; v<color.rows; v++ )
            for ( int u=0; u<color.cols; u++ )
            {
                unsigned int d = depth.ptr<unsigned short> (v)[u]; // 深度值
                if ( d==0 ) continue; // 为0表示没有测量到
                Eigen::Vector3d point; 
                point[2] = double(d)/depthScale; 
                point[0] = (u-cx)*point[2]/fx;
                point[1] = (v-cy)*point[2]/fy;   

                pcl::PointXYZRGB p ;
                p.x = *(T_Tag+0) * point[0] + *(T_Tag+1) * point[1] + *(T_Tag+2) * point[2] + *(T_Tag+3);
                p.y = *(T_Tag+4) * point[0] + *(T_Tag+5) * point[1] + *(T_Tag+6) * point[2] + *(T_Tag+7);
                p.z = *(T_Tag+8) * point[0] + *(T_Tag+9) * point[1] + *(T_Tag+10) * point[2] + *(T_Tag+11);
                p.b = color.data[ v*color.step+u*color.channels() ];
                p.g = color.data[ v*color.step+u*color.channels()+1 ];
                p.r = color.data[ v*color.step+u*color.channels()+2 ];
                cloud_base_arm->points.push_back( p );
            }
         
        get_cloud_flag = true;
    } 
}


bool pose_to_mat(geometry_msgs::Pose pose, double* T)
{
    // trans
    double r14 = pose.position.x;
    double r24 = pose.position.y;
    double r34 = pose.position.z;

    // quaternion
    double q0 = pose.orientation.w;
    double q1 = pose.orientation.x;
    double q2 = pose.orientation.y;
    double q3 = pose.orientation.z;
    
    // Trans-Marix
    double r11 = 2.0 * (q0 * q0 + q1 * q1) - 1.0;
    double r12 = 2.0 * (q1 * q2 - q0 * q3);
    double r13 = 2.0 * (q1 * q3 + q0 * q2);
    double r21 = 2.0 * (q1 * q2 + q0 * q3);
    double r22 = 2.0 * (q0 * q0 + q2 * q2) - 1.0;
    double r23 = 2.0 * (q2 * q3 - q0 * q1);
    double r31 = 2.0 * (q1 * q3 - q0 * q2);
    double r32 = 2.0 * (q2 * q3 + q0 * q1);
    double r33 = 2.0 * (q0 * q0 + q3 * q3) - 1.0;

    *T = r11; T++;
    *T = r12; T++;
    *T = r13; T++;
    *T = r14; T++;
    *T = r21; T++;
    *T = r22; T++;
    *T = r23; T++;
    *T = r24; T++;
    *T = r31; T++;
    *T = r32; T++;
    *T = r33; T++;
    *T = r34; T++;
    *T = 0.0; T++;
    *T = 0.0; T++;
    *T = 0.0; T++;
    *T = 1.0; T++;

    return true;
}

