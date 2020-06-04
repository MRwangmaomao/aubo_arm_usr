
/*
 * @Author: 南山二毛
 * @Date: 2019-12-29 11:18:52
 * @LastEditTime : 2020-06-02 21:18:16
 * @LastEditors  : 南山二毛
 * @Description: In User Settings Edit
 * @FilePath: /catkin_ws/src/aubo_arm_usr/src/gpd_grasp.cpp
 */ 
#include <ros/ros.h>
#include <ros/console.h>

// opencv ros
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h> 
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
// pcl
#include <pcl/io/pcd_io.h>// 读写
#include <pcl/common/transforms.h>// 点云坐标变换
#include <pcl/point_types.h>      // 点类型
#include <pcl/filters/voxel_grid.h>// 体素格滤波
#include <pcl/filters/passthrough.h>//  直通滤波
#include <pcl/sample_consensus/method_types.h>// 采样一致性，采样方法
#include <pcl/sample_consensus/model_types.h>// 模型
#include <pcl/segmentation/sac_segmentation.h>// 采样一致性分割
#include <pcl/filters/extract_indices.h>// 提取点晕索引
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h> 
#include <pcl_conversions/pcl_conversions.h>
// c++ std
#include <memory>
#include <string> 
#include <vector>
#include <map> 
#include <thread>

#include <tf/transform_broadcaster.h>

// moveit
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>

#include <moveit_visual_tools/moveit_visual_tools.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "std_msgs/Int8.h"
#include "apriltag_ros/AprilTagDetection.h"
#include "apriltag_ros/AprilTagDetectionArray.h" 

#include "gpd_ros/detect_grasps.h"

using namespace Eigen; 

#define PI 3.1415926

// ========================================================
// 全局变量定义
// ========================================================
// 关节角度限制
const double Jlimit_min[6] = {-PI, -PI, -PI, -PI, -PI, -PI};
const double Jlimit_max[6] = {PI, PI, PI, PI, PI, PI};

// 机械臂连杆参数 
const double a2 =  0.408;
const double a3 =  0.376;
const double d1 =  0.122;
const double d2 =  0.1215;
const double d5 =  0.1025;
const double d6 =  0.094; 
const double ltool = 0.185;

// 跟踪目标的存在与更新变量
bool tags_find_flag = false; // 是否找到二维码
bool tags_update = false; // 是否更新二维码

// 抓取物体平面的像素和空间坐标变量
std::vector<apriltag_ros::AprilTagDetection> tags_corners;
 
// 彩色图像 深度图像  接收彩色图像触发标志 相机内参
cv::Mat kinect_color;
cv::Mat kinect_depth;
bool is_sub_kinect_cloud;
double fx = 571.8319822931295;
double fy = 436.7418038692657;
double cx = 568.5477678366276;
double cy = 259.9872860050479; 

// 坐标齐次变换变量
tf::StampedTransform Transform_target;
tf::StampedTransform Trans_Gripper_in_wrist3Link;
tf::StampedTransform Trans_wrist3Link_in_Gripper;

tf::StampedTransform Trans_Camera_in_wrist3Link;
tf::StampedTransform Trans_wrist3Link_in_Camera;
 
tf::StampedTransform Trans_Camera_in_robot_base;
geometry_msgs::Pose  Pose_Camera_in_robot_base;

geometry_msgs::Pose  Pose_target_obj_in_Camera; 
tf::StampedTransform Trans_base_link_in_robot_base;
tf::StampedTransform Trans_Camera_in_base_link;  
tf::StampedTransform Trans_QR_corner_in_robot_base;
tf::StampedTransform Trans_QR_corner_in_Camera;

// 自己定义的机械臂功能函数
#define ZERO_THRESH 0.001
int find_nearest_joint_index(double* q_sols, int slov_num, double* q_ref);
bool pose_to_mat(geometry_msgs::Pose pose, double* T);
void forward(const double* q, double* T);
int SIGN(double x);
int inverse(const double* T, double* q_sols, double q6_des); 
int inverse_jointlimited(const double* T, double* q_sols, double q6_des);
double antiSinCos(double sA, double cA);

void rosinfo_joint(std::vector<std::string> joint_names, std::vector<double> j_values);
void handle_tag_in_camera(const apriltag_ros::AprilTagDetectionArray::ConstPtr& tag);
void thread_function();
bool tarPose_nearJ_bestJ(geometry_msgs::Pose tarPose, std::vector<double> nearJ, std::vector<double> *bestJ);
bool tarTrans_nearJ_bestJ(tf::StampedTransform tarTrans, std::vector<double> nearJ, std::vector<double> *bestJ);
void ImageCallback (const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);

// ------------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------------------
//                                                        主函数
// ------------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------------------

/**
 * @brief 主函数
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char **argv)
{
    ros::init(argc, argv, "grasp_april_cube");
    ros::NodeHandle node_handle;
    ros::AsyncSpinner spinner(1);
    spinner.start();    // 开始进入ros消息回调机制


    // ---------------------------------
    // step1 初始化变量, 最终夹爪的目标位姿
    // ---------------------------------
    Transform_target.setOrigin(tf::Vector3(0, 0, 0));
    Transform_target.setRotation(tf::Quaternion(0, 0, 0, 1));
 
    // ---------------------------------
    // step2 注册话题和服务的订阅与发布
    // ---------------------------------
    ros::Subscriber sub_tag_detections = node_handle.subscribe("/tag_detections", 100, handle_tag_in_camera); // 二维码检测
    std_msgs::Int8 dhhand_msg;
    ros::Publisher dhhand_pub = node_handle.advertise<std_msgs::Int8>("/dh_hand", 10);  // 电动夹爪控制
    ros::Publisher point_cloud2_pub = node_handle.advertise<sensor_msgs::PointCloud2>("/gpd_grasp_pointcloud2", 10);  // 电动夹爪控制
    ros::ServiceClient gpd_ros_client = node_handle.serviceClient<gpd_ros::detect_grasps>("/detect_grasps"); // gpd客户端
 
    // kinect消息接收
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Subscriber<sensor_msgs::Image> * rgb_subscriber_ = new message_filters::Subscriber<sensor_msgs::Image> (node_handle, "/kinect2/qhd/image_color_rect", 10);
    message_filters::Subscriber<sensor_msgs::Image> * depth_subscriber_ = new message_filters::Subscriber<sensor_msgs::Image> (node_handle, "/kinect2/qhd/image_depth_rect", 10);
    message_filters::Synchronizer<sync_pol> * sync_ = new message_filters::Synchronizer<sync_pol> (sync_pol(10), *rgb_subscriber_, *depth_subscriber_);
    sync_->registerCallback(boost::bind(ImageCallback, _1, _2));

    // 张开电爪    
    dhhand_msg.data = 90;
    dhhand_pub.publish(dhhand_msg);

    // ---------------------------------------
    // step3 固连坐标系变换关系求解
    // ---------------------------------------
    tf::TransformListener listener; 
    tf::Transform ttf; // 临时变量

    bool tf_OK = false;
    while (tf_OK == false)
    {
        try{
            listener.waitForTransform("wrist3_Link", "dh_grasp_link", ros::Time(0), ros::Duration(1.0));
            listener.lookupTransform("wrist3_Link", "dh_grasp_link", ros::Time(0), Trans_Gripper_in_wrist3Link);
            tf_OK = true;
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
            ros::Duration(1.0).sleep();
        }
        ros::Duration(1.0).sleep();
        ROS_INFO("get_Trans_Gripper_in_wrist3Link");
    }

    ttf = Trans_Gripper_in_wrist3Link.inverse();
    Trans_wrist3Link_in_Gripper.setData(ttf);
 
    // 确定相机坐标和机械臂末端是否在tf树上 
    tf_OK = false;
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

    tf_OK = false;
    while (tf_OK == false)
    {
        try{
            listener.waitForTransform("robot_base", "base_link", ros::Time(0), ros::Duration(1.0));
            listener.lookupTransform("robot_base", "base_link", ros::Time(0), Trans_base_link_in_robot_base);
            tf_OK = true;
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
            ros::Duration(1.0).sleep();
        }
        ros::Duration(1.0).sleep();
        ROS_INFO("get_Trans_base_link_in_robot_base");
    }

    // ---------------------------------
    // step4 moveit相关模块初始化
    // ---------------------------------
    static const std::string PLANNING_GROUP = "manipulator_i5"; // moveit规划组

    moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP); // movegroup 接口对象

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface; // 规划场景 接口对象

    // Raw pointers are frequently used to refer to the planning group for improved performance.
    const robot_state::JointModelGroup* joint_model_group =
        move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP);  // 关节组 对象

    const std::vector<std::string>& joint_names = joint_model_group->getVariableNames(); // 各个关节名称

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description"); // 机器人模型对象
    
    robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();  // 机器人对应的运动学模型
    ROS_INFO_STREAM("model_frame is:" << kinematic_model->getModelFrame().c_str());

    robot_state::RobotStatePtr kinematic_state(new robot_state::RobotState(kinematic_model)); // 运动学模型下的机器人状态
    kinematic_state->setToDefaultValues();

    const Eigen::Affine3d &end_effector_state = kinematic_state->getGlobalLinkTransform("wrist3_Link"); // 获得当前机器人的末端
    ROS_INFO_STREAM("Translation: " << end_effector_state.translation());
    ROS_INFO_STREAM("Rotation: " << end_effector_state.rotation());
    
    moveit::planning_interface::MoveGroupInterface::Plan my_plan; // moveit 运动规划接口 对象


    // The package MoveItVisualTools provides many capabilties for visualizing objects, robots,
    // and trajectories in RViz as well as debugging tools such as step-by-step introspection of a script/
    // 定义rviz可视化对象
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

    // 机器人速度控制
    move_group.setMaxVelocityScalingFactor(0.15);
    move_group.setMaxAccelerationScalingFactor(0.1);

    // -------------------------------------------
    // step5 变量的各种定义
    // -------------------------------------------
    std::vector<double> joint_safe = {86.806916/57.3, -32.096968/57.3, -54.607499/57.3, 40.357928/57.3, -90.535073/57.3, -4.269872/57.3}; // 机器人的安全位置
    std::vector<double> joint_place_p1 = {86.8193/57.3, 23.1162/57.3, -72.1184/57.3, -1.72438/57.3, -90.5764/57.3, -5.0056/57.3}; // 机器人放置物体的位置
    std::vector<double> joint_group_seekbase1 = {82.205172/57.3, 9.583464/57.3, -34.987446/57.3, 35.508641/57.3, -89.745284/57.3, -4.482136/57.3}; // 搜索物体 拍照位置1
    std::vector<double> joint_group_seekbase2 = {39.885310/57.3, 11.211055/57.3, -35.072530/57.3, 50.559430/57.3, -74.059379/57.3, -29.007264/57.3}; // 搜索物体 拍照位置2
    std::vector<double> joint_group_seekbase3 = {82.205172/57.3, 9.583464/57.3, -34.987446/57.3, 35.508641/57.3, -89.745284/57.3, -4.482136/57.3}; // 搜索物体 拍照位置3
    std::vector<std::vector<double>> all_joint_group_seekbase;
    all_joint_group_seekbase.push_back(joint_group_seekbase1);
    all_joint_group_seekbase.push_back(joint_group_seekbase2);
    all_joint_group_seekbase.push_back(joint_group_seekbase3);
    uint16_t take_photo_time = 1; // 拍照次数设置

     
    // 机器人放置物体的固定位姿
    geometry_msgs::Pose place_pose0;    
    place_pose0.position.x = -0.0017;
    place_pose0.position.y = 0.3005;
    place_pose0.position.z = 0.1962; 
    place_pose0.orientation.x = -0.6571; 
    place_pose0.orientation.y = 0.2649; 
    place_pose0.orientation.z = 0.6541; 
    place_pose0.orientation.w = 0.2649;

    // 通用变量:计算过程变量
    double* T = new double[16];
    double q_sols[8*6];
    double jq[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int num_sols;
    tf::Quaternion q;
    q.setRPY(0.0, 0.0, 0.0);

    // 通用变量:获取当前状态
    geometry_msgs::Pose pose_current;
    geometry_msgs::Pose pose_base_link_current;
    std::vector<double> joint_current = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // 物体坐标相关变量 
    tf::StampedTransform transfrom_T;
    tf::StampedTransform transform_tar; // 机械臂到达目标位置的位姿变换 
    //tf::StampedTransform transform_tool0;

    // 相机坐标系下目标物体的位姿
    geometry_msgs::Pose pose_Obj;
    double* T_Obj = new double[16];
       
    // --------------------------------------------------------------------------------------------------------------------------------
    // step6 while主循环寻找目标物体, 发送gpd_ros夹爪位姿生成服务请求和等待消息回应, 获得最佳位姿后控制机械臂运动  
    // --------------------------------------------------------------------------------------------------------------------------------
    bool search_done = false;
    ros::Rate rate(1.0);
    int detect_times = 0;
    is_sub_kinect_cloud = false;
    while(search_done == false && ros::ok())
    {
        // 查找到二维码
        if(tags_find_flag)
        {
            sensor_msgs::PointCloud2 kinect_pointcloud; 
            pcl::PointCloud<pcl::PointXYZRGB> all_cloud;
            gpd_ros::detect_grasps detect_grasp_req;
            kinect_pointcloud.header.frame_id = "/robot_base"; // 点云的frame
            kinect_pointcloud.header.stamp = ros::Time::now();
            tags_find_flag = false;
            float desk_height = 0;
            // --------------------------------
            // step6.1 拍照片,提取感兴趣区域,合成点云,将点云转换到机器人坐标系下
            // --------------------------------
            // visual_tools.prompt("Press 'next' : ready to seek object ! ");
            // 设定视觉搜索物体的起始位置
            for(uint16_t take_photo_i = 0; take_photo_i < take_photo_time; take_photo_i++){
                // step6.1.1 运动到安全位置
                printf("move to the start pose: joint_group_seekbase \n");
                move_group.setJointValueTarget(all_joint_group_seekbase[take_photo_i]); 
                move_group.plan(my_plan);
                move_group.execute(my_plan);
                ros::Duration(1.0).sleep(); // 移动完成后停留1s拍照
                // visual_tools.prompt("next step");
                

                // step6.1.2 计算ROI大小 
                // 采集一次图像,根据二维码坐标生成区域
                std::cout << "检测到角点1:" << tags_corners[0].center_point[0] << " " << tags_corners[0].center_point[1] << std::endl;
                std::cout << "检测到角点2:" << tags_corners[1].center_point[0] << " " << tags_corners[1].center_point[1] << std::endl;
 
                uint16_t x_pixel_start, x_pixel_end, y_pixel_start, y_pixel_end;
                if(tags_corners[0].center_point[0] < tags_corners[1].center_point[0]){
                    x_pixel_start = tags_corners[0].center_point[0];
                    x_pixel_end = tags_corners[1].center_point[0];
                }else{
                    x_pixel_end = tags_corners[0].center_point[0];
                    x_pixel_start = tags_corners[1].center_point[0];
                }

                if(tags_corners[0].center_point[1] < tags_corners[1].center_point[1]){
                    y_pixel_start = tags_corners[0].center_point[1];
                    y_pixel_end = tags_corners[1].center_point[1];
                }else{
                    y_pixel_end = tags_corners[0].center_point[1];
                    y_pixel_start = tags_corners[1].center_point[1];
                }
 
                 // step6.1.4 对ROI进行点云合成 提取区域内的点云
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
                for ( int m=y_pixel_start; m<y_pixel_end; m+=1 )// 每一行
                {
                    for ( int n=x_pixel_start; n<x_pixel_end; n+=1 )//每一列
                    { 
                        pcl::PointXYZRGB p; 
                        
                        float d = double(kinect_depth.ptr<unsigned short>(m)[n])/1000.0;// 深度 m为单位 保留0～2m内的点 
                         
                        if (d < 0.1 || d>6.0) // 相机测量范围 0.5～6m 
                            continue;  
                        float y = ( m - cy) * d / fy;
                        if(y<-3.0 || y>3.0) continue;// 保留 垂直方向 -3～3m范围内的点 
                        int ind = m * kinect_depth.cols + n;// 总索引
                        p.z = d;
                        p.x = ( n - cx) * d / fx;
                        p.y = y; 
                        p.b = kinect_color.ptr<uchar>(m)[n*3+0];// 点颜色=====
                        p.g = kinect_color.ptr<uchar>(m)[n*3+1];
                        p.r = kinect_color.ptr<uchar>(m)[n*3+2]; 
                        cloud->push_back(p);
                        // std::cout << p << std::endl;
                    }
                }
                // std::cout << "cloud的点云数量为: " << cloud->size() << std::endl;


                // step6.1.3 计算得到QR在相机中的tf, 获得相机和基座的tf关系 QR在机器人中的tf关系
                geometry_msgs::Pose Pose_QR_corner_in_Camera = tags_corners[0].pose.pose.pose;
                Trans_QR_corner_in_Camera.setOrigin(tf::Vector3( 
                                                            Pose_QR_corner_in_Camera.position.x,  //x轴修正-0.15
                                                            Pose_QR_corner_in_Camera.position.y,         //y轴修正0.1
                                                            Pose_QR_corner_in_Camera.position.z));
                Trans_QR_corner_in_Camera.setRotation(tf::Quaternion(Pose_QR_corner_in_Camera.orientation.x, 
                                                                Pose_QR_corner_in_Camera.orientation.y,
                                                                Pose_QR_corner_in_Camera.orientation.z,
                                                                Pose_QR_corner_in_Camera.orientation.w));
                
                tf_OK = false;
                while (tf_OK == false)
                {
                    try{
                        listener.waitForTransform("base_link", "kinect2_rgb_optical_frame", ros::Time(0), ros::Duration(1.0));
                        listener.lookupTransform("base_link", "kinect2_rgb_optical_frame", ros::Time(0), Trans_Camera_in_base_link);
                        tf_OK = true;
                    }
                    catch (tf::TransformException ex)
                    {
                        ROS_ERROR("%s", ex.what());
                        ros::Duration(1.0).sleep();
                    }
                    ros::Duration(1.0).sleep();
                    ROS_INFO("get_Trans_Camera_in_base_link");
                } 
                // 获取在机器人坐标中的相机位姿
                Trans_Camera_in_robot_base.mult(Trans_base_link_in_robot_base, Trans_Camera_in_base_link); 
                // 获取QR在机器人中的tf关系
                Trans_QR_corner_in_robot_base.mult(Trans_Camera_in_robot_base, Trans_QR_corner_in_Camera); 
                // 获得桌子的高度
                desk_height = Trans_QR_corner_in_robot_base.getOrigin().getZ(); 
                std::cout << "desk_height is " << desk_height << std::endl;

 
                // step 6.1.5 将pcl点云从相机坐标系下转变到移动机器人坐标系下
                Eigen::Matrix4d T_robot_camera(Eigen::Matrix4d::Identity());   
                T_robot_camera(0,3) = Trans_Camera_in_robot_base.getOrigin().getX(); 
                T_robot_camera(1,3) = Trans_Camera_in_robot_base.getOrigin().getY(); 
                T_robot_camera(2,3) = Trans_Camera_in_robot_base.getOrigin().getZ(); 
                Eigen::Quaterniond q_robot_camera; 
                q_robot_camera.w() = static_cast<double>(Trans_Camera_in_robot_base.getRotation().getW());
                q_robot_camera.x() = static_cast<double>(Trans_Camera_in_robot_base.getRotation().getX());
                q_robot_camera.y() = static_cast<double>(Trans_Camera_in_robot_base.getRotation().getY());
                q_robot_camera.z() = static_cast<double>(Trans_Camera_in_robot_base.getRotation().getZ()); 
                
                T_robot_camera.block(0,0,3,3) = q_robot_camera.toRotationMatrix(); 
                
                pcl::PointCloud<pcl::PointXYZRGB> temp;
                std::cout << "将点云转到机器人坐标系下: " << std::endl;
                pcl::transformPointCloud( *cloud, temp, T_robot_camera);
                
                for(uint32_t cloud_index = 0; cloud_index < temp.size(); cloud_index++){
                    if(temp[cloud_index].z > desk_height+0.020){
                        all_cloud.push_back(temp[cloud_index]);
                    }
                }  // 将点云加入到总点云中  *all_cloud += temp; 

                // 相机的坐标点
                geometry_msgs::Point camera_pose;
                camera_pose.x = 0.0;
                camera_pose.y = 0.0;
                camera_pose.z = 0.0;
                detect_grasp_req.request.cloud_indexed.cloud_sources.view_points.push_back(camera_pose); 
                // detect_grasp_req.request.cloud_indexed.cloud_sources.camera_source[take_photo_i].data = 0;
            }


            std::cout << "all_cloud 点云总共有 " << all_cloud.size() << " 个点!" << std::endl;
            if(all_cloud.size() > 1){  // 点云数量太少
                // --------------------------------
                // step6.2 合成detect_grasps的服务消息,请求服务后等待消息
                // -------------------------------- 
                // 将pcl格式点云转为pointcloud2格式 
                
                pcl::toROSMsg<pcl::PointXYZRGB>(all_cloud, kinect_pointcloud); // 这是个模板函数,编译器居然不会对没写模板参数报错 , 函数的参数方向也搞反了,怪不得导致pcl的点云也是空的了 调试到了凌晨2点
                std::cout << "转换为pointcloud2点云: " << std::endl;
                detect_grasp_req.request.cloud_indexed.cloud_sources.cloud = kinect_pointcloud;
                kinect_pointcloud.header.frame_id = "robot_base";
                point_cloud2_pub.publish(kinect_pointcloud);
                // 采样点索引
                // for(int64 i = 0; i <= 1000; i++)
                //     detect_grasp_req.request.cloud_indexed.indices.data.push_back(i);
                std::cout << "等待gpd服务......: " << std::endl;
                // 开始向服务发送请求    
                if (gpd_ros_client.call(detect_grasp_req))
                { 
                    // --------------------------------
                    // step6.3 收到服务响应后对数据进行解析,选取分数前两位,筛选夹爪的位姿是否满足需要
                    // --------------------------------
                    // 找到分数最高的结果
                    uint16_t grasp_num = detect_grasp_req.response.grasp_configs.grasps.size();
                    float score = -1;
                    uint16_t max_score_id = -1;
                    for(int i = 0; i < grasp_num; i++){
                        if(detect_grasp_req.response.grasp_configs.grasps[i].score.data > score){
                            max_score_id = i;
                        }
                    }
                    std::cout << "最高分为:" << detect_grasp_req.response.grasp_configs.grasps[max_score_id].score.data << std::endl;

                    // 根据目前的信息进行筛选,比如电爪末端和中心不能低于桌子高度
                    if(desk_height > detect_grasp_req.response.grasp_configs.grasps[max_score_id].position.z - 0.02 ||
                        desk_height > detect_grasp_req.response.grasp_configs.grasps[max_score_id].approach.z - 0.04 ||
                        desk_height > detect_grasp_req.response.grasp_configs.grasps[max_score_id].binormal.z)
                    {
                        std::cout << "生成的夹爪位姿不合理,有可能会和桌面发生碰撞" << std::endl;
                    }
                    std::cout << "目标夹爪的位姿: " << detect_grasp_req.response.grasp_configs.grasps[max_score_id].approach << std::endl;
                    // --------------------------------
                    // step6.4 根据最优的夹爪位姿 计算出机械臂的末端状态,控制机械臂移动至目标位置
                    // --------------------------------
                    // step6.4.1 计算夹爪的靠近位姿,这是一个安全过度点,计算机械臂的末端位姿,通过逆运动学求出六轴的角度,控制机械臂运动
                    geometry_msgs::Pose dh_grasp_target_approach_pose;
                    dh_grasp_target_approach_pose.position.x = detect_grasp_req.response.grasp_configs.grasps[max_score_id].approach.x;
                    dh_grasp_target_approach_pose.position.y = detect_grasp_req.response.grasp_configs.grasps[max_score_id].approach.y;
                    dh_grasp_target_approach_pose.position.z = detect_grasp_req.response.grasp_configs.grasps[max_score_id].approach.z;
                    std::cout << "夹爪的最终生成位姿为:" << dh_grasp_target_approach_pose << std::endl;
                    
                    // step6.4.2 计算夹爪的最终位置,然后求出末端的位姿,求出通过逆运动学求出六轴的角度,控制机械臂运动
                    

                    // step6.4.3 控制电动夹爪闭合,抓取物体

                    // step6.4.4 回到安全过渡位置

                    // step6.4.5 移动到盒子上方

                    // step6.4.6 电动夹爪松开

                    /************************************* 一个抓取回合完成 ***********************************/
                }
                else // 请求失败,可能是消息格式不对,也可能是服务程序没能正常运行
                {
                    ROS_ERROR("Failed to call service Service_demo");
                    // return 1;
                }
            }else{
                ROS_ERROR("gpd点云检测不好,请确保桌面是否摆放了物体");
                detect_times += 10;
            } // if(all_cloud.size() < 1)
              
            detect_times++;
            all_cloud.clear();
        }else{
            std::cout << "无法看到二维码, 请移动确认相机视野中存在二维码!!!" << std::endl;
        } // 查找到二维码 if(tags_find_flag)
        
        
        if(detect_times >= 20 ){
            search_done == true;        
        }
        ros::spinOnce();
        ros::Duration(1.0).sleep(); // 休息1s继续抓取
    } // while(search_done == false && ros::ok())

    printf("---系统结束--------\n");
     
    // 运动完毕后,清除约束   
    move_group.clearPathConstraints(); 
    ros::shutdown();
    return 0;
}

// ========================================================================================================================================================================= 
//                                  ======================================================================================
//                                                                      私 有 函 数 定 义
//                                  ======================================================================================
// =========================================================================================================================================================================
 

/**
 * @brief 接收AprilTag消息回调函数
 * 
 * @param tag 接收到的消息
 */ 
void handle_tag_in_camera(const apriltag_ros::AprilTagDetectionArray::ConstPtr& tag)
{ 
    static uint16_t loseCnt = 0;

    uint16_t num_of_tag = tag->detections.size();

    std::vector<int> v_id1; // 专门存放581 和582 tag
    std::vector<int> v_id2; // 专门存放583 和584 tag

    // 遍历所有看到的二维码,对应id存入
    if (num_of_tag >= 2) // 至少看到两个二维码
    {
        // 寻找距离最近的tag索引号 
        for(uint16_t i = 0; i < num_of_tag; i++){
            if(tag->detections[i].id[0] == 581 || tag->detections[i].id[0] == 582)
                v_id1.push_back(tag->detections[i].id[0]);
            if(tag->detections[i].id[0] == 583 || tag->detections[i].id[0] == 584)
                v_id2.push_back(tag->detections[i].id[0]);
        } 
    }

    else
    {  
        // 丢失检测
        loseCnt ++;
        if (loseCnt > 10)
        { 
            loseCnt = 0; 
        }
        tags_find_flag = false;
        return;
    }

    // 发现左上角后右下角 
    if(v_id1.size() == 2){
        tags_find_flag = true;
        tags_corners.clear(); // 先清除之前的内容
        for(int i = 0; i < num_of_tag; i++){
            if(tag->detections[i].id[0] == 581 || tag->detections[i].id[0] == 582)
                tags_corners.push_back(tag->detections[i]);
        }
    }
    // 右上角和左下角
    else if(v_id2.size() == 2){ 
        tags_find_flag = true;
        tags_corners.clear(); // 先清除之前的内容
        for(int i = 0; i < num_of_tag; i++){  // 将对应的位置数据存入
            if(tag->detections[i].id[0] == 583 || tag->detections[i].id[0] == 584)
                tags_corners.push_back(tag->detections[i]);
        }
    }
    else{
        tags_find_flag = false;
    }  
}

/**
 * @brief kinect相机的回调函数
 * 
 * @param msgRGB 
 * @param msgD 
 */
void ImageCallback (const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD){
    
    // if(is_sub_kinect_cloud){
        // std::cout << "开始接收图像" << std::endl;
        cv_bridge::CvImageConstPtr cv_ptrRGB;
        try {
            cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv_bridge::CvImageConstPtr cv_ptrD;
        try {
            cv_ptrD = cv_bridge::toCvShare(msgD);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return; 
        }  
        kinect_color = cv_ptrRGB->image.clone(); 
        kinect_depth = cv_ptrD->image.clone(); 
        is_sub_kinect_cloud = false; // 接收完图像之后值为false  
    // }
}

/**
 * @brief thread_function 函数
 * 
 * @param   
 */ 
void thread_function()
{
  tf::TransformBroadcaster br;

  ros::Rate rate(1.0);
  while (1)
  {
    // 将目标位姿发布至tf-tree,供显示      
    br.sendTransform(tf::StampedTransform(Transform_target, ros::Time::now(), "/base_link", "target"));
    rate.sleep();
  }
}

/**
 * @brief 
 * 
 * @param q_sols 
 * @param slov_num 
 * @param q_ref 
 * @return int 
 */
int find_nearest_joint_index(double* q_sols, int slov_num, double* q_ref)
{
    int index = 0;
    double error[slov_num];
    double err_min = 0.0;
    int min_index = 0;
    for (index = 0; index < slov_num; index++)
    {
       error[index] = std::fabs(q_sols[index*6 + 0] - q_ref[0])
                    + std::fabs(q_sols[index*6 + 1] - q_ref[1])
                    + std::fabs(q_sols[index*6 + 2] - q_ref[2])
                    + std::fabs(q_sols[index*6 + 3] - q_ref[3])
                    + std::fabs(q_sols[index*6 + 4] - q_ref[4])
                    + std::fabs(q_sols[index*6 + 5] - q_ref[5]);
       if (index == 0)
       {
           err_min = error[index];
           min_index = 0;
       }
       else
       {
           if (err_min > error[index])
           {
               err_min = error[index];
               min_index = index;
           }
       }
       // printf("error[%d] = %1.4f \n", index, error[index]);
    }
    return min_index;
}

/**
 * @brief 
 * 
 * @param pose 
 * @param T 
 * @return true 
 * @return false 
 */
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

/**
 * @brief 
 * 
 * @param q 
 * @param T 
 */
void forward(const double* q, double* T)
{
    double  q1 = *q;q++;
    double  q2 = *q;q++;
    double  q3 = *q;q++;
    double  q4 = *q;q++;
    double  q5 = *q;q++;
    double  q6 = *q;q++;
    double  C1 = cos(q1), C2 = cos(q2), C4 = cos(q4), C5 = cos(q5), C6 = cos(q6);
    double  C23 = cos(q2 - q3), C234 = cos(q2 - q3 + q4), C2345 = cos(q2 - q3 + q4 - q5), C2345p = cos(q2 - q3 + q4 + q5);
    double  S1 = sin(q1), S2 = sin(q2), S4 = sin(q4), S5 = sin(q5), S6 = sin(q6);
    double  S23 = sin(q2 - q3), S234 = sin(q2 - q3 + q4);

    *T = -C6 * S1 * S5 + C1 * (C234 * C5 * C6 - S234 * S6);T++;
    *T = S1 * S5 * S6 - C1 * (C4 * C6 * S23 + C23 * C6 * S4 + C234 * C5 * S6);T++;
    *T = C5 * S1 + C1 * C234 * S5;T++;
    *T = (d2 + C5 * d6) * S1 - C1 * (a2 * S2 + (a3 + C4 * d5) * S23 + C23 * d5 * S4 - C234 * d6 * S5);T++;

    *T = C234 * C5 * C6 * S1 + C1 * C6 * S5 - S1 * S234 * S6;T++;
    *T = -C6 * S1 * S234 - (C234 * C5 * S1 + C1 * S5) * S6;T++;
    *T = -C1 * C5 + C234 * S1 * S5;T++;
    *T = -C1 * (d2 + C5 * d6) - S1 * (a2 * S2 + (a3 + C4 * d5) * S23 + C23 * d5 * S4 - C234 * d6 * S5);T++;

    *T = C5 * C6 * S234 + C234 * S6;T++;
    *T = C234 * C6 - C5 * S234 * S6;T++;
    *T = S234 * S5;T++;
    *T = d1 + a2 * C2 + a3 * C23 + d5 * C234 + d6 * C2345/2 - d6 * C2345p / 2;T++;
    *T = 0;T++;
    *T = 0;T++;
    *T = 0;T++;
    *T = 1; 
}
 
/**
 * @brief 
 * 
 * @param x 
 * @return int 
 */
int SIGN(double x) {
      return (x > 0) - (x < 0);
    }
 
/**
 * @brief 
 * 
 * @param sA 
 * @param cA 
 * @return double 
 */
double antiSinCos(double sA, double cA)
{
    double eps = 1e-8;
    double angle = 0;
    if((fabs(sA) < eps)&&(fabs(cA) < eps))
    {
        return 0;
    }
    if(fabs(cA) < eps)
        angle = M_PI/2.0*SIGN(sA);
    else if(fabs(sA) < eps)
    {
        if (SIGN(cA) == 1)
            angle = 0;
        else
            angle = M_PI;
    }
    else
    {
        angle = atan2(sA, cA);
    }

    return angle;
}

/**
 * @brief 
 * 
 * @param T 
 * @param q_sols 
 * @param q6_des 
 * @return int 
 */
int inverse(const double* T, double* q_sols, double q6_des)
{
    bool singularity = false;

    int num_sols = 0;
    double nx = *T;T++; double ox = *T;T++; double ax = *T;T++; double px = *T;T++;
    double ny = *T;T++; double oy = *T;T++; double ay = *T;T++; double py = *T;T++;
    double nz = *T;T++; double oz = *T;T++; double az = *T;T++; double pz = *T;

    //////////////////////// shoulder rotate joint (q1) //////////////////////////////
    VectorXd q1(2);

    double A1 = d6 * ay - py;
    double B1 = d6 * ax - px;
    double R1 = A1 * A1 + B1 * B1 - d2 * d2;


    if(R1 < 0.0)
        return num_sols;
    else
    {
        double R12 = sqrt(R1);
        q1(0) =  antiSinCos(A1, B1) -  antiSinCos(d2, R12);
        q1(1) =  antiSinCos(A1, B1) -  antiSinCos(d2, -R12);
        for(int i = 0; i < 2; i++)
        {
            while(q1(i) > M_PI)
                q1(i) -= 2 * M_PI;
            while(q1(i) < -M_PI)
                q1(i) += 2 * M_PI;
        }
    }

    ////////////////////////////// wrist 2 joint (q5) //////////////////////////////
    MatrixXd q5(2,2);

    for(int i = 0; i < 2; i++)
    {

        double C1 = cos(q1(i)), S1 = sin(q1(i));
        double B5 = -ay * C1 + ax * S1;
        double M5 = (-ny * C1 + nx * S1);
        double N5 = (-oy * C1 + ox * S1);

        double R5 = sqrt(M5 * M5 + N5 * N5);

        q5(i,0) = antiSinCos(R5, B5);
        q5(i,1) = antiSinCos(-R5, B5);
    }

    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////// wrist 3 joint (q6) //////////////////////////////
    double q6;
    VectorXd q3(2), q2(2), q4(2);

    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            // wrist 3 joint (q6) //
            double C1 = cos(q1(i)), S1 = sin(q1(i));
            double S5 = sin(q5(i,j));

            double A6 = (-oy * C1 + ox * S1);
            double B6 = (ny * C1 - nx * S1);

            if(fabs(S5) < ZERO_THRESH) //the condition is only dependent on q1
            {
                singularity = true;
                break;
            }
            else
                q6 = antiSinCos(A6 * S5, B6 * S5);

            /////// joints (q3,q2,q4) //////
            double C6 = cos(q6);
            double S6 = sin(q6);

            double pp1 = C1 * (ax * d6 - px + d5 * ox * C6 + d5 * nx * S6) + S1 * (ay * d6 - py + d5 * oy * C6 + d5 * ny * S6);
            double pp2 = -d1 - az * d6 + pz - d5 * oz * C6 - d5 * nz * S6;
            double B3 = (pp1 * pp1 + pp2 * pp2 - a2 * a2 - a3 * a3) / (2 * a2 * a3);


            if((1 - B3 * B3) < ZERO_THRESH)
            {
                singularity = true;
                continue;
            }
            else
            {
                double Sin3 = sqrt(1 - B3 * B3);
                q3(0) = antiSinCos(Sin3, B3);
                q3(1) = antiSinCos(-Sin3, B3);
            }

            for(int k = 0; k < 2; k++)
            {

                double C3 = cos(q3(k)), S3 = sin(q3(k));
                double A2 = pp1 * (a2 + a3 * C3) + pp2 * (a3 * S3);
                double B2 = pp2 * (a2 + a3 * C3) - pp1 * (a3 * S3);

                q2(k) = antiSinCos(A2, B2);

                double C2 = cos(q2(k)), S2 = sin(q2(k));

                double A4 = -C1 * (ox * C6 + nx * S6) - S1 * (oy * C6 + ny * S6);
                double B4 = oz * C6 + nz * S6;
                double A41 = pp1 - a2 * S2;
                double B41 = pp2 - a2 * C2;

                q4(k) = antiSinCos(A4, B4) - antiSinCos(A41, B41);
                while(q4(k) > M_PI)
                    q4(k) -= 2 * M_PI;
                while(q4(k) < -M_PI)
                    q4(k) += 2 * M_PI;

                q_sols[num_sols*6+0] = q1(i);    q_sols[num_sols*6+1] = q2(k);
                q_sols[num_sols*6+2] = q3(k);    q_sols[num_sols*6+3] = q4(k);
                q_sols[num_sols*6+4] = q5(i,j);  q_sols[num_sols*6+5] = q6;
                num_sols++;
            }
        }
    }

    return num_sols;
} 

/**
 * @brief 
 * 
 * @param T 
 * @param q_sols 
 * @param q6_des 
 * @return int 
 */
int inverse_jointlimited(const double* T, double* q_sols, double q6_des)
{
  int num_sols = inverse(T, q_sols, q6_des);
  for(int i=0; i<num_sols; i++)
  {
      for (int j=0; j<6; j++)
      {
        if (q_sols[i*6+j] > Jlimit_max[j])
        {
            q_sols[i*6+j] -= 2*PI;
        }
        else if (q_sols[i*6+j] < Jlimit_min[j])
        {
            q_sols[i*6+j] += 2*PI;
        }
        else
        {
            // none
        }
      }
  }
  return num_sols;
}

/**
 * @brief 
 * 
 * @param joint_names 
 * @param j_values 
 */
void rosinfo_joint(std::vector<std::string> joint_names, std::vector<double> j_values)
{
   ROS_INFO("--- this is a new ros info message ------");
  for (std::size_t j = 0; j < joint_names.size(); j++)
  {
      ROS_INFO("Joint %s: %f", joint_names[j].c_str(), j_values[j]);
  }
}

/**
 * @brief 
 * 
 * @param tarPose 
 * @param nearJ 
 * @param bestJ 
 * @return true 
 * @return false 
 */
bool tarPose_nearJ_bestJ(geometry_msgs::Pose tarPose, std::vector<double> nearJ, std::vector<double> *bestJ)
{
  double* trans = new double[16];
  double q_sols[8*6];
  double q_near[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int num_sols;
  int near_index;

  pose_to_mat(tarPose, trans);    
     
  num_sols = inverse(trans, q_sols, 0.0);
  if (num_sols == 0)
  {
      return false;
  }

  for(int i=0; i<num_sols; i++)
  {
      for (int j=0; j<6; j++)
      {
          if (q_sols[i*6+j] > Jlimit_max[j])
          {
              q_sols[i*6+j] -= 2*PI;
          }
          else if (q_sols[i*6+j] < Jlimit_min[j])
          {
              q_sols[i*6+j] += 2*PI;
          }
          else
          {
              // none
          }
      }      
      // printf("%1.6f %1.6f %1.6f %1.6f %1.6f %1.6f\n", 
      // q_sols[i*6+0], q_sols[i*6+1], q_sols[i*6+2], q_sols[i*6+3], q_sols[i*6+4], q_sols[i*6+5]);
  }

  for (int i=0; i <6; i ++)
  {
      q_near[i] = (double)(nearJ[i]);
  }
  near_index = find_nearest_joint_index(q_sols, num_sols, q_near);
  // printf("near_index is :%d \n", near_index);

  for (int i = 0; i <6; i ++)
  {
      (*bestJ)[i] = (double)(q_sols[near_index*6+i]);
      // printf("bestJ[%d] = %1.4f \n", i, (*bestJ)[i]);
  }

  return true;
}

/**
 * @brief 
 * 
 * @param tarTrans 
 * @param nearJ 
 * @param bestJ 
 * @return true 
 * @return false 
 */
bool tarTrans_nearJ_bestJ(tf::StampedTransform tarTrans, std::vector<double> nearJ, std::vector<double> *bestJ)
{
  geometry_msgs::Pose tarPose;
  tarPose.position.x = tarTrans.getOrigin().x();
  tarPose.position.y = tarTrans.getOrigin().y();
  tarPose.position.z = tarTrans.getOrigin().z();
  tarPose.orientation.x = tarTrans.getRotation().x();
  tarPose.orientation.y = tarTrans.getRotation().y();
  tarPose.orientation.z = tarTrans.getRotation().z();
  tarPose.orientation.w = tarTrans.getRotation().w();
  return tarPose_nearJ_bestJ(tarPose, nearJ, bestJ);
}


