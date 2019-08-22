

/* Author: wang prirong 2019-8-22 */
#include <ros/ros.h>
#include <ros/console.h>

#include <cv_bridge/cv_bridge.h>
// #include <XmlRpcException.h>

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

#include "std_msgs/Int8.h"
#include "apriltag_ros/AprilTagDetection.h"
#include "apriltag_ros/AprilTagDetectionArray.h" 

using namespace Eigen; 

#define PI 3.1415926

// ========================================================
// 全局变量定义
// ========================================================
// 关节角度限制
const double Jlimit_min[6] = {-PI, -PI, -PI, -PI, -PI, -PI};
const double Jlimit_max[6] = {PI, PI, PI, PI, PI, PI};

// 机械臂连杆参数
// const double d1 =  0.122;
// const double a2 =  0.408;
// const double a3 =  0.376; 
// const double d4 =  0.1215;
// const double d5 =  0.1025;
// const double d6 =  0.094;

const double a2 =  0.408;
const double a3 =  0.376;
const double d1 =  0.122;
const double d2 =  0.1215;
const double d5 =  0.1025;
const double d6 =  0.094;
// const double d1 =  0.1518;
// const double a2 = -0.2435;
// const double a3 = -0.213025;
// const double d4 =  0.13095;
// const double d5 =  0.0855;
// const double d6 =  0.09195;
const double ltool = 0.185;

// 跟踪目标的存在与更新变量
bool object_flag = false;
bool object_update = false;
// 跟踪目标的像素变量
float object_u;
float object_v;



// 坐标齐次变换变量
tf::StampedTransform Transform_target;
tf::StampedTransform Trans_Gripper_in_wrist3Link;
tf::StampedTransform Trans_wrist3Link_in_Gripper;
tf::StampedTransform Trans_Camera_in_wrist3Link;
tf::StampedTransform Trans_wrist3Link_in_Camera;
tf::StampedTransform Trans_QRcode_in_Camera;
geometry_msgs::Pose  Pose_QRcode_in_Camera;
tf::StampedTransform Trans_Camera_in_World;
tf::StampedTransform Trans_base_link_in_World;
tf::StampedTransform Trans_Camera_in_base_link;

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


int main(int argc, char **argv)
{
    ros::init(argc, argv, "grasp_april_cube");
    ros::NodeHandle node_handle;
    ros::AsyncSpinner spinner(1);
    spinner.start();

    // 初始化变量
    Transform_target.setOrigin(tf::Vector3(0, 0, 0));
    Transform_target.setRotation(tf::Quaternion(0, 0, 0, 1));


    // 创建一个线程,用来定时显示目标坐标系;
    std::thread threadObj(thread_function);
    
    // ---------------------------------
    // 话题订阅与发布
    // ---------------------------------
    ros::Subscriber sub_tag_detections = node_handle.subscribe("/tag_detections", 100, handle_tag_in_camera);
    std_msgs::Int8 dhhand_msg;
    ros::Publisher dhhand_pub = node_handle.advertise<std_msgs::Int8>("/dh_hand", 1);
      
    tf::TransformListener listener; 

    // ---------------------------------------
    // 固连坐标系变换关系求解
    // ---------------------------------------
    tf::Transform ttf;
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

    // ^^^^^^^^^^^^^
    //
    // The package MoveItVisualTools provides many capabilties for visualizing objects, robots,
    // and trajectories in RViz as well as debugging tools such as step-by-step introspection of a script
    namespace rvt = rviz_visual_tools;
    moveit_visual_tools::MoveItVisualTools visual_tools("shoulder_link");  // ? ? ?
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

    // -------------------------------------------
    // 变量定义
    // -------------------------------------------
    std::vector<double> joint_place_saft = {79.57/57.3, -2.13/57.3, -53.65/57.3, 17.71/57.3, -93.67/57.3, -5.97/57.3};
    std::vector<double> joint_place_p1 = {79.57/57.3, -2.13/57.3, -53.65/57.3, 17.71/57.3, -93.67/57.3, -5.97/57.3};
    const double place_dx = -0.05;
    const double place_dy = -0.05*0.9;
    const double place_up = 0.05;
    const double place_down = 0.054;
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
    tf::StampedTransform transform_QRcode;
    tf::StampedTransform transfrom_T;
    tf::StampedTransform transform_tar;
    tf::StampedTransform trans_ee_link; 
    //tf::StampedTransform transform_tool0;

    geometry_msgs::Pose pose_Tag;
    double* T_Tag = new double[16];
    
    // 定时间间隔打印当前的pose和关节角度;
    // std::vector<double> joint_values0;
    // while (1)
    // {
    //   pose_rec1 = move_group.getCurrentPose();
    //   printf("pose px = %1.4f, py = %1.4f, pz = %1.4f, rx = %1.4f, ry = %1.4f, rz = %1.4f, rw = %1.4f \n", 
    //           pose_rec1.pose.position.x, pose_rec1.pose.position.y,pose_rec1.pose.position.z,
    //           pose_rec1.pose.orientation.x, pose_rec1.pose.orientation.y, pose_rec1.pose.orientation.z, pose_rec1.pose.orientation.w);
    //   joint_values0 = move_group.getCurrentJointValues();
    //   for (int jx=0; jx<6; jx++)
    //   {
    //     printf("joint axis = %d, value = %1.4f \n", jx, joint_values0[jx]*180.0/PI);

    //     jq[jx] = (double)(joint_values0[jx]);
    //   }

    //   // pose_to_mat(move_group.getCurrentPose().pose, T);
    //   // num_sols = inverse(T, q_sols, 0.0);
    //   // printf("num_sols = %d \n", num_sols);
    //   // for(int i=0; i<num_sols; i++)
    //   // {
    //   //     for (int j=0; j<6; j++)
    //   //     {
    //   //       if (q_sols[i*6+j] > Jlimit_max[j])
    //   //       {
    //   //           q_sols[i*6+j] -= 2*PI;
    //   //       }
    //   //       else if (q_sols[i*6+j] < Jlimit_min[j])
    //   //       {
    //   //           q_sols[i*6+j] += 2*PI;
    //   //       }
    //   //       else
    //   //       {
    //   //           // none
    //   //       }
    //   //     }
        
    //   //   printf("%1.6f %1.6f %1.6f %1.6f %1.6f %1.6f\n", 
    //   //     q_sols[i*6+0]*180.0/PI, q_sols[i*6+1]*180.0/PI, q_sols[i*6+2]*180.0/PI, q_sols[i*6+3]*180.0/PI, q_sols[i*6+4]*180.0/PI, q_sols[i*6+5]*180.0/PI);
    //   // }
    //   forward(jq, T);
    //   printf("forward T x = %1.4f, y = %1.4f, z = %1.4f \n", *(T+3), *(T+3+4),*(T+3+8));

    //   ros::Duration(2.0).sleep();
    // }

 
    ros::Rate rate(1.0);

    // ---------------------------------------
    // 张开电爪
    // ---------------------------------------    
    dhhand_msg.data = 90;
    dhhand_pub.publish(dhhand_msg);


    
    // --------------------------------
    // 寻找目标物体  
    // --------------------------------   
    std::vector<double> joint_group_seekbase = {79.57/57.3, -2.13/57.3, -53.65/57.3, 17.71/57.3, -93.67/57.3, -5.97/57.3};
    // visual_tools.prompt("Press 'next' : ready to seek object ! ");
    // 设定视觉搜索物体的起始位置
    printf("move to the start pose: joint_group_seekbase \n");
    move_group.setJointValueTarget(joint_group_seekbase);
    move_group.plan(my_plan);
    move_group.execute(my_plan);
    // visual_tools.prompt("next step");

    uint16_t seek_time = 0;
    uint16_t object_index = 0;
    std::vector<double> joint_group_seekstep = joint_group_seekbase;
    


    bool search_done = false;
    while(search_done == false)
    {
        // search_done = true;
        printf("-----------------------------------\n");
        printf("---stage: seek for objects --------\n");

        // 移动至本轮的初始位置,
        move_group.setJointValueTarget(joint_group_seekstep);
        move_group.plan(my_plan);
        move_group.execute(my_plan);
        // 延时等待图像识别结果
        ros::Duration(0.3).sleep();

        // 若视线范围内无目标物体,则转动基座轴(关节0)继续寻找物体
        while ((!object_flag)&&(seek_time < 6))
        {
            seek_time ++;
            joint_group_seekstep[0] += PI/20.0;

            printf("seek for QRcode, seek_time = %d\n", seek_time);
            move_group.setJointValueTarget(joint_group_seekstep);
            move_group.plan(my_plan);
            move_group.execute(my_plan);
            ros::Duration(0.3).sleep();
        }

        if (seek_time >= 10)
        {
            search_done = true;
            printf("seek_time is 10, search_done \n");
        }

        if (!object_flag)
        {
            printf("thers is none QRcode \n");
            break;
        }

        printf("---result: have found QRcode ----- \n");

        // --------------------------------
        // 摄像头近距离精准探测一次:
        // --------------------------------
        printf("-----------------------------------------\n");
        printf("---stage: move camera close to object ---\n");
        // ---------------------------------------------------------------------------
        // 读取目标物体的位姿
        pose_base_link_current = move_group.getCurrentPose("base_link").pose;
        Trans_base_link_in_World.setOrigin(tf::Vector3(pose_base_link_current.position.x, pose_base_link_current.position.y, pose_base_link_current.position.z));
        Trans_base_link_in_World.setRotation(tf::Quaternion(pose_base_link_current.orientation.x, pose_base_link_current.orientation.y, 
                                                        pose_base_link_current.orientation.z, pose_base_link_current.orientation.w));
        pose_current = move_group.getCurrentPose("kinect2_rgb_optical_frame").pose;
        Trans_Camera_in_World.setOrigin(tf::Vector3(pose_current.position.x, pose_current.position.y, pose_current.position.z));
        Trans_Camera_in_World.setRotation(tf::Quaternion(pose_current.orientation.x, pose_current.orientation.y, 
                                                        pose_current.orientation.z, pose_current.orientation.w));
        Trans_Camera_in_base_link.mult(Trans_base_link_in_World.inverse(), Trans_Camera_in_World);
        transform_QRcode.mult(Trans_Camera_in_base_link, Trans_QRcode_in_Camera);


        // 计算摄像头与物体距离,若距离较远,则近距离探测一次
        double length_obeject_camera = sqrt(Pose_QRcode_in_Camera.position.x*Pose_QRcode_in_Camera.position.x
                                        + Pose_QRcode_in_Camera.position.y*Pose_QRcode_in_Camera.position.y
                                        + Pose_QRcode_in_Camera.position.z*Pose_QRcode_in_Camera.position.z);
        printf("length_obeject_camera = %1.5f \n", length_obeject_camera);
        if (0)//(length_obeject_camera > 0.3)
        {
            double h = (length_obeject_camera > 0.2) ? 0.2 : length_obeject_camera;
            printf("move camera close to object \n");
            tf::StampedTransform trans_camera_seek;
            std::vector<double> joint_seek_near = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            // 计算摄像头的最佳位姿
            double theta = atan2(transform_QRcode.getOrigin().y(), transform_QRcode.getOrigin().x());
            double alpha = theta - PI*0.5;
            q.setRPY(PI, 0.0, alpha);
            trans_camera_seek.setOrigin(tf::Vector3(transform_QRcode.getOrigin().x(), 
                                                    transform_QRcode.getOrigin().y(), 
                                                    transform_QRcode.getOrigin().z() + h));
            trans_camera_seek.setRotation(q);

            // 根据QRcode位姿推测wrist3Link位姿  
            trans_ee_link.mult(trans_camera_seek, Trans_wrist3Link_in_Camera);

            joint_current = move_group.getCurrentJointValues();
            bool reached = tarTrans_nearJ_bestJ(trans_ee_link, joint_current, &joint_seek_near);

            if (reached == true)
            {
            printf("move to joint_seek_near, and photo again \n");
            // 设置目标位置:
            move_group.setJointValueTarget(joint_seek_near);
            move_group.plan(my_plan);
            move_group.execute(my_plan);

            // 近距离下,重新获取QRcode位姿
            ros::Duration(0.3).sleep();
            if (object_flag)
            {   
                pose_base_link_current = move_group.getCurrentPose("base_link").pose;
                Trans_base_link_in_World.setOrigin(tf::Vector3(pose_base_link_current.position.x, pose_base_link_current.position.y, pose_base_link_current.position.z));
                Trans_base_link_in_World.setRotation(tf::Quaternion(pose_base_link_current.orientation.x, pose_base_link_current.orientation.y, 
                                                                pose_base_link_current.orientation.z, pose_base_link_current.orientation.w));
                pose_current = move_group.getCurrentPose("kinect2_rgb_optical_frame").pose;
                Trans_Camera_in_World.setOrigin(tf::Vector3(pose_current.position.x, pose_current.position.y, pose_current.position.z));
                Trans_Camera_in_World.setRotation(tf::Quaternion(pose_current.orientation.x, pose_current.orientation.y, 
                                                                pose_current.orientation.z, pose_current.orientation.w));
                Trans_Camera_in_base_link.mult(Trans_base_link_in_World.inverse(), Trans_Camera_in_World);
                transform_QRcode.mult(Trans_Camera_in_base_link, Trans_QRcode_in_Camera);
            }
            }
            else
            {
            printf("cann't reach joint_seek_near \n");
            }
        }

        // --------------------------------
        // 抓 取 物 体
        // --------------------------------
        // 抓取位姿相关变量
        geometry_msgs::Pose pose_RdyToPick;
        geometry_msgs::Pose pose_Pick;
        std::vector<double> joint_RdyToPick = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        std::vector<double> joint_Pick = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        printf("-----------------------------------\n");
        printf("--- stage : pick the object -------\n");
        // visual_tools.prompt("next step");

        // 主区前先打开手抓
        printf("open the gripper fingers \n");
        dhhand_msg.data = 95;
        dhhand_pub.publish(dhhand_msg);

        // target_pose的建立流程:
        // --------------------------------
        // 步骤1:将目标物体位姿TF进行数据格式变换
        // --------------------------------
        pose_Tag.position.x = transform_QRcode.getOrigin().x();
        pose_Tag.position.y = transform_QRcode.getOrigin().y();
        pose_Tag.position.z = transform_QRcode.getOrigin().z();
        pose_Tag.orientation.x = transform_QRcode.getRotation().x();
        pose_Tag.orientation.y = transform_QRcode.getRotation().y();
        pose_Tag.orientation.z = transform_QRcode.getRotation().z();
        pose_Tag.orientation.w = transform_QRcode.getRotation().w();
        pose_to_mat(pose_Tag, T_Tag);

        // --------------------------------
        // 步骤2:判断Tag朝向,根据朝向选择抓取面与姿态,
        // --------------------------------
        // 方法:transform_tar基于transform_QRcode做变换,使得z轴朝下; 
        double rz_tag[3] = {*(T_Tag+2), *(T_Tag+2+4), *(T_Tag+2+8)};
        double rz_world[3] = {0.0, 0.0, 1.0};
        double rzz_mul;
        double rx_tag[3] = {*(T_Tag+0), *(T_Tag+0+4), *(T_Tag+0+8)};
        double rxz_mul;
        double ry_tag[3] = {*(T_Tag+1), *(T_Tag+1+4), *(T_Tag+1+8)};
        double ryz_mul;
        rxz_mul = rx_tag[0]*rz_world[0] + rx_tag[1]*rz_world[1] + rx_tag[2]*rz_world[2];
        ryz_mul = ry_tag[0]*rz_world[0] + ry_tag[1]*rz_world[1] + ry_tag[2]*rz_world[2];
        rzz_mul = rz_tag[0]*rz_world[0] + rz_tag[1]*rz_world[1] + rz_tag[2]*rz_world[2];
        printf("rxz_mul = %1.4f \n", rxz_mul);
        printf("ryz_mul = %1.4f \n", ryz_mul);
        printf("rzz_mul = %1.4f \n", rzz_mul);
        printf("the QRcode orientation is: ");
        q.setRPY(0, 0, 0);
        if (rzz_mul > 0.707)
        {
        // 朝上  // 如果Tag正面朝上,则tar_pose为tag_pose绕x轴旋转PI  
        printf("up \n");  q.setRPY(PI, 0, 0);
        }
        else if (rzz_mul < -0.707)
        {
        // 朝下,因为遮挡,该情况不会出现
        printf("down \n"); q.setRPY(0, 0, 0);
        }
        else
        {
        // z侧面;
        printf("side \n");
        // x水平
        if (std::fabs(rxz_mul) < 0.707)
        {
            // y向上
            if (ryz_mul > 0.707) { q.setRPY(PI*0.5, 0, 0);  }
            else { q.setRPY(-PI*0.5, 0, 0); }
        }
        else // y水平
        {
            // x向上
            if (rxz_mul > 0.707) { q.setRPY(0, -PI*0.5, 0); }
            else { q.setRPY(0, PI*0.5, 0); }
        }
        }

        transfrom_T.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
        transfrom_T.setRotation(q);
        
        transform_tar = transform_QRcode;
        transform_tar.operator*=(transfrom_T);

        // 判断x轴与基座的方向,选择合适抓取位姿
        geometry_msgs::Pose pose_temp;
        pose_temp.position.x = transform_tar.getOrigin().x();
        pose_temp.position.y = transform_tar.getOrigin().y();
        pose_temp.position.z = transform_tar.getOrigin().z();
        pose_temp.orientation.x = transform_tar.getRotation().x();
        pose_temp.orientation.y = transform_tar.getRotation().y();
        pose_temp.orientation.z = transform_tar.getRotation().z();
        pose_temp.orientation.w = transform_tar.getRotation().w();
        pose_to_mat(pose_temp, T_Tag);
        double rx_tar[3] = {*(T_Tag+0), *(T_Tag+0+4), *(T_Tag+0+8)};
        double ry_tar[3] = {*(T_Tag+1), *(T_Tag+1+4), *(T_Tag+1+8)};
        double r_ob[3] = {fabs(pose_temp.position.x)/sqrt(pose_temp.position.x*pose_temp.position.x + pose_temp.position.y*pose_temp.position.y),
                        fabs(pose_temp.position.y)/sqrt(pose_temp.position.x*pose_temp.position.x + pose_temp.position.y*pose_temp.position.y),
                        0.0};
        double rxob_mul = 0.0;
        double ryob_mul = 0.0;
        rxob_mul = rx_tar[0]*r_ob[0] + rx_tar[1]*r_ob[1] + rx_tar[2]*r_ob[2];
        ryob_mul = ry_tar[0]*r_ob[0] + ry_tar[1]*r_ob[1] + ry_tar[2]*r_ob[2];

        q.setRPY(0, 0, 0);
        if (rxob_mul > 0.707)
        {
        // 朝向合适
        q.setRPY(0, 0, 0);
        }
        else if (rxob_mul > -0.707)
        {
        // 需要旋转90度或者-90度
        if (ryob_mul > 0) { q.setRPY(0, 0, PI/2.0); }
        else { q.setRPY(0, 0, -PI/2.0); }
        }
        else
        {
        // 需要旋转180度
        q.setRPY(0, 0, PI);
        }

        transfrom_T.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
        transfrom_T.setRotation(q);
        transform_tar.operator*=(transfrom_T);

        // 显示监控
        Transform_target = transform_tar;
        // visual_tools.prompt("next step");

        // --------------------------------
        // 步骤3:判断当前位置是否可达,不可达则绕z转旋转PI/2
        // --------------------------------
        bool pick_reached = false;
        bool pick_try = true;
        uint16_t pick_try_times = 0;
        const double pick_up = 0.05;
        const double pick_down = 0.025;

        q.setRPY(0, 0, PI/2.0);
        transfrom_T.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
        transfrom_T.setRotation(q);
        while ((pick_try)&&(pick_try_times < 4))
        {
            if (pick_try_times > 0)
            {
                transform_tar.operator*=(transfrom_T);
            }
            pick_try_times ++;
            printf("pick_try_times is %d \n", pick_try_times);

            // 显示监控
            Transform_target = transform_tar;

            // wrist3Link的准备抓取点位-目标位姿1        
            trans_ee_link.mult(transform_tar, Trans_wrist3Link_in_Gripper);
            pose_RdyToPick.position.x = trans_ee_link.getOrigin().x();
            pose_RdyToPick.position.y = trans_ee_link.getOrigin().y();
            pose_RdyToPick.position.z = trans_ee_link.getOrigin().z() + pick_up;
            pose_RdyToPick.orientation.x = trans_ee_link.getRotation().x();
            pose_RdyToPick.orientation.y = trans_ee_link.getRotation().y();
            pose_RdyToPick.orientation.z = trans_ee_link.getRotation().z();
            pose_RdyToPick.orientation.w = trans_ee_link.getRotation().w();
            // visual_tools.prompt("next step");
            // ee_link的抓取点位-目标位姿2
            // 因为是刚性连接,直接平移也可以
            // 抓取位姿在准备抓取位姿下方0.0xm处
            pose_Pick = pose_RdyToPick;
            pose_Pick.position.z -= (pick_up + pick_down);

            joint_current = move_group.getCurrentJointValues();

            bool reached1 = tarPose_nearJ_bestJ(pose_RdyToPick, joint_current, &joint_RdyToPick);
            bool reached2 = tarPose_nearJ_bestJ(pose_Pick, joint_current, &joint_Pick);
            pick_reached = reached1 && reached2;
            
            if (pick_reached == true)
            {
                // 设置目标位置:
                printf("move to pose_RdyToPick \n");
                move_group.setJointValueTarget(joint_RdyToPick);
                move_group.plan(my_plan);
                move_group.execute(my_plan);
                // visual_tools.prompt("next step");
                printf("move to pose_Pick \n");
                move_group.setJointValueTarget(joint_Pick);
                move_group.plan(my_plan);
                move_group.execute(my_plan);
                // visual_tools.prompt("next step");
                printf("close the gripper \n");
                dhhand_msg.data = 1;
                dhhand_pub.publish(dhhand_msg);
                ros::Duration(2.0).sleep();

                printf("move to pose_RdyToPick \n");
                move_group.setJointValueTarget(joint_RdyToPick);
                move_group.plan(my_plan);
                move_group.execute(my_plan);
                
                object_index ++;

                printf("this is Number:  %d object\n", object_index);

                pick_try = false;
            }

            rate.sleep();     
        }
        
        if (pick_reached == false)
        {
            printf("cann't pick this object \n");
            break;
        }

        // --------------------------------
        // 放 置 物 体
        // --------------------------------
        printf("-----------------------------------\n");
        printf("------stage : place the object ------\n");
        // // 抓取物体后的运动:准备抓取位置 -> 搜索位置
        // printf("move to joint_group_seekstep \n");
        // move_group.setJointValueTarget(joint_group_seekstep);
        // move_group.plan(my_plan);
        // move_group.execute(my_plan);

        // ---------------------
        // 放置物体:
        // ---------------------
        // 放置物体的相关变量
        std::vector<double> joint_place_up = {6.814206, -33.4987, -146.3508, -24.8628, -86.7676, 6.0375};
        std::vector<double> joint_place_down = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double* placeTc = new double[16];

        // 首先到达放置安全过渡位
        move_group.setJointValueTarget(joint_place_saft);     
        move_group.plan(my_plan);
        move_group.execute(my_plan);

        // 第N个问题的放置位姿
        geometry_msgs::Pose place_poseN;
        place_poseN = place_pose0;
        place_poseN.position.x = place_pose0.position.x + (object_index-1)*place_dx;
        place_poseN.position.y = place_pose0.position.y + (object_index-1)*place_dy;
        
        // 生成放置位姿的关节角度
        place_poseN.position.z += place_up;
        bool reach_place_up = tarPose_nearJ_bestJ(place_poseN, joint_place_p1, &joint_place_up);
        place_poseN.position.z -= place_down;
        bool reach_place_down = tarPose_nearJ_bestJ(place_poseN, joint_place_p1, &joint_place_down);
        bool reach_place = reach_place_up && reach_place_down;

        // 若可达则放置,若不可达则放至安全过度位
        if (reach_place)
        {
            printf("place the object:  joint_place_up, joint_place_down \n");
            move_group.setJointValueTarget(joint_place_up);
            move_group.plan(my_plan);
            move_group.execute(my_plan);
            move_group.setJointValueTarget(joint_place_down);     
            move_group.plan(my_plan);
            move_group.execute(my_plan);
        }
        else
        {
            printf("place the object:  cann't reach palce pose, move to saft pose \n");
            move_group.setJointValueTarget(joint_place_saft);     
            move_group.plan(my_plan);
            move_group.execute(my_plan);
        }
        printf("open the gripper \n");
        dhhand_msg.data = 80;
        dhhand_pub.publish(dhhand_msg);

        ros::Duration(1.0).sleep();

        if (reach_place)
        {
            printf("finish place the object, back to saft pose \n");
            move_group.setJointValueTarget(joint_place_up);
            move_group.plan(my_plan);
            move_group.execute(my_plan);
            
            move_group.setJointValueTarget(joint_place_saft);
            move_group.plan(my_plan);
            move_group.execute(my_plan);
        }
    }

    // 运动完毕后,清除约束   
   move_group.clearPathConstraints();

   visual_tools.prompt("Press 'next',  all things done!!! \n");


    // // 结束线程
    threadObj.join();

    ros::shutdown();
    return 0;
}


// ======================================================================================
//                                  私 有 函 数 定 义
// ======================================================================================


/**
 * @brief 接收AprilTag消息回调函数
 * 
 * @param tag 
 */ 
void handle_tag_in_camera(const apriltag_ros::AprilTagDetectionArray::ConstPtr& tag)
{
    static tf::TransformBroadcaster brQRcode;
    static uint16_t loseCnt = 0;

    uint16_t num_of_tag = tag->detections.size();
    if (num_of_tag != 0)
    {
        // 寻找距离最近的tag索引号
        double pmin = 0.0;
        geometry_msgs::Pose pose;
        uint16_t index_min = 0;
        pose = tag->detections[0].pose.pose.pose;
        pmin = pose.position.x*pose.position.x + pose.position.y*pose.position.y + pose.position.z*pose.position.z; 
        for (uint16_t i = 1; i < num_of_tag; i++)
        {
            pose = tag->detections[i].pose.pose.pose;
            double p = pose.position.x*pose.position.x + pose.position.y*pose.position.y + pose.position.z*pose.position.z;
            if (pmin < p)
            {
                pmin = p;
                index_min = i;
            }
        }

        // 查询距离最近tag的位姿
        Pose_QRcode_in_Camera = tag->detections[index_min].pose.pose.pose;
        Trans_QRcode_in_Camera.setOrigin(tf::Vector3(Pose_QRcode_in_Camera.position.x,
                                                    Pose_QRcode_in_Camera.position.y,
                                                    Pose_QRcode_in_Camera.position.z));
        Trans_QRcode_in_Camera.setRotation(tf::Quaternion(Pose_QRcode_in_Camera.orientation.x, 
                                                        Pose_QRcode_in_Camera.orientation.y,
                                                        Pose_QRcode_in_Camera.orientation.z,
                                                        Pose_QRcode_in_Camera.orientation.w));

        object_u = tag->detections[index_min].center_point[0] - 640.0/2.0;
        object_v = tag->detections[index_min].center_point[1] - 480.0/2.0;
        object_update = true;

        object_flag = true;
        loseCnt = 0;
    }
    else
    {
        object_u = 0.0;
        object_v = 0.0;
        object_update = false;

        // 丢失检测
        loseCnt ++;
        if (loseCnt > 10)
        { 
        loseCnt = 0;
        Pose_QRcode_in_Camera.position.x = 0.0;
        Pose_QRcode_in_Camera.position.y = 0.0;
        Pose_QRcode_in_Camera.position.z = 0.0;
        Pose_QRcode_in_Camera.orientation.x = 0.0;
        Pose_QRcode_in_Camera.orientation.y = 0.0;
        Pose_QRcode_in_Camera.orientation.z = 0.0;
        Pose_QRcode_in_Camera.orientation.w = 1.0;
        Trans_QRcode_in_Camera.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
        Trans_QRcode_in_Camera.setRotation(tf::Quaternion(0.0, 0.0, 0.0, 1.0));
        object_flag = false;
        }

    }

    brQRcode.sendTransform(tf::StampedTransform(Trans_QRcode_in_Camera, ros::Time::now(), "kinect2_rgb_optical_frame", "QRcode"));
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

// input radian
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


// void forward(const double* q, double* T)
// {
//     double s1 = sin(*q), c1 = cos(*q); q++;
//     double q234 = *q, s2 = sin(*q), c2 = cos(*q); q++;
//     double s3 = sin(*q), c3 = cos(*q); q234 += *q; q++;
//     q234 += *q; q++;
//     double s5 = sin(*q), c5 = cos(*q); q++;
//     double s6 = sin(*q), c6 = cos(*q); 
//     double s234 = sin(q234), c234 = cos(q234);
    
//     *T = ((c1*c234-s1*s234)*s5)/2.0 - c5*s1 + ((c1*c234+s1*s234)*s5)/2.0; T++;
//     *T = (c6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0) - 
//           (s6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0); T++;
//     *T = (-(c6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0 - 
//           s6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0)); T++;
//     *T = ((d5*(s1*c234-c1*s234))/2.0 - (d5*(s1*c234+c1*s234))/2.0 - 
//           d4*s1 + (d6*(c1*c234-s1*s234)*s5)/2.0 + (d6*(c1*c234+s1*s234)*s5)/2.0 - 
//           a2*c1*c2 - d6*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3); T++;
//     *T = c1*c5 + ((s1*c234+c1*s234)*s5)/2.0 + ((s1*c234-c1*s234)*s5)/2.0; T++;
//     *T = (c6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0) + 
//           s6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0)); T++;
//     *T = (c6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0) - 
//           s6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0)); T++;
//     *T = ((d5*(c1*c234-s1*s234))/2.0 - (d5*(c1*c234+s1*s234))/2.0 + d4*c1 + 
//           (d6*(s1*c234+c1*s234)*s5)/2.0 + (d6*(s1*c234-c1*s234)*s5)/2.0 + d6*c1*c5 - 
//           a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3); T++;
//     *T = ((c234*c5-s234*s5)/2.0 - (c234*c5+s234*s5)/2.0); T++;
//     *T = ((s234*c6-c234*s6)/2.0 - (s234*c6+c234*s6)/2.0 - s234*c5*c6); T++;
//     *T = (s234*c5*s6 - (c234*c6+s234*s6)/2.0 - (c234*c6-s234*s6)/2.0); T++;
//     *T = (d1 + (d6*(c234*c5-s234*s5))/2.0 + a3*(s2*c3+c2*s3) + a2*s2 - 
//          (d6*(c234*c5+s234*s5))/2.0 - d5*c234); T++;
//     *T = 0.0; T++; *T = 0.0; T++; *T = 0.0; T++; *T = 1.0;
// }


int SIGN(double x) {
      return (x > 0) - (x < 0);
    }

/**
 * @brief antiSinCos
 * @param sA
 * @param cA
 * @return
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

// int inverse(const double* T, double* q_sols, double q6_des) {
//     int num_sols = 0;
//     double T02 = -*T; T++; double T00 =  *T; T++; double T01 =  *T; T++; double T03 = -*T; T++; 
//     double T12 = -*T; T++; double T10 =  *T; T++; double T11 =  *T; T++; double T13 = -*T; T++; 
//     double T22 =  *T; T++; double T20 = -*T; T++; double T21 = -*T; T++; double T23 =  *T;

//     ////////////////////////////// shoulder rotate joint (q1) //////////////////////////////
//     double q1[2];
//     {
//       double A = d6*T12 - T13;
//       double B = d6*T02 - T03;
//       double R = A*A + B*B;
//       if(fabs(A) < ZERO_THRESH) {
//         double div;
//         if(fabs(fabs(d4) - fabs(B)) < ZERO_THRESH)
//           div = -SIGN(d4)*SIGN(B);
//         else
//           div = -d4/B;
//         double arcsin = asin(div);
//         if(fabs(arcsin) < ZERO_THRESH)
//           arcsin = 0.0;
//         if(arcsin < 0.0)
//           q1[0] = arcsin + 2.0*PI;
//         else
//           q1[0] = arcsin;
//         q1[1] = PI - arcsin;
//       }
//       else if(fabs(B) < ZERO_THRESH) {
//         double div;
//         if(fabs(fabs(d4) - fabs(A)) < ZERO_THRESH)
//           div = SIGN(d4)*SIGN(A);
          
//         else
//           div = d4/A;
//         double arccos = acos(div);
//         q1[0] = arccos;
//         q1[1] = 2.0*PI - arccos;
//       }
//       else if(d4*d4 > R) {
//         return num_sols;
//       }
//       else {
//         double arccos = acos(d4 / sqrt(R)) ;
//         double arctan = atan2(-B, A);
//         double pos = arccos + arctan;
//         double neg = -arccos + arctan;
//         if(fabs(pos) < ZERO_THRESH)
//           pos = 0.0;
//         if(fabs(neg) < ZERO_THRESH)
//           neg = 0.0;
//         if(pos >= 0.0)
//           q1[0] = pos;
//         else
//           q1[0] = 2.0*PI + pos;
//         if(neg >= 0.0)
//           q1[1] = neg; 
//         else
//           q1[1] = 2.0*PI + neg;
//       }
//     }
//     ////////////////////////////////////////////////////////////////////////////////

//     ////////////////////////////// wrist 2 joint (q5) //////////////////////////////
//     double q5[2][2];
//     {
//       for(int i=0;i<2;i++) {
//         double numer = (T03*sin(q1[i]) - T13*cos(q1[i])-d4);
//         double div;
//         if(fabs(fabs(numer) - fabs(d6)) < ZERO_THRESH)
//           div = SIGN(numer) * SIGN(d6);
//         else
//           div = numer / d6;
//         double arccos = acos(div);
//         q5[i][0] = arccos;
//         q5[i][1] = 2.0*PI - arccos;
//       }
//     }
//     ////////////////////////////////////////////////////////////////////////////////

//     {
//       for(int i=0;i<2;i++) {
//         for(int j=0;j<2;j++) {
//           double c1 = cos(q1[i]), s1 = sin(q1[i]);
//           double c5 = cos(q5[i][j]), s5 = sin(q5[i][j]);
//           double q6;
//           ////////////////////////////// wrist 3 joint (q6) //////////////////////////////
//           if(fabs(s5) < ZERO_THRESH)
//             q6 = q6_des;
//           else {
//             q6 = atan2(SIGN(s5)*-(T01*s1 - T11*c1), 
//                        SIGN(s5)*(T00*s1 - T10*c1));
//             if(fabs(q6) < ZERO_THRESH)
//               q6 = 0.0;
//             if(q6 < 0.0)
//               q6 += 2.0*PI;
//           }
//           ////////////////////////////////////////////////////////////////////////////////

//           double q2[2], q3[2], q4[2];
//           ///////////////////////////// RRR joints (q2,q3,q4) ////////////////////////////
//           double c6 = cos(q6), s6 = sin(q6);
//           double x04x = -s5*(T02*c1 + T12*s1) - c5*(s6*(T01*c1 + T11*s1) - c6*(T00*c1 + T10*s1));
//           double x04y = c5*(T20*c6 - T21*s6) - T22*s5;
//           double p13x = d5*(s6*(T00*c1 + T10*s1) + c6*(T01*c1 + T11*s1)) - d6*(T02*c1 + T12*s1) + 
//                         T03*c1 + T13*s1;
//           double p13y = T23 - d1 - d6*T22 + d5*(T21*c6 + T20*s6);

//           double c3 = (p13x*p13x + p13y*p13y - a2*a2 - a3*a3) / (2.0*a2*a3);
//           if(fabs(fabs(c3) - 1.0) < ZERO_THRESH)
//             c3 = SIGN(c3);
//           else if(fabs(c3) > 1.0) {
//             // TODO NO SOLUTION
//             continue;
//           }
//           double arccos = acos(c3);
//           q3[0] = arccos;
//           q3[1] = 2.0*PI - arccos;
//           double denom = a2*a2 + a3*a3 + 2*a2*a3*c3;
//           double s3 = sin(arccos);
//           double A = (a2 + a3*c3), B = a3*s3;
//           q2[0] = atan2((A*p13y - B*p13x) / denom, (A*p13x + B*p13y) / denom);
//           q2[1] = atan2((A*p13y + B*p13x) / denom, (A*p13x - B*p13y) / denom);
//           double c23_0 = cos(q2[0]+q3[0]);
//           double s23_0 = sin(q2[0]+q3[0]);
//           double c23_1 = cos(q2[1]+q3[1]);
//           double s23_1 = sin(q2[1]+q3[1]);
//           q4[0] = atan2(c23_0*x04y - s23_0*x04x, x04x*c23_0 + x04y*s23_0);
//           q4[1] = atan2(c23_1*x04y - s23_1*x04x, x04x*c23_1 + x04y*s23_1);
//           ////////////////////////////////////////////////////////////////////////////////
//           for(int k=0;k<2;k++) {
//             if(fabs(q2[k]) < ZERO_THRESH)
//               q2[k] = 0.0;
//             else if(q2[k] < 0.0) q2[k] += 2.0*PI;
//             if(fabs(q4[k]) < ZERO_THRESH)
//               q4[k] = 0.0;
//             else if(q4[k] < 0.0) q4[k] += 2.0*PI;
//             q_sols[num_sols*6+0] = q1[i];    q_sols[num_sols*6+1] = q2[k]; 
//             q_sols[num_sols*6+2] = q3[k];    q_sols[num_sols*6+3] = q4[k]; 
//             q_sols[num_sols*6+4] = q5[i][j]; q_sols[num_sols*6+5] = q6; 
//             num_sols++;
//           }

//         }
//       }
//     }
//     return num_sols;
// }

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

void rosinfo_joint(std::vector<std::string> joint_names, std::vector<double> j_values)
{
   ROS_INFO("--- this is a new ros info message ------");
  for (std::size_t j = 0; j < joint_names.size(); j++)
  {
      ROS_INFO("Joint %s: %f", joint_names[j].c_str(), j_values[j]);
  }
}


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