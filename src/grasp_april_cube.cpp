 
#include <ros/ros.h>
#include <ros/console.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <XmlRpcException.h>
#include <memory>
#include <string> 
#include <vector>
#include <map> 
#include <tf/transform_broadcaster.h> 

  
#include "apriltag_ros/AprilTagDetection.h"
#include "apriltag_ros/AprilTagDetectionArray.h" 

using namespace Eigen; 

ros::Subscriber apriltags_subscriber_; // 接收apriltag消息



/**
 * @brief 接收AprilTag消息回调函数
 * 
 * @param tag_detections 
 */
void apriltagsCallback(const apriltag_ros::AprilTagDetectionArray::ConstPtr& tag_detections)
{ 
     
    if (tag_detections->detections.size() == 0)
    {  
        return;
    } 
    else // 检测到四个二维码
    {
      Eigen::Quaterniond temp_q;
      ROS_DEBUG_STREAM("detected tags nums is " << tag_detections->detections.size());
      for(int i = 0; i < tag_detections->detections.size(); i++)
      { 
          if(tag_detections->detections[i].id[0] == 11)
          {
            // ROS_DEBUG_STREAM("detected position " << tag_detections->detections[i].center_point[0] << "," << tag_detections->detections[i].center_point[1]);
            geometry_msgs::PoseWithCovariance tag_pose = tag_detections->detections[i].pose.pose;  
            temp_q.x() = tag_pose.pose.orientation.x;
            temp_q.y() = tag_pose.pose.orientation.y;
            temp_q.z() = tag_pose.pose.orientation.z;
            temp_q.w() = tag_pose.pose.orientation.w;
            Eigen::Vector3d temp_trans(tag_pose.pose.position.x, tag_pose.pose.position.y, tag_pose.pose.position.z); 
            m_trans[tag_detections->detections[i].id[0]] = temp_trans; 
            // ROS_DEBUG_STREAM("Camera detect tag is " << tag_detections->detections[i].id[0] << "  " << tag_pose.pose.position.x << " "  << tag_pose.pose.position.y << " "  << tag_pose.pose.position.z);
            // v_trans.push_back(temp_trans);
          } 
      } 
    //   find_cup_in_four_apriltag(src_, m_src_points, m_trans, temp_q, cup_height_ , camera_K_);
    }  
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "find_cup_ros");
  
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

   apriltags_subscriber_ =
      nh.subscribe("/tag_detections", 100,
                          &apriltagsCallback );

}