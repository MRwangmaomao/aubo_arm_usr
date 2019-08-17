/***********************************************************************
Copyright 2019 Wuhan PS-Micro Technology Co., Itd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
***********************************************************************/

#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "moveit_fk_demo");
    ros::AsyncSpinner spinner(1);
    spinner.start();

    moveit::planning_interface::MoveGroupInterface arm("manipulator_i5");

    arm.setGoalJointTolerance(0.001);

    arm.setMaxAccelerationScalingFactor(0.2);
    arm.setMaxVelocityScalingFactor(0.2);

    // 控制机械臂先回到初始化位置
    double homePose[6] = {2.647702082592703, -0.09461623539508346, -0.07743478571253157, -0.020532026387635086, -2.504847350952217, -0.1068087411027895};
    std::vector<double> joint_group_positions(6);
    joint_group_positions[0] = homePose[0];
    joint_group_positions[1] = homePose[1];
    joint_group_positions[2] = homePose[2];
    joint_group_positions[3] = homePose[3];
    joint_group_positions[4] = homePose[4];
    joint_group_positions[5] = homePose[5];

    arm.setJointValueTarget(joint_group_positions); 
    arm.move();
    sleep(1);

    double targetPose[6] = {0.391410, -0.676384, -0.376217, 0.0, 1.052834, 0.454125}; 
    joint_group_positions[0] = targetPose[0];
    joint_group_positions[1] = targetPose[1];
    joint_group_positions[2] = targetPose[2];
    joint_group_positions[3] = targetPose[3];
    joint_group_positions[4] = targetPose[4];
    joint_group_positions[5] = targetPose[5];

    arm.setJointValueTarget(joint_group_positions);
    arm.move();
    sleep(1);

    // 控制机械臂先回到初始化位置
    joint_group_positions[0] = homePose[0];
    joint_group_positions[1] = homePose[1];
    joint_group_positions[2] = homePose[2];
    joint_group_positions[3] = homePose[3];
    joint_group_positions[4] = homePose[4];
    joint_group_positions[5] = homePose[5];
    arm.setJointValueTarget(joint_group_positions);
    arm.move();
    sleep(1);

    ros::shutdown(); 

    return 0;
}
