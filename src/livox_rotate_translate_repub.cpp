#define _USE_MATH_DEFINES
#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Header.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/transforms.h>
#include <pcl/common/transforms.h>
#include <sensor_msgs/PointCloud2.h>
#include <cmath>
#include "Eigen/Dense"
#include "yaml-cpp/yaml.h"

using namespace Eigen;
using namespace std;

ros::Publisher livox_repub;
std::vector<sensor_msgs::PointCloud2ConstPtr> livox_data;
Eigen::Matrix3d lidar_rotation_m;
std::string DB_CONF;
typedef Matrix<double, 3, 1> MytranslationType;
MytranslationType lidar_translation_v;

float alpha_angle, theta_angle, gamma_angle, x, y, z;


Eigen::Matrix3d rotation() {
    Eigen::Matrix3d rz, ry, rx;
    rz << cos(alpha_angle), -sin(alpha_angle), 0,
          sin(alpha_angle), cos(alpha_angle), 0,
          0 , 0, 1;
    ry << cos(theta_angle), 0 ,sin(theta_angle), 
          0, 1, 0,
          -sin(theta_angle), 0, cos(theta_angle);
    rx << 1, 0, 0,
          0, cos(gamma_angle), -sin(gamma_angle),
          0, sin(gamma_angle), cos(gamma_angle);
    return rz * ry * rx;
}

MytranslationType translation() {
    MytranslationType translation_vector;
    translation_vector << x, y, z;
    return translation_vector;
}

void getParameters(){
    ros::param::get("/DB_CONF", DB_CONF);
    YAML::Node config = YAML::LoadFile(DB_CONF);
    alpha_angle = config["tower_confg"]["alpha"].as<float>() * M_PI / 180;
    theta_angle = config["tower_confg"]["theta"].as<float>() * M_PI / 180;
    gamma_angle = config["tower_confg"]["gamma"].as<float>() * M_PI / 180;
    x = config["tower_confg"]["x"].as<float>();
    y = config["tower_confg"]["y"].as<float>();
    z = config["tower_confg"]["z"].as<float>();
    lidar_rotation_m = rotation();
    lidar_translation_v = translation();
    std::cout << lidar_translation_v << ' ' << lidar_rotation_m << endl;
}



void livox_callback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg ){
    
    MytranslationType points_list;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*lidar_msg, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);
    for(int i=0; i<temp_cloud->points.size(); i++){
        float x = temp_cloud->points[i].x;
        float y = temp_cloud->points[i].y;
        float z = temp_cloud->points[i].z;

        points_list << x, y, z;
        Eigen::MatrixXd result(3, 1);
        result = lidar_rotation_m * points_list;

        temp_cloud->points[i].x = result(0, 0) + lidar_translation_v(0, 0);
        temp_cloud->points[i].y = result(1, 0) + lidar_translation_v(1, 0);
        temp_cloud->points[i].z = result(2, 0) + lidar_translation_v(2, 0);

    }
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*temp_cloud, output);
    output.header = lidar_msg->header;
    livox_repub.publish(output); // publish the cloud point to rviz
    ROS_INFO("livox_repub !");
}


int main(int argc, char** argv) {
  ros::init(argc, argv, "livox_repub");
  ros::NodeHandle n;
  getParameters();
  
  ROS_INFO("start livox_repub");

  ros::Subscriber sub = n.subscribe("/livox/lidar/time_sync", 1, livox_callback);
  livox_repub = n.advertise<sensor_msgs::PointCloud2>("/livox/lidar/time_sync/repub", 1);
  ros::spin();
  return 0;
}
