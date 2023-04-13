#include <iostream>
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <ros/ros.h>
#include <std_msgs/Header.h>
// #include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/transforms.h>
// #include <pcl_ros/transforms.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
namespace py = pybind11;
using namespace std;


std::vector<double> my_function(sensor_msgs::PointCloud2ConstPtr& lidar_msg){
    
    vector<double> lidar_datas; 

    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*lidar_msg, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);
    int length = temp_cloud->points.size();

    // py::array_t<float> result = py::array_t<float>(length * 3);
    // result.resize({1, LENGTH});
    // py::buffer_info buf_result = result.request();

    for (uint64_t i = 0; i < length; ++i){
        float x = temp_cloud->points[i].x;
        float y = temp_cloud->points[i].y;
        float z = temp_cloud->points[i].z;
        if((abs(x) < 1.5) and (abs(y)<1.5)){
            continue;
        }
        lidar_datas.push_back(x);
        lidar_datas.push_back(y);
        lidar_datas.push_back(z);
    }
    // py::list<float> result = py::cast(lidar_datas)
    return lidar_datas;
}
PYBIND11_MODULE(read_pointcloud2, m) 
{

    m.def("my_function", &my_function);
}

int32_t main() {
  return 0;
}
