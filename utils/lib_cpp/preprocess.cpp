// #include <iostream>
// #include <cmath>
// #include <vector>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>
// #include <ros/ros.h>
// #include <std_msgs/Header.h>
// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/conversions.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// namespace py = pybind11;
// using namespace std;

// float DX = 0.2;
// float DY = 0.2;
// float DZ = 0.2;
// float X_MIN = 0;
// float X_MAX = 100;
// float Y_MIN = -8;
// float Y_MAX = 8;
// float Z_MIN = -3.0;
// float Z_MAX = 3.0;

// float overlap = 11.2;
// int HEIGHT = round((X_MAX - X_MIN + 2 * overlap) / DX);
// int WIDTH = round((Y_MAX - Y_MIN) / DY);
// int CHANNELS = round((Z_MAX - Z_MIN) / DZ);
// int LENGTH = HEIGHT*WIDTH*CHANNELS;

// py::array_t<float> my_function(py::array_t<float> input)
// {
//     py::array_t<float> result = py::array_t<float>(LENGTH);
//     result.resize({1, LENGTH});
    
//     py::buffer_info buf = input.request();
//     py::buffer_info buf_result = result.request();

//     float* my_ptr = (float*)buf.ptr;
//     float* ptr_result = (float*)buf_result.ptr;

//     int channel;
//     int pixel_x;
//     int pixel_y;
//     int index;

//     for(int t=0; t<LENGTH; t++)
//     {
//         ptr_result[t] = 0.0;
//     }

//     int slice = buf.shape[1] / 4;
//     for(int i =0; i<slice; i++)
//     {
//         float X = my_ptr[i*4+0];
//         float Y = my_ptr[i*4+1];
//         float Z = my_ptr[i*4+2];


//         if((Y > Y_MIN) && (Y < Y_MAX) &&
//            (X > X_MIN) && (X < X_MAX) &&
//            (Z > Z_MIN) && (Z < Z_MAX))
//         {
//             channel = static_cast<int>((Z_MAX - Z)/DZ);
//             if( fabs(X)<3 && fabs(Y)< 3)
//             {
//                 continue;  
//             }
//             if( X > -overlap)
//             {
//                 pixel_x = static_cast<int>((X - X_MIN + 2*overlap)/DX);
//                 pixel_y = static_cast<int>((-Y + Y_MAX)/DY);
//                 index = pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel;
//                 ptr_result[index] = 1.0;
                
//             }
//             if(X < overlap)
//             {
//                 pixel_x = static_cast<int>((-X + overlap)/DX);
//                 pixel_y = static_cast<int>((Y + Y_MAX)/DY);
//                 index = pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel;
//                 ptr_result[index] = 1.0;
//             }
//         }
//     }
//     return result;

// }


// std::vector<double> read_pointcloud2(const sensor_msgs::PointCloud2ConstPtr& lidar_msg){
    
//     vector<double> lidar_datas; 

//     pcl::PCLPointCloud2 pcl_pc2;
//     pcl_conversions::toPCL(*lidar_msg, pcl_pc2);
//     pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);
//     int length = temp_cloud->points.size();

//     // py::array_t<float> result = py::array_t<float>(length * 3);
//     // result.resize({1, LENGTH});
//     // py::buffer_info buf_result = result.request();

//     for (uint64_t i = 0; i < length; ++i){
//         float x = temp_cloud->points[i].x;
//         float y = temp_cloud->points[i].y;
//         float z = temp_cloud->points[i].z;
//         if((abs(x) < 1.5) and (abs(y)<1.5)){
//             continue;
//         }
//         lidar_datas.push_back(x);
//         lidar_datas.push_back(y);
//         lidar_datas.push_back(z);
//     }
//     // py::list<float> result = py::cast(lidar_datas)
//     return lidar_datas;
// }
// PYBIND11_MODULE(preprocess, m) 
// {
//     m.def("my_function", &my_function);
//     m.def("read_pointcloud2", &read_pointcloud2);
// }

// int32_t main() {
//   return 0;
// }


#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include "Eigen/Dense"

namespace py = pybind11;
using namespace std;
using namespace Eigen;




py::array_t<float> data2voxel(py::array_t<float> input, float DX, float DY, float DZ, 
                                                        float X_MIN, float X_MAX, 
                                                        float Y_MIN, float Y_MAX, 
                                                        float Z_MIN, float Z_MAX,
                                                        float overlap, int HEIGHT, int WIDTH, int CHANNELS, int LENGTH)
{

    // input 1 * Length
    py::array_t<float> result = py::array_t<float>(LENGTH);
    result.resize({1, LENGTH});
    
    py::buffer_info buf = input.request();
    py::buffer_info buf_result = result.request();

    float* my_ptr = (float*)buf.ptr;
    float* ptr_result = (float*)buf_result.ptr;

    int channel;
    int pixel_x;
    int pixel_y;
    int index;

    for(int t=0; t<LENGTH; t++)
    {
        ptr_result[t] = 0.0;
    }

    int slice = buf.shape[1] / 3;
    for(int i =0; i<slice; i++)
    {
        float X = my_ptr[i*3+0];
        float Y = my_ptr[i*3+1];
        float Z = my_ptr[i*3+2];


        if((Y > Y_MIN) && (Y < Y_MAX) &&
           (X > X_MIN) && (X < X_MAX) &&
           (Z > Z_MIN) && (Z < Z_MAX))
        {
            channel = static_cast<int>((Z_MAX - Z)/DZ);
            if( fabs(X)<3 && fabs(Y)< 3)
            {
                continue;  
            }
            if( X > -overlap)
            {
                pixel_x = static_cast<int>((X - X_MIN + 2*overlap)/DX);
                pixel_y = static_cast<int>((-Y + Y_MAX)/DY);
                index = pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel;
                ptr_result[index] = 1.0;
                
            }
            if(X < overlap)
            {
                pixel_x = static_cast<int>((-X + overlap)/DX);
                pixel_y = static_cast<int>((Y + Y_MAX)/DY);
                index = pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel;
                ptr_result[index] = 1.0;
            }
        }
    }
    result.resize({HEIGHT, WIDTH, CHANNELS});
    return result;

}

// typedef Matrix<double, 3, 1> MytranslationType;
// Eigen::MatrixXd rotate_translate_pcd(Eigen::RowVectorXd points_list, Eigen::Matrix3d lidar_rotation_m, MytranslationType lidar_translation_v)
// {
//     int length = points_list.size() / 3;
//     cout << length << endl;
//     cout << points_list.size() << endl;
 
//     Eigen::MatrixXd point_cloud;
//     point_cloud = Map<MatrixXd>(points_list.data(), 3, length);
//     Eigen::MatrixXd result(3, length);
//     result = lidar_rotation_m * point_cloud;

//     py::array_t<float> result_arr = py::array_t<float>(points_list.size());
//     py::buffer_info buf = result_arr.request();
//     float* my_ptr = (float*)buf.ptr;

//     for(iont i=0; i< length, i++){
//         my_ptr[i*3 + 0] = result(0, i) + lidar_translation_v(0, 0);
//         my_ptr[i*3 + 0] = result(1, i) + lidar_translation_v(1, 0);
//         my_ptr[i*3 + 0] = result(2, i) + lidar_translation_v(0, 0);
//     }


//     return result;
// }
typedef Matrix<double, 3, 1> MytranslationType;
py::array_t<float> rotate_translate_pcd(Eigen::RowVectorXd points_list, Eigen::Matrix3d lidar_rotation_m, MytranslationType lidar_translation_v)
{
    // points_list: python list with 1 row
    // lidar_rotation_m: python numpy array with 3 * 3
    // lidar_translation_v: python numpy array with 3 * 1
    int length = points_list.size() / 3;
 
    Eigen::MatrixXd point_cloud;
    point_cloud = Map<MatrixXd>(points_list.data(), 3, length);
    Eigen::MatrixXd result(3, length);
    result = lidar_rotation_m * point_cloud;

    py::array_t<float> result_arr = py::array_t<float>(points_list.size());
    py::buffer_info buf = result_arr.request();
    float* my_ptr = (float*)buf.ptr;

    for(int i=0; i< length; i++){
        my_ptr[i*3 + 0] = result(0, i) + lidar_translation_v(0, 0);
        my_ptr[i*3 + 1] = result(1, i) + lidar_translation_v(1, 0);
        my_ptr[i*3 + 2] = result(2, i) + lidar_translation_v(2, 0);
    }
    result_arr.resize({length, 3});
    return result_arr;
}

PYBIND11_MODULE(preprocess, m) 
{
    m.def("data2voxel", &data2voxel);
    m.def("rotate_translate_pcd", &rotate_translate_pcd);

}

int32_t main() {
  return 0;
}
