
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
#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include<opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "Eigen/Dense"
namespace py = pybind11;
using namespace std;
using namespace Eigen;
using namespace cv;
typedef Matrix<float, 3, 1> MytranslationType;
typedef Matrix<float, 3, 4> MyProjectionType;
typedef Matrix<float, 3, 8> MyMatrix38D;
typedef Matrix<float, 8, 3> MyMatrix83D;
typedef Matrix<float, 8, 2> MyMatrix82D;
typedef Matrix<float, Dynamic, 7> MyMatrixd7D;
cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input) {
    if (input.ndim() != 3)
        throw std::runtime_error("3-channel image must be 3 dims ");
    py::buffer_info buf = input.request();
    cv::Mat mat(static_cast<int>(buf.shape[0]), static_cast<int>(buf.shape[1]), CV_8UC3, (unsigned char*)buf.ptr);
    return mat;
}

py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(cv::Mat& input) {
    py::array_t<unsigned char> dst = py::array_t<unsigned char>({ input.rows,input.cols,3}, input.data);
    return dst;
}

MyMatrix83D inverse_roate_pcd(MyMatrix83D corners_3d, Eigen::Matrix3f camera_rotation_m, MytranslationType camera_translation_v){
    Eigen::MatrixXf result_corners_3d(8, 3);
    result_corners_3d << corners_3d(0, 0)-camera_translation_v(0, 0), corners_3d(0, 1)-camera_translation_v(1, 0), corners_3d(0, 2)-camera_translation_v(2, 0),
                         corners_3d(1, 0)-camera_translation_v(0, 0), corners_3d(1, 1)-camera_translation_v(1, 0), corners_3d(1, 2)-camera_translation_v(2, 0),
                         corners_3d(2, 0)-camera_translation_v(0, 0), corners_3d(2, 1)-camera_translation_v(1, 0), corners_3d(2, 2)-camera_translation_v(2, 0),
                         corners_3d(3, 0)-camera_translation_v(0, 0), corners_3d(3, 1)-camera_translation_v(1, 0), corners_3d(3, 2)-camera_translation_v(2, 0),
                         corners_3d(4, 0)-camera_translation_v(0, 0), corners_3d(4, 1)-camera_translation_v(1, 0), corners_3d(4, 2)-camera_translation_v(2, 0),
                         corners_3d(5, 0)-camera_translation_v(0, 0), corners_3d(5, 1)-camera_translation_v(1, 0), corners_3d(5, 2)-camera_translation_v(2, 0),
                         corners_3d(6, 0)-camera_translation_v(0, 0), corners_3d(6, 1)-camera_translation_v(1, 0), corners_3d(6, 2)-camera_translation_v(2, 0),
                         corners_3d(7, 0)-camera_translation_v(0, 0), corners_3d(7, 1)-camera_translation_v(1, 0), corners_3d(7, 2)-camera_translation_v(2, 0);
    Eigen::MatrixXf new_result_corners_3d = camera_rotation_m * result_corners_3d.transpose();
    MyMatrix83D new_result_corners_3d_ = new_result_corners_3d.transpose();
    return new_result_corners_3d_;
    // Eigen::MatrixXf result_corners_3d(3, 8);
    // result_corners_3d << corners_3d(0, 0)-camera_translation_v(0, 0), corners_3d(0, 1)-camera_translation_v(0, 0), corners_3d(0, 2)-camera_translation_v(0, 0), corners_3d(0, 3)-camera_translation_v(0, 0), 
    //                      corners_3d(0, 4)-camera_translation_v(0, 0), corners_3d(0, 5)-camera_translation_v(0, 0), corners_3d(0, 6)-camera_translation_v(0, 0), corners_3d(0, 7)-camera_translation_v(0, 0),
    //                      corners_3d(1, 0)-camera_translation_v(1, 0), corners_3d(1, 1)-camera_translation_v(1, 0), corners_3d(1, 2)-camera_translation_v(1, 0), corners_3d(1, 3)-camera_translation_v(1, 0), 
    //                      corners_3d(1, 4)-camera_translation_v(1, 0), corners_3d(1, 5)-camera_translation_v(1, 0), corners_3d(1, 6)-camera_translation_v(1, 0), corners_3d(1, 7)-camera_translation_v(1, 0),
    //                      corners_3d(2, 0)-camera_translation_v(2, 0), corners_3d(2, 1)-camera_translation_v(2, 0), corners_3d(2, 2)-camera_translation_v(2, 0), corners_3d(2, 3)-camera_translation_v(2, 0), 
    //                      corners_3d(2, 4)-camera_translation_v(2, 0), corners_3d(2, 5)-camera_translation_v(2, 0), corners_3d(2, 6)-camera_translation_v(2, 0), corners_3d(2, 7)-camera_translation_v(2, 0);
    // Eigen::MatrixXf new_result_corners_3d(8, 3);
    // new_result_corners_3d = (camera_rotation_m * result_corners_3d).transpose();
    // return new_result_corners_3d;
}

MyMatrix83D compute_box_3d(float dim_h, float dim_w, float dim_l, float location_x, float location_y, float location_z, float ry){
    Eigen::Matrix3f R;
    float c = cos(ry);
    float s = sin(ry);
    R << c, 0, s,
         0, 1, 0,
         -s, 0, c;
    Eigen::MatrixXf xyz_corners(3, 8);
    xyz_corners << dim_l/2, dim_l/2, -dim_l/2, -dim_l/2, dim_l/2, dim_l/2, -dim_l/2, -dim_l/2,
                   0,       0,       0,        0,        -dim_h,  -dim_h,  -dim_h,   -dim_h,
                   dim_w/2, -dim_w/2, -dim_w/2, dim_w/2, dim_w/2, -dim_w/2, -dim_w/2, dim_w/2;
    Eigen::MatrixXf corners_3d(3,8);
    
    corners_3d = R * xyz_corners;
    MyMatrix83D result_corners_3d;
    result_corners_3d << corners_3d(0, 0) + location_x, corners_3d(1, 0) + location_y, corners_3d(2, 0) + location_z,
                         corners_3d(0, 1) + location_x, corners_3d(1, 1) + location_y, corners_3d(2, 1) + location_z,
                         corners_3d(0, 2) + location_x, corners_3d(1, 2) + location_y, corners_3d(2, 2) + location_z,
                         corners_3d(0, 3) + location_x, corners_3d(1, 3) + location_y, corners_3d(2, 3) + location_z,
                         corners_3d(0, 4) + location_x, corners_3d(1, 4) + location_y, corners_3d(2, 4) + location_z,
                         corners_3d(0, 5) + location_x, corners_3d(1, 5) + location_y, corners_3d(2, 5) + location_z,
                         corners_3d(0, 6) + location_x, corners_3d(1, 6) + location_y, corners_3d(2, 6) + location_z,
                         corners_3d(0, 7) + location_x, corners_3d(1, 7) + location_y, corners_3d(2, 7) + location_z;

    return result_corners_3d;
}

MyMatrix82D project_to_image(MyMatrix83D corners_3d_camera, MyProjectionType P){
    Eigen::MatrixXf pts_3d_homo(8,4);
    pts_3d_homo << corners_3d_camera(0, 0), corners_3d_camera(0, 1), corners_3d_camera(0, 2), 1,
                   corners_3d_camera(1, 0), corners_3d_camera(1, 1), corners_3d_camera(1, 2), 1,
                   corners_3d_camera(2, 0), corners_3d_camera(2, 1), corners_3d_camera(2, 2), 1,
                   corners_3d_camera(3, 0), corners_3d_camera(3, 1), corners_3d_camera(3, 2), 1,
                   corners_3d_camera(4, 0), corners_3d_camera(4, 1), corners_3d_camera(4, 2), 1,
                   corners_3d_camera(5, 0), corners_3d_camera(5, 1), corners_3d_camera(5, 2), 1,
                   corners_3d_camera(6, 0), corners_3d_camera(6, 1), corners_3d_camera(6, 2), 1,
                   corners_3d_camera(7, 0), corners_3d_camera(7, 1), corners_3d_camera(7, 2), 1;
    Eigen::MatrixXf pts_2d(3,8);
    pts_2d = P * pts_3d_homo.transpose();

    MyMatrix82D pts_2d_str;
    pts_2d_str <<  pts_2d(0, 0)/pts_2d(2, 0), pts_2d(1, 0)/pts_2d(2, 0),
                   pts_2d(0, 1)/pts_2d(2, 1), pts_2d(1, 1)/pts_2d(2, 1),
                   pts_2d(0, 2)/pts_2d(2, 2), pts_2d(1, 2)/pts_2d(2, 2),
                   pts_2d(0, 3)/pts_2d(2, 3), pts_2d(1, 3)/pts_2d(2, 3),
                   pts_2d(0, 4)/pts_2d(2, 4), pts_2d(1, 4)/pts_2d(2, 4),
                   pts_2d(0, 5)/pts_2d(2, 5), pts_2d(1, 5)/pts_2d(2, 5),
                   pts_2d(0, 6)/pts_2d(2, 6), pts_2d(1, 6)/pts_2d(2, 6),
                   pts_2d(0, 7)/pts_2d(2, 7), pts_2d(1, 7)/pts_2d(2, 7);
    return pts_2d_str;

}


cv::Mat draw_box_3d(cv::Mat img, Eigen::MatrixXi corners_2d){   
    Eigen::Matrix4i face_idx;
    face_idx << 0,1,5,4,
                1,2,6,5,
                2,3,7,6,
                3,0,4,7;
    for(int i=3; i>=0; i--)
    {
        for(int j=0; j<4; j++)
        {
            cv::line(img, Point(corners_2d(face_idx(i,j),0),       corners_2d(face_idx(i,j),1)),
                             Point(corners_2d(face_idx(i,(j+1)%4),0), corners_2d(face_idx(i,(j+1)%4),1)),
                             Scalar(0,0,255), 2, cv::LINE_AA);
        }
        if(i==0){
            cv::line(img, Point(corners_2d(face_idx(i,0),0), corners_2d(face_idx(i,0),1)),
                  Point(corners_2d(face_idx(i,2),0), corners_2d(face_idx(i,2),1)),
                  Scalar(0,0,255), 1, cv::LINE_AA);
            cv::line(img, Point(corners_2d(face_idx(i,1),0), corners_2d(face_idx(i,1),1)),
                  Point(corners_2d(face_idx(i,3),0), corners_2d(face_idx(i,3),1)),
                  Scalar(0,0,255), 1, cv::LINE_AA);
        }
    }
    
    int i=0;
    
    return img;

}

py::array_t<unsigned char> show_rgb_image_with_boxes(MyMatrixd7D kitti_dets, py::array_t<unsigned char> my_img, Eigen::Matrix3f camera_rotation_m, MytranslationType camera_translation_v, MyProjectionType P) {
    // convert image
    cv::Mat img = numpy_uint8_3c_to_cv_mat(my_img);
    // cv::Mat img;
    // eigen2cv(my_img, img);


    for(int i=0; i< kitti_dets.size()/7; i++){

        float location_x = kitti_dets(i,0);
        float location_y = kitti_dets(i,1);
        float location_z = kitti_dets(i,2);
        float dim_h = kitti_dets(i,3);
        float dim_w = kitti_dets(i,4);
        float dim_l = kitti_dets(i,5);
        float ry = kitti_dets(i,6);


        MyMatrix83D corners_3d = compute_box_3d(dim_h, dim_w, dim_l, location_x, location_y, location_z, ry);     
        MyMatrix83D corners_3d_camera = inverse_roate_pcd(corners_3d, camera_rotation_m, camera_translation_v); 
        MyMatrix82D corners_2d =  project_to_image(corners_3d_camera, P);
        Eigen::MatrixXi corners_2d_(8, 2);
        corners_2d_ = corners_2d.cast<int>();
        // if(i==0){
        //     cout << P << endl;
        //     cout << corners_3d_camera << endl;
        //     cout << corners_2d << endl;
        // }  
        img = draw_box_3d(img, corners_2d_);
    }
    return cv_mat_uint8_3c_to_numpy(img);
 
}

MyMatrix83D get_3d_box(float dim_h, float dim_w, float dim_l, float location_x, float location_y, float location_z, float ry){
    Eigen::Matrix3f R;
    float c = cos(ry);
    float s = sin(ry);
    R << c, 0, s,
         0, 1, 0,
         -s, 0, c;
    Eigen::MatrixXf xyz_corners(3, 8);
    xyz_corners << dim_l/2, dim_l/2, -dim_l/2, -dim_l/2, dim_l/2, dim_l/2, -dim_l/2, -dim_l/2,
                   dim_h/2, dim_h/2, dim_h/2, dim_h/2,  -dim_h/2, -dim_h/2, -dim_h/2, -dim_h/2,
                   dim_w/2, -dim_w/2, -dim_w/2, dim_w/2, dim_w/2, -dim_w/2, -dim_w/2, dim_w/2;
    Eigen::MatrixXf corners_3d(3,8);
    
    corners_3d = R * xyz_corners;
    MyMatrix83D result_corners_3d;
    result_corners_3d << corners_3d(0, 0) + location_x, corners_3d(1, 0) + location_y, corners_3d(2, 0) + location_z,
                         corners_3d(0, 1) + location_x, corners_3d(1, 1) + location_y, corners_3d(2, 1) + location_z,
                         corners_3d(0, 2) + location_x, corners_3d(1, 2) + location_y, corners_3d(2, 2) + location_z,
                         corners_3d(0, 3) + location_x, corners_3d(1, 3) + location_y, corners_3d(2, 3) + location_z,
                         corners_3d(0, 4) + location_x, corners_3d(1, 4) + location_y, corners_3d(2, 4) + location_z,
                         corners_3d(0, 5) + location_x, corners_3d(1, 5) + location_y, corners_3d(2, 5) + location_z,
                         corners_3d(0, 6) + location_x, corners_3d(1, 6) + location_y, corners_3d(2, 6) + location_z,
                         corners_3d(0, 7) + location_x, corners_3d(1, 7) + location_y, corners_3d(2, 7) + location_z;

    return result_corners_3d;
}

PYBIND11_MODULE(postprocess, m) 
{
    m.def("show_rgb_image_with_boxes", &show_rgb_image_with_boxes);
    m.def("get_3d_box", &get_3d_box);
}

int32_t main() {
  return 0;
}
