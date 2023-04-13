# 暂时用于 单个 velodyne的检测以及显示
# config file
# preprocess file
import os
import numpy as np
import tensorflow as tf
import copy
import config.config as cfg
from networks.model import *
import time
from cv_bridge import CvBridge, CvBridgeError
import rospy
import std_msgs.msg
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, Image
# from geometry_msgs.msg import Point32
# from geometry_msgs.msg import Quaternion
import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import cv2
import message_filters
import timeit
import preprocess
import postprocess
import lib_cpp
from AB3DMOT_libs.model import AB3DMOT
frame_id = 0
# 
# sess = tf.Session(config=tf.ConfigProto())
mnum = 0
marker_array = MarkerArray()
marker_array_text = MarkerArray()


DX = cfg.VOXEL_SIZE[0]
DY = cfg.VOXEL_SIZE[1]
DZ = cfg.VOXEL_SIZE[2]

X_MIN = cfg.RANGE['X_MIN']
X_MAX = cfg.RANGE['X_MAX']

Y_MIN = cfg.RANGE['Y_MIN']
Y_MAX = cfg.RANGE['Y_MAX']

Z_MIN = cfg.RANGE['Z_MIN']
Z_MAX = cfg.RANGE['Z_MAX']

overlap = cfg.OVERLAP
HEIGHT = round((X_MAX - X_MIN+2*overlap) / DX)
WIDTH = round((Y_MAX - Y_MIN) / DY)
CHANNELS = round((Z_MAX - Z_MIN) / DZ)
LENGTH = HEIGHT*WIDTH*CHANNELS

print(HEIGHT, WIDTH, CHANNELS)

T1 = np.array([[0.0, -1.0, 0.0, 0.0],
               [0.0, 0.0, -1.0, 0.0],
               [1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]]
              )
lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
         [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


def rotation_translation(alpha=cfg.tower_confg['alpha'], theta=cfg.tower_confg['theta'], gama=cfg.tower_confg['gamma'], 
                         x=cfg.tower_confg['x'], y=cfg.tower_confg['y'], z=cfg.tower_confg['z']):
    rz = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    rx = np.array([[1, 0, 0], [0, np.cos(gama), -np.sin(gama)], [0, np.sin(gama), np.cos(gama)]])
    return np.matmul(rz, np.matmul(ry, rx)), np.array([[x], [y], [z]])

def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle

def draw_box_3d(img, corners, color=(0, 0, 255)):
    image = img.copy()
    ''' Draw 3d bounding box in image
        corners: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    '''

    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                     (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]), color, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                     (corners[f[2], 0], corners[f[2], 1]), color, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                     (corners[f[3], 0], corners[f[3], 1]), color, 1, lineType=cv2.LINE_AA)
    return image


class Detector(object):
    def __init__(self, *, nms_threshold=0.1, weight_file=None):
        self.V2C = np.array([[-0.00478365, -0.999914, 0.012176, 0.052084],
                             [0.0250726, -0.0122923, -0.99961, -0.106565], 
                             [0.999674, -0.0044765, 0.0251292, 0.351872]])

        self.R0 = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.P = np.array([[759.0084, 0, 990.1901, 0], 
                          [0, 757.6592, 537.1919, 0], 
                          [0, 0, 1, 0]])
        self.bridge = CvBridge()
        self.net = livox_model(HEIGHT, WIDTH, CHANNELS)
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(cfg.GPU_INDEX)):
                input_bev_img_pl = \
                    self.net.placeholder_inputs(cfg.BATCH_SIZE)
                end_points = self.net.get_model(input_bev_img_pl)

                saver = tf.train.Saver()
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
                config = tf.ConfigProto(gpu_options=gpu_options)
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True
                config.log_device_placement = False
                self.sess = tf.Session(config=config)
                saver.restore(self.sess, cfg.MODEL_PATH)
                self.ops = {'input_bev_img_pl': input_bev_img_pl,  # input
                            'end_points': end_points,  # output
                            }
                            # tensor_node_name = [tensor.name for tensor in self.sess.graph_def.node]
                # self.constant_graph = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["Conv_39/BiasAdd"])
                # self.trt_graph = trt.create_inference_graph(input_graph_def=self.constant_graph,
                #                                         outputs=["Conv_39/BiasAdd"],
                #                                         max_batch_size=cfg.BATCH_SIZE,
                #                                         max_workspace_size_bytes=600000,
                #                                         precision_mode="FP32")
                # self.output_node = tf.import_graph_def(self.trt_graph, ["Conv_39/BiasAdd"])
                # with tf.gfile.FastGFile('./model_pb/saved_model.pb', mode='wb') as f:
                #     f.write(constant_graph.SerializeToString())

                # 保存freeze graph形式
                # frozen_graph = tf.graph_util.convert_variables_to_constants(self.sess,
                #                                 tf.get_default_graph().as_graph_def(),
                #                                 output_node_names=['Conv_39/BiasAdd'])
                # with tf.gfile.FastGFile('./model_pb/saved_model.pb', mode='wb') as f:
                #     f.write(frozen_graph.SerializeToString())
                
                # # savedModel 保存形式
                # print(type(input_bev_img_pl))
                # tf.compat.v1.saved_model.simple_save(self.sess,"./model_saved/saved_model",inputs={'input_bev_img_pl': input_bev_img_pl},outputs={'end_points': end_points})

        rospy.init_node('livox_test', anonymous=True)
        self.lidar_sub = rospy.Subscriber('/livox/lidar/time_sync', PointCloud2, queue_size=1, callback=self.CallBack)

        self.marker_pub = rospy.Publisher(
            '/detect_box3d', MarkerArray, queue_size=1)
        self.marker_text_pub = rospy.Publisher(
            '/text_det', MarkerArray, queue_size=1)
        self.pointcloud_pub = rospy.Publisher(
            '/pointcloud', PointCloud2, queue_size=1)
        
        self.lidar_rotation_m, self.lidar_translation_v = rotation_translation()
                
        self.mot_tracker = AB3DMOT(max_age=4, min_hits=3)

    # def roty(self, t):
    #     c = np.cos(t)
    #     s = np.sin(t)
    #     return np.array([[c,  0,  s],
    #                      [0,  1,  0],
    #                      [-s, 0,  c]])

    # def get_3d_box(self, box_size, heading_angle, location):
    #     ''' Calculate 3D bounding box corners from its parameterization.
    #     Input:
    #         box_size: tuple of (l,w,h)
    #         heading_angle: rad scalar, clockwise from pos x axis
    #         location: tuple of (x,y,z)
    #     Output:
    #         corners_3d: numpy array of shape (8,3) for 3D box cornders
    #     '''
    #     R = self.roty(heading_angle)
    #     l, w, h = box_size

    #     x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    #     y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    #     z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        
    #     corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    #     corners_3d[0, :] = corners_3d[0, :] + location[0]
    #     corners_3d[1, :] = corners_3d[1, :] + location[1]
    #     corners_3d[2, :] = corners_3d[2, :] + location[2]
    #     corners_3d = np.transpose(corners_3d)
    #     return corners_3d


    # def lidar_to_camera_box(self, box_size, heading_angle, center):
    #     x, y, z = center
    #     x, y, z = z, -x, -y
    #     l, w, h = box_size
    #     rz =  heading_angle
    #     (x, y, z), h, w, l, ry = self.lidar_to_camera(x, y, z), h, w, l, rz
    #     return [x, y, z, h, w, l, ry]

    # def lidar_to_camera(self, x, y, z):
    #     p = np.array([x, y, z, 1])
    #     p = np.matmul(self.V2C, p)
    #     p = np.matmul(self.R0, p)
    #     p = p[0:3]
    #     return tuple(p)
    
    def data2voxel(self, pclist):
        # n * 3
        w, h = pclist.shape  
        voxel = preprocess.data2voxel(pclist.reshape((1, w*h)), DX, DY, DZ, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, overlap, HEIGHT, WIDTH, CHANNELS, LENGTH)
        return voxel

    def detect(self, batch_bev_img):

        feed_dict = {self.ops['input_bev_img_pl']: batch_bev_img}
        feature_out,\
            = self.sess.run([self.ops['end_points']['feature_out'],
                             ], feed_dict=feed_dict)

        # feature_out, \
        #     = self.sess.run(self.output_node)
        # constant_graph = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ['Conv_39/BiasAdd'])
        # with tf.gfile.FastGFile('./model_pb/saved_model.pb', mode='wb') as f:
        #     f.write(constant_graph.SerializeToString())
        
        
        result = lib_cpp.cal_result(feature_out[0,:,:,:], \
                                    cfg.BOX_THRESHOLD,overlap,X_MIN,HEIGHT, WIDTH, cfg.VOXEL_SIZE[0], cfg.VOXEL_SIZE[1], cfg.VOXEL_SIZE[2], cfg.NMS_THRESHOLD)
        # is_obj_list = result[:, 0].tolist()
        obj_cls_list = result[:, 1].tolist()
        
        # reg_m_x_list = result[:, 5].tolist()
        # reg_w_list = result[:, 4].tolist()
        # reg_l_list = result[:, 3].tolist()
        # reg_m_y_list = result[:, 6].tolist()
        # reg_theta_list = result[:, 2].tolist()
        # reg_m_z_list = result[:, 8].tolist()
        # reg_h_list = result[:, 7].tolist()

        result_raw = []
        for i in range(len(obj_cls_list)):
            if int(obj_cls_list[i]) == 0:
                cls_name = "car"
            elif int(obj_cls_list[i]) == 1:
                cls_name = "bus"
            elif int(obj_cls_list[i]) == 2:
                cls_name = "truck"
            elif int(obj_cls_list[i]) == 3:
                cls_name = "pedestrian"
            else:
                cls_name = "bimo"
            result_raw.append([cls_name, result[i, 0], result[i, 7], result[i, 4], result[i, 3], result[i, 5], result[i, 8], result[i, 6], result[i, 2]])
            # result_raw.append([cls_name, is_obj_list[i], reg_h_list[i], reg_w_list[i], reg_l_list[i], reg_m_x_list[i], reg_m_z_list[i], reg_m_y_list[i], reg_theta_list[i]])
                      
        return np.array(result_raw).reshape(-1, 9)
    

    def post_process(self, dets, info):

        global frame_id
        cls_names = info[:, 0].tolist()

        is_obj_list = np.array(info[:, 1], dtype=float)
        reg_h_list = np.array(dets[:, 0], dtype=float)
        reg_w_list = np.array(dets[:, 1], dtype=float)
        reg_l_list = np.array(dets[:, 2], dtype=float)
        reg_m_x_list = np.array(dets[:, 3], dtype=float)
        reg_m_z_list = np.array(dets[:, 4], dtype=float)
        reg_m_y_list = np.array(dets[:, 5], dtype=float)
        reg_theta_list = np.array(dets[:, 6], dtype=float)
        ids = np.array(dets[:, -1])

        results = []

        # save_data = []

        for i in range(len(is_obj_list)):
            box3d_pts_3d = np.ones((8, 4), float)

            # box3d_pts_3d[:, 0:3] = self.get_3d_box( \
            #     (reg_l_list[i], reg_w_list[i], reg_h_list[i]), \
            #     reg_theta_list[i], (reg_m_x_list[i], reg_m_z_list[i], reg_m_y_list[i]))
            box3d_pts_3d[:, 0:3] = postprocess.get_3d_box(reg_h_list[i], reg_w_list[i], reg_l_list[i],
                                                          reg_m_x_list[i], reg_m_z_list[i], reg_m_y_list[i], reg_theta_list[i])
            
            box3d_pts_3d = np.dot(np.linalg.inv(T1), box3d_pts_3d.T).T  # n*4

            cls_name = str(cls_names[i])
            results.append([cls_name,  
                            box3d_pts_3d[0][0], box3d_pts_3d[1][0], box3d_pts_3d[2][0], box3d_pts_3d[3][0],
                            box3d_pts_3d[4][0], box3d_pts_3d[5][0], box3d_pts_3d[6][0], box3d_pts_3d[7][0],
                            box3d_pts_3d[0][1], box3d_pts_3d[1][1], box3d_pts_3d[2][1], box3d_pts_3d[3][1],
                            box3d_pts_3d[4][1], box3d_pts_3d[5][1], box3d_pts_3d[6][1], box3d_pts_3d[7][1],
                            box3d_pts_3d[0][2], box3d_pts_3d[1][2], box3d_pts_3d[2][2], box3d_pts_3d[3][2],
                            box3d_pts_3d[4][2], box3d_pts_3d[5][2], box3d_pts_3d[6][2], box3d_pts_3d[7][2],
                            is_obj_list[i], ids[i]])
            # save_data.append([frame_id, ids[i], cls_name, reg_h_list[i], reg_w_list[i], reg_l_list[i], reg_m_x_list[i], reg_m_z_list[i], reg_m_y_list[i], reg_theta_list[i]])

        return results
        
    def CallBack(self, lidar_msg):
        start = timeit.default_timer()
        global mnum

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'livox_frame'
        points_list = []
        for point in pcl2.read_points(lidar_msg, skip_nans=True, field_names=("x", "y", "z")):
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
                continue
            # if np.abs(point[0]) < 1.5 and np.abs(point[1]) < 1.5:
            #     continue
            # if point[0] < X_MAX and point[0] > X_MIN and point[1] < Y_MAX and point[1] > Y_MIN and point[2] < Z_MAX and point[2]>Z_MIN:
            # points_list.append([point[0], point[1], point[2]])
            points_list.append(point[0])
            points_list.append(point[1])
            points_list.append(point[2])
        
        # points_list = np.asarray(points_list)
        # points_list = self.rotate_translate_pcd(points_list)
        points_list = preprocess.rotate_translate_pcd(points_list, self.lidar_rotation_m, self.lidar_translation_v)
        # if there is any rotation of livox ?
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(points_list)
        # open3d.io.write_point_cloud("./sync.pcd", pcd)

        # # republish pointcloud2
        pointcloud_msg = pcl2.create_cloud_xyz32(header, points_list)

        # poincloud to voxel data
        vox = self.data2voxel(points_list)
        vox = np.expand_dims(vox, axis=0)

        # detection
        # t1 = timeit.default_timer()
        result_raw = self.detect(vox)
        
        # MOT 3D tracker
        dets_all = {'dets': np.float64(result_raw[:, 2:]), 'info': np.c_[result_raw[:, 0:2], np.zeros((result_raw.shape[0], 5))]}
        trackers = self.mot_tracker.update(dets_all)
        # t2 = timeit.default_timer()
        # print('inference_time: ', t2 - t1)
        # post process 
        if trackers.size != 0:
            result = self.post_process(trackers[:, 0:8], trackers[:, 8:])
            print('track_numbers', len(result))
            boxes = result
        else:
            boxes = []

        marker_array.markers.clear()
        marker_array_text.markers.clear()
        for obid in range(len(boxes)):
            ob = boxes[obid]
            tid = 0
            detect_points_set = []
            for i in range(0, 8):
                detect_points_set.append(Point(ob[i+1], ob[i+9], ob[i+17]))

            marker = Marker()
            marker.header.frame_id = 'livox_frame'
            marker.header.stamp = rospy.Time.now()

            marker.id = obid*2
            marker.action = Marker.ADD
            marker.type = Marker.LINE_LIST

            marker.lifetime = rospy.Duration(0)

            marker.color.r = 1
            marker.color.g = 0
            marker.color.b = 0

            marker.color.a = 1
            marker.scale.x = 0.2
            marker.points = []

            for line in lines:
                marker.points.append(detect_points_set[line[0]])
                marker.points.append(detect_points_set[line[1]])
                
            marker_array.markers.append(marker)

            marker1 = Marker()
            marker1.header.frame_id = 'livox_frame'
            marker1.header.stamp = rospy.Time.now()
            marker1.ns = "basic_shapes"
                
            marker1.id = obid*2+1
            marker1.action = Marker.ADD

            marker1.type = Marker.TEXT_VIEW_FACING

            marker1.lifetime = rospy.Duration(0)

            marker1.color.r = 1  # cr
            marker1.color.g = 1  # cg
            marker1.color.b = 1  # cb

            marker1.color.a = 1
            marker1.scale.z = 1

            marker1.pose.orientation.w = 1.0
            marker1.pose.position.x = (ob[1]+ob[3])/2
            marker1.pose.position.y = (ob[9]+ob[11])/2
            marker1.pose.position.z = (ob[21]+ob[23])/2+1
            # marker1.text =  str(ob[-1]) + ob[0]+':'+str(np.floor(ob[25]*100)/100)
            #marker1.text =  str(ob[-1]) +': ' + ob[0] + ': '+str(np.floor(ob[-2]*100)/100)
            marker1.text =  str(ob[-1]) +': ' + ob[0]

            marker_array_text.markers.append(marker1)
                
        if mnum > len(boxes):
            for obid in range(len(boxes), mnum):
                marker = Marker()
                marker.header.frame_id = 'livox_frame'
                marker.header.stamp = rospy.Time.now()
                marker.id = obid*2
                marker.action = Marker.ADD
                marker.type = Marker.LINE_LIST
                marker.lifetime = rospy.Duration(0.01)
                marker.color.r = 1
                marker.color.g = 1
                marker.color.b = 1
                marker.color.a = 0
                marker.scale.x = 0.2
                marker.points = []
                    
                marker_array.markers.append(marker)
               

                marker1 = Marker()
                marker1.header.frame_id = 'livox_frame'
                marker1.header.stamp = rospy.Time.now()
                marker1.ns = "basic_shapes"

                marker1.id = obid*2+1
                marker1.action = Marker.ADD

                marker1.type = Marker.TEXT_VIEW_FACING

                marker1.lifetime = rospy.Duration(0.01)
                marker1.color.a = 0
                marker1.text = 'aaa'

                marker_array_text.markers.append(marker1)                   
        mnum = len(boxes)
                
        self.marker_pub.publish(marker_array)
        self.marker_text_pub.publish(marker_array_text)


        self.pointcloud_pub.publish(pointcloud_msg)

        marker_array.markers.clear()
        marker_array_text.markers.clear()

        stop = timeit.default_timer()
        print('det_time: ', stop - start)
        print('FPS: ', 1/(stop - start))
        print('\n')


    def rotate_translate_pcd(self, point_cloud):
        # point_cloud shpae: 8 * 3
        # return 8 * 3
        pcd = np.matmul(self.lidar_rotation_m, point_cloud.T) + self.lidar_translation_v
        return pcd.T

    # def inverse_roate_pcd(self, corners_3d):
    #     # corners_3d shape (8*3)

    #     corners_3d = corners_3d - self.camera_translation_v.T
    #     corners_3d = np.matmul(self.camera_rotation_m, corners_3d.T)
    #     return corners_3d.T

    



if __name__ == '__main__':
    livox = Detector()
    rospy.spin()