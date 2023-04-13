
# 双雷达Lidar检测
# 都要进行旋转操作
import os
import re
import numpy as np
from numpy.core.arrayprint import printoptions
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
import lib_cpp
import preprocess
import postprocess
from utils.tools import write_csv, create_csv, write_data, evaluation_fuction
import open3d
from AB3DMOT_libs.model import AB3DMOT

frame_id = 0
num_frame = 0
# detection list with frame_id, id, cls_name, h, w, l, y, - x, z, theta, t

transformation = np.loadtxt('./refinement.txt')
R, T = transformation[0:3, 0:3], transformation[0:3, -1]

mnum1, mnum2 = 0, 0
marker_array1, marker_array2 = MarkerArray(), MarkerArray()
marker_array_text1, marker_array_text2 = MarkerArray(), MarkerArray()


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

points_list_all = np.empty((0,3))
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
                        
        rospy.init_node('livox_test', anonymous=True)

        self.lidar_sub1 = message_filters.Subscriber('/livox/lidar_1HDDH1200100801/time_sync', PointCloud2)
        self.lidar_sub2 = message_filters.Subscriber('/livox/lidar_3WEDH7600103381/time_sync', PointCloud2)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.lidar_sub1, self.lidar_sub2], 1, 0.1)
        self.ts.registerCallback(self.CallBack)

        self.marker_pub = rospy.Publisher(
            '/detect_box3d', MarkerArray, queue_size=1)
        self.marker_text_pub = rospy.Publisher(
            '/text_det', MarkerArray, queue_size=1)
        self.pointcloud_pub = rospy.Publisher(
            '/pointcloud', PointCloud2, queue_size=1)

        self.marker_pub2 = rospy.Publisher(
            '/detect_box3d_lidar2', MarkerArray, queue_size=1)
        self.marker_text_pub2 = rospy.Publisher(
            '/text_det_lidar2', MarkerArray, queue_size=1)
        self.pointcloud_pub2 = rospy.Publisher(
            '/pointcloud_lidar2', PointCloud2, queue_size=1)
        
        self.lidar_rotation_m, self.lidar_translation_v = rotation_translation()
        self.lidar_rotation_m2, self.lidar_translation_v2 = rotation_translation(alpha=cfg.tower_confg2['alpha'], theta=cfg.tower_confg2['theta'], gama=cfg.tower_confg2['gamma'], 
                                                                                x=cfg.tower_confg2['x'], y=cfg.tower_confg2['y'], z=cfg.tower_confg2['z'])
        
        self.camera_rotation_m, self.camera_translation_v = rotation_translation(alpha=cfg.tower_confg['alpha'], theta=cfg.tower_confg['gamma'], gama=cfg.tower_confg['theta'], x=-cfg.tower_confg['y'], y=-cfg.tower_confg['z'], z=cfg.tower_confg['x'])
        
        self.mot_tracker1 = AB3DMOT(max_age=2, min_hits=2)
        self.mot_tracker2 = AB3DMOT(max_age=2, min_hits=2)
    

    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    def roll(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    

    def get_3d_box(self, box_size, heading_angle, location):
        ''' Calculate 3D bounding box corners from its parameterization.
        Input:
            box_size: tuple of (l,w,h)
            heading_angle: rad scalar, clockwise from pos x axis
            location: tuple of (x,y,z)
        Output:
            corners_3d: numpy array of shape (8,3) for 3D box cornders
        '''
        R = self.roty(heading_angle)
        l, w, h = box_size

        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + location[0]
        corners_3d[1, :] = corners_3d[1, :] + location[1]
        corners_3d[2, :] = corners_3d[2, :] + location[2]
        corners_3d = np.transpose(corners_3d)
        return corners_3d


    def lidar_to_camera_box(self, box_size, heading_angle, center):
        x, y, z = center
        x, y, z = z, -x, -y
        l, w, h = box_size
        rz =  heading_angle
        (x, y, z), h, w, l, ry = self.lidar_to_camera(x, y, z), h, w, l, rz
        return [x, y, z, h, w, l, ry]

    def lidar_to_camera(self, x, y, z):
        p = np.array([x, y, z, 1])
        p = np.matmul(self.V2C, p)
        p = np.matmul(self.R0, p)
        p = p[0:3]
        return tuple(p)

    # def show_rgb_image_with_boxes(self, labels, img):
    #     n = 1
    #     for label in labels:
    #         location, dim, ry = label[0:3], label[3:6], label[6]
    #         corners_3d = self.compute_box_3d(dim, location, ry)
    #         corners_3d_camera = self.inverse_roate_pcd(corners_3d)
    #         corners_2d = self.project_to_image(corners_3d_camera)
    #         if n ==1:
    #             print(self.P)
    #             print(corners_3d_camera)
    #             print(corners_2d)
    #         n+=1   
    #         img = draw_box_3d(img, corners_2d)

    #     return img
    

    def compute_box_3d(self, dim, location, heading_angle):
        # dim: 3
        # location: 3
        # ry: 1
        # return: 8 x 3
        R = self.roty(heading_angle)
        h, w, l = dim
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + location[0]
        corners_3d[1, :] = corners_3d[1, :] + location[1]
        corners_3d[2, :] = corners_3d[2, :] + location[2]
        return corners_3d.transpose(1, 0)
    

    def project_to_image(self, pts_3d): 
        # pts_3d: 8 x 3
        # P: 3 x 4
        # return: 8 x 2
        pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
        pts_2d = np.dot(self.P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        return pts_2d.astype(np.int)
    
    def data2voxel(self, pclist):
        # n * 3
        w, h = pclist.shape  
        # voxel = preprocess.data2voxel(pclist.reshape((1, w*h)))
        voxel = preprocess.data2voxel(pclist.reshape((1, w*h)), DX, DY, DZ, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, overlap, HEIGHT, WIDTH, CHANNELS, LENGTH)
        return voxel

    def detect(self, batch_bev_img):

        #print(HEIGHT, WIDTH, CHANNELS)
        feed_dict = {self.ops['input_bev_img_pl']: batch_bev_img}
        feature_out,\
            = self.sess.run([self.ops['end_points']['feature_out'],
                             ], feed_dict=feed_dict)
        
        # tensor_name = [tensor.name for tensor in self.sess.graph_def.node]
        # print(tensor_name)
        # tensor_name_list = [tensor.name for tensor in self.sess.graph_def.node]
        # f = open('node.txt', 'w+')
        # for tensor_name in tensor_name_list:
        #     f.write(str(tensor_name) + '\n')
        # f.close()

        # write to pb file
        # constant_graph = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["Conv_39/BiasAdd"])
        # with tf.gfile.FastGFile('model_pb/saved_model.pb', mode='wb') as f:
        #     f.write(constant_graph.SerializeToString())
        
        result = lib_cpp.cal_result(feature_out[0,:,:,:], \
                                    cfg.BOX_THRESHOLD,overlap,X_MIN,HEIGHT, WIDTH, cfg.VOXEL_SIZE[0], cfg.VOXEL_SIZE[1], cfg.VOXEL_SIZE[2], cfg.NMS_THRESHOLD)
        is_obj_list = result[:, 0].tolist()
        obj_cls_list = result[:, 1].tolist()
        
        # reg_m_x_list = result[:, 5].tolist()
        # reg_w_list = result[:, 4].tolist()
        # reg_l_list = result[:, 3].tolist()
        # reg_m_y_list = result[:, 6].tolist()
        # reg_theta_list = result[:, 2].tolist()
        # reg_m_z_list = result[:, 8].tolist()
        # reg_h_list = result[:, 7].tolist()

        result_raw = []
        for i in range(len(is_obj_list)):
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

            # result_raw.append([cls_name, is_obj_list[i], reg_h_list[i], reg_w_list[i], reg_l_list[i], reg_m_x_list[i], reg_m_z_list[i], reg_m_y_list[i], reg_theta_list[i]])
            result_raw.append([cls_name, result[i, 0], result[i, 7], result[i, 4], result[i, 3], result[i, 5], result[i, 8], result[i, 6], result[i, 2]])
            
        return np.array(result_raw).reshape(-1, 9)
    

    def post_process(self, dets, info, lidar_translation_v, lidar_rotation_m, file_name=None):

        t =  timeit.default_timer()

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
        kitti_dets = []


        save_data = []

        for i in range(len(is_obj_list)):
            
            box3d_pts_3d = np.ones((8, 4), float)

            box3d_pts_3d[:, 0:3] = self.get_3d_box( \
                (reg_l_list[i], reg_w_list[i], reg_h_list[i]), \
                reg_theta_list[i], (reg_m_x_list[i], reg_m_z_list[i], reg_m_y_list[i]))
            
            box3d_pts_3d = np.dot(np.linalg.inv(T1), box3d_pts_3d.T).T  # n*4

            # box3d_kitti_dets = self.lidar_to_camera_box( \
            #     (reg_l_list[i], reg_w_list[i], reg_h_list[i]), \
            #     reg_theta_list[i], (reg_m_x_list[i], reg_m_z_list[i], reg_m_y_list[i]))

            # kitti_dets.append(box3d_kitti_dets)
            cls_name = str(cls_names[i])
            results.append([cls_name,  
                            box3d_pts_3d[0][0], box3d_pts_3d[1][0], box3d_pts_3d[2][0], box3d_pts_3d[3][0],
                            box3d_pts_3d[4][0], box3d_pts_3d[5][0], box3d_pts_3d[6][0], box3d_pts_3d[7][0],
                            box3d_pts_3d[0][1], box3d_pts_3d[1][1], box3d_pts_3d[2][1], box3d_pts_3d[3][1],
                            box3d_pts_3d[4][1], box3d_pts_3d[5][1], box3d_pts_3d[6][1], box3d_pts_3d[7][1],
                            box3d_pts_3d[0][2], box3d_pts_3d[1][2], box3d_pts_3d[2][2], box3d_pts_3d[3][2],
                            box3d_pts_3d[4][2], box3d_pts_3d[5][2], box3d_pts_3d[6][2], box3d_pts_3d[7][2],
                            is_obj_list[i], ids[i]])
                        
            save_data.append([frame_id, ids[i], cls_name, reg_h_list[i], reg_w_list[i], reg_l_list[i], reg_m_y_list[i], - reg_m_x_list[i], reg_m_z_list[i], reg_theta_list[i], t,
                            box3d_pts_3d[0][0], box3d_pts_3d[1][0], box3d_pts_3d[2][0], box3d_pts_3d[3][0],
                            box3d_pts_3d[4][0], box3d_pts_3d[5][0], box3d_pts_3d[6][0], box3d_pts_3d[7][0],
                            box3d_pts_3d[0][1], box3d_pts_3d[1][1], box3d_pts_3d[2][1], box3d_pts_3d[3][1],
                            box3d_pts_3d[4][1], box3d_pts_3d[5][1], box3d_pts_3d[6][1], box3d_pts_3d[7][1],
                            box3d_pts_3d[0][2], box3d_pts_3d[1][2], box3d_pts_3d[2][2], box3d_pts_3d[3][2],
                            box3d_pts_3d[4][2], box3d_pts_3d[5][2], box3d_pts_3d[6][2], box3d_pts_3d[7][2],])
        
        
        return results, np.array(save_data) #, np.array(kitti_dets).reshape(-1, 7)
    
        
    def CallBack(self, lidar_msg1, lidar_msg2):
        start = timeit.default_timer()
        global mnum1, mnum2
        

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'livox_frame'
        
        ##############################################
        # lidar1
        points_list1 = []
        for point in pcl2.read_points(lidar_msg1, skip_nans=True, field_names=("x", "y", "z")):
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
                continue
            if np.abs(point[0]) < 1.5 and np.abs(point[1]) < 1.5:
                continue
            # if point[0] < X_MAX and point[0] > X_MIN and point[1] < Y_MAX and point[1] > Y_MIN and point[2] < Z_MAX and point[2]>Z_MIN:
            # points_list.append([point[0], point[1], point[2]])
            points_list1.append(point[0])
            points_list1.append(point[1])
            points_list1.append(point[2])

        points_list1 = preprocess.rotate_translate_pcd(points_list1, self.lidar_rotation_m, self.lidar_translation_v)
        # if there is any rotation of livox ?
        
        # republish pointcloud2
        
        # poincloud to voxel data
        # vox1 = self.data2voxel(points_list1)
        # vox1 = np.expand_dims(vox1, axis=0)

        # # detection
        # result_raw1 = self.detect(vox1)
        
        # # MOT 3D tracker
        # # result_raw:
        # # N*9: class_name, probability; h, w, l, x, y, z, theta;
        # dets_all1 = {'dets': np.float64(result_raw1[:, 2:]), 'info': np.c_[result_raw1[:, 0:2], np.zeros((result_raw1.shape[0], 3))]}
        
        # # info: class_name, probability; Vx, Vy, Vz
        # trackers1 = self.mot_tracker1.update(dets_all1)

        # # post process 
        # if trackers1.size != 0:
        #     result1, source_pts = self.post_process(trackers1[:, 0:8], trackers1[:, 8:], self.lidar_translation_v, self.lidar_rotation_m, file_name='result1.csv')
        #     print('track_numbers', len(result1))


        #     boxes1 = result1
        # else:
        #     boxes1 = []
        
        # marker_array1.markers.clear()
        # marker_array_text1.markers.clear()
        # for obid in range(len(boxes1)):
        #     ob = boxes1[obid]
        #     tid = 0
        #     detect_points_set = []
        #     for i in range(0, 8):
        #         detect_points_set.append(Point(ob[i+1], ob[i+9], ob[i+17]))

        #     marker = Marker()
        #     marker.header.frame_id = 'livox_frame'
        #     marker.header.stamp = rospy.Time.now()

        #     marker.id = obid*2
        #     marker.action = Marker.ADD
        #     marker.type = Marker.LINE_LIST

        #     marker.lifetime = rospy.Duration(0)

        #     marker.color.r = 1
        #     marker.color.g = 0
        #     marker.color.b = 0

        #     marker.color.a = 1
        #     marker.scale.x = 0.2
        #     marker.points = []

        #     for line in lines:
        #         marker.points.append(detect_points_set[line[0]])
        #         marker.points.append(detect_points_set[line[1]])
                
        #     marker_array1.markers.append(marker)

        #     marker1 = Marker()
        #     marker1.header.frame_id = 'livox_frame'
        #     marker1.header.stamp = rospy.Time.now()
        #     marker1.ns = "basic_shapes"
                
        #     marker1.id = obid*2+1
        #     marker1.action = Marker.ADD

        #     marker1.type = Marker.TEXT_VIEW_FACING

        #     marker1.lifetime = rospy.Duration(0)

        #     marker1.color.r = 1  # cr
        #     marker1.color.g = 1  # cg
        #     marker1.color.b = 1  # cb

        #     marker1.color.a = 1
        #     marker1.scale.z = 1

        #     marker1.pose.orientation.w = 1.0
        #     marker1.pose.position.x = (ob[1]+ob[3])/2
        #     marker1.pose.position.y = (ob[9]+ob[11])/2
        #     marker1.pose.position.z = (ob[21]+ob[23])/2+1
        #     # marker1.text =  str(ob[-1]) + ob[0]+':'+str(np.floor(ob[25]*100)/100)
        #     #marker1.text =  str(ob[-1]) +': ' + ob[0] + ': '+str(np.floor(ob[-2]*100)/100)
        #     marker1.text =  str(ob[-1]) +': ' + ob[0]

        #     marker_array_text1.markers.append(marker1)
                
        # if mnum1 > len(boxes1):
        #     for obid in range(len(boxes1), mnum1):
        #         marker = Marker()
        #         marker.header.frame_id = 'livox_frame'
        #         marker.header.stamp = rospy.Time.now()
        #         marker.id = obid*2
        #         marker.action = Marker.ADD
        #         marker.type = Marker.LINE_LIST
        #         marker.lifetime = rospy.Duration(0.01)
        #         marker.color.r = 1
        #         marker.color.g = 1
        #         marker.color.b = 1
        #         marker.color.a = 0
        #         marker.scale.x = 0.2
        #         marker.points = []
                    
        #         marker_array1.markers.append(marker)
               

        #         marker1 = Marker()
        #         marker1.header.frame_id = 'livox_frame'
        #         marker1.header.stamp = rospy.Time.now()
        #         marker1.ns = "basic_shapes"

        #         marker1.id = obid*2+1
        #         marker1.action = Marker.ADD

        #         marker1.type = Marker.TEXT_VIEW_FACING

        #         marker1.lifetime = rospy.Duration(0.01)
        #         marker1.color.a = 0
        #         marker1.text = 'aaa'

        #         marker_array_text1.markers.append(marker1)                   
        # mnum1 = len(boxes1)

        # self.marker_pub.publish(marker_array1)
        # self.marker_text_pub.publish(marker_array_text1)
        

        

        # marker_array1.markers.clear()
        # marker_array_text1.markers.clear()
        
        ##############################################
        # lidar2
        points_list2 = []
        for point in pcl2.read_points(lidar_msg2, skip_nans=True, field_names=("x", "y", "z")):
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
                continue
            # if np.abs(point[0]) < 1.5 and np.abs(point[1]) < 1.5:
            #     continue
            # if point[0] < X_MAX and point[0] > X_MIN and point[1] < Y_MAX and point[1] > Y_MIN and point[2] < Z_MAX and point[2]>Z_MIN:
            # points_list2.append([point[0], point[1], point[2]])
            points_list2.append(point[0])
            points_list2.append(point[1])
            points_list2.append(point[2])
        
        points_list2 = np.array(points_list2).reshape((-1, 3))
        points_list2 = (np.matmul(R, points_list2.T) + T.reshape((3,1))).T
        points_list2 = points_list2.reshape((1,-1)).tolist()
        points_list2 = points_list2[0]
        
        # points_list = np.asarray(points_list)
        # points_list = self.rotate_translate_pcd(points_list)
        points_list2 = preprocess.rotate_translate_pcd(points_list2, self.lidar_rotation_m, self.lidar_translation_v)
        # if there is any rotation of livox ?
        
        # # republish pointcloud2

        # points_list2, transformation_refine = icp_pcd(points_list2, points_list1)
        # transformation_refine = np.matmul(transformation, transformation_refine)
        # np.savetxt('./refinement.txt',transformation_refine)
        
        #pointcloud_msg1 = pcl2.create_cloud_xyz32(header, points_list_all)
        #points_list2 = np.concatenate((points_list1, points_list2), axis=0)
        
        # points = np.zeros((points_list1.shape[0], 4))
        # points[:, 0:3] = points_list1
        # np.save('my_data.npy', points) 
        pointcloud_msg1 = pcl2.create_cloud_xyz32(header, points_list1)
        self.pointcloud_pub.publish(pointcloud_msg1)

        pointcloud_msg2 = pcl2.create_cloud_xyz32(header, points_list2)
        self.pointcloud_pub2.publish(pointcloud_msg2)
        # # poincloud to voxel data
        # vox2 = self.data2voxel(points_list2)
        # vox2 = np.expand_dims(vox2, axis=0)

        # # detection
        # result_raw2 = self.detect(vox2)
        
        # # MOT 3D tracker
        # # result_raw:
        # # N*9: class_name, probability; h, w, l, x, y, z, theta;
        # dets_all = {'dets': np.float64(result_raw2[:, 2:]), 'info': np.c_[result_raw2[:, 0:2], np.zeros((result_raw2.shape[0], 3))]}
        
        # # info: class_name, probability; Vx, Vy, Vz
        # trackers2 = self.mot_tracker2.update(dets_all)

        # # post process 
        # if trackers2.size != 0:
        #     result2, target_pts = self.post_process(trackers2[:, 0:8], trackers2[:, 8:], self.lidar_translation_v2, self.lidar_rotation_m2, file_name='result2.csv')
        #     print('track_numbers', len(result2))
        #     boxes2 = result2
        # else:
        #     boxes2 = []
        
        # marker_array2.markers.clear()
        # marker_array_text2.markers.clear()
        # for obid in range(len(boxes2)):
        #     ob = boxes2[obid]
        #     tid = 0
        #     detect_points_set = []
        #     for i in range(0, 8):
        #         detect_points_set.append(Point(ob[i+1], ob[i+9], ob[i+17]))

        #     marker = Marker()
        #     marker.header.frame_id = 'livox_frame'
        #     marker.header.stamp = rospy.Time.now()

        #     marker.id = obid*2
        #     marker.action = Marker.ADD
        #     marker.type = Marker.LINE_LIST

        #     marker.lifetime = rospy.Duration(0)

        #     marker.color.r = 0
        #     marker.color.g = 1
        #     marker.color.b = 0

        #     marker.color.a = 1
        #     marker.scale.x = 0.2
        #     marker.points = []

        #     for line in lines:
        #         marker.points.append(detect_points_set[line[0]])
        #         marker.points.append(detect_points_set[line[1]])
                
        #     marker_array2.markers.append(marker)

        #     marker1 = Marker()
        #     marker1.header.frame_id = 'livox_frame'
        #     marker1.header.stamp = rospy.Time.now()
        #     marker1.ns = "basic_shapes"
                
        #     marker1.id = obid*2+1
        #     marker1.action = Marker.ADD

        #     marker1.type = Marker.TEXT_VIEW_FACING

        #     marker1.lifetime = rospy.Duration(0)

        #     marker1.color.r = 1  # cr
        #     marker1.color.g = 1  # cg
        #     marker1.color.b = 1  # cb

        #     marker1.color.a = 1
        #     marker1.scale.z = 1

        #     marker1.pose.orientation.w = 1.0
        #     marker1.pose.position.x = (ob[1]+ob[3])/2
        #     marker1.pose.position.y = (ob[9]+ob[11])/2
        #     marker1.pose.position.z = (ob[21]+ob[23])/2+1
        #     # marker1.text =  str(ob[-1]) + ob[0]+':'+str(np.floor(ob[25]*100)/100)
        #     #marker1.text =  str(ob[-1]) +': ' + ob[0] + ': '+str(np.floor(ob[-2]*100)/100)
        #     marker1.text =  str(ob[-1]) +': ' + ob[0]

        #     marker_array_text2.markers.append(marker1)
                
        # if mnum2 > len(boxes2):
        #     for obid in range(len(boxes2), mnum2):
        #         marker = Marker()
        #         marker.header.frame_id = 'livox_frame'
        #         marker.header.stamp = rospy.Time.now()
        #         marker.id = obid*2
        #         marker.action = Marker.ADD
        #         marker.type = Marker.LINE_LIST
        #         marker.lifetime = rospy.Duration(0.01)
        #         marker.color.r = 1
        #         marker.color.g = 1
        #         marker.color.b = 1
        #         marker.color.a = 0
        #         marker.scale.x = 0.2
        #         marker.points = []
                    
        #         marker_array1.markers.append(marker)
               

        #         marker1 = Marker()
        #         marker1.header.frame_id = 'livox_frame'
        #         marker1.header.stamp = rospy.Time.now()
        #         marker1.ns = "basic_shapes"

        #         marker1.id = obid*2+1
        #         marker1.action = Marker.ADD

        #         marker1.type = Marker.TEXT_VIEW_FACING

        #         marker1.lifetime = rospy.Duration(0.01)
        #         marker1.color.a = 0
        #         marker1.text = 'aaa'

        #         marker_array_text2.markers.append(marker1)                   
        # mnum2 = len(boxes2)

        # self.marker_pub2.publish(marker_array2)
        # self.marker_text_pub2.publish(marker_array_text2)
        

        

        # marker_array2.markers.clear()
        # marker_array_text2.markers.clear()

        # if target_pts.shape[0] >=3 and source_pts.shape[0]>=3:
        #     transformation = evaluation_fuction(np.array(target_pts), np.array(source_pts))

        stop = timeit.default_timer()
        print('det_time: ', stop - start)
        print('FPS: ', 1/(stop - start))
        print('\n')

        


    def rotate_translate_pcd(self, point_cloud):
        # point_cloud shpae: n * 3
        # return n * 3
        pcd = np.matmul(self.lidar_rotation_m, point_cloud.T) + self.lidar_translation_v
        return pcd.T

    def inverse_roate_pcd(self, corners_3d, lidar_translation_v, lidar_rotation_m):
        # corners_3d shape (n*3)
        corners_3d = corners_3d.T - np.float64(lidar_translation_v)
        corners_3d = np.matmul(np.linalg.inv(lidar_rotation_m), corners_3d)
        return corners_3d.T
    


def icp_pcd(source, target):
    init = np.identity(4)
    threshold = 0.2
    src_pcd = open3d.geometry.PointCloud()
    src_pcd.points = open3d.utility.Vector3dVector(source)
    open3d.geometry.estimate_normals(src_pcd)
    dst_pcd = open3d.geometry.PointCloud()
    dst_pcd.points = open3d.utility.Vector3dVector(target)
    open3d.geometry.estimate_normals(dst_pcd)

    result = open3d.registration.registration_icp(src_pcd, dst_pcd, threshold, init, open3d.registration.TransformationEstimationPointToPlane())
    print(result)
    src_pcd.transform(result.transformation)
    evaluatrion = open3d.registration.evaluate_registration(src_pcd, dst_pcd, threshold, result.transformation)
    print('evaluation: ', evaluatrion)
    return np.asarray(src_pcd.points), result.transformation
    



if __name__ == '__main__':
    livox = Detector()
    # if source_pts.size != 0 and target_pts.size != 0:
    #     evaluation_fuction(source_pts, target_pts)
    rospy.spin()
    
