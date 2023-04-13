import sys
import rospy
from std_msgs.msg import String
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtGui
import os
import config.config as cfg
from networks.model import *
import rospy
import std_msgs.msg
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
# from tools import write_data_IHI
import tensorrt as trt
import pycuda.driver as cuda
from livox_trt_model import TRTModel
from AB3DMOT_libs.model import AB3DMOT
from datetime import datetime
import timeit
import lib_cpp
import preprocess
import postprocess
frame_id = 0
det_list_pre = []
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




class PyGui(QWidget):
    def __init__(self):
        super(PyGui, self).__init__()
        self.setObjectName('PyGui')
        rospy.init_node('livox_detection_GUI')

        # GUI 模型
        self.start_inference = False
        
        self.my_hlayout1 = QHBoxLayout()
        self.my_hlayout2 = QHBoxLayout()
        self.my_hlayout3 = QHBoxLayout()

        self.my_hight_lable = QLabel()
        self.my_hight_lable.setText("Height:")
        self.my_hight_line = QLineEdit()
        self.my_hight_line.setValidator(QDoubleValidator())
        self.my_hight_line.move(80, 20)
        self.my_hight_line.resize(200, 30)
        self.my_hight_lable.move(20, 20)

        self.my_angle_lable = QLabel()
        self.my_angle_lable.setText("Angle:")
        self.my_angle_line = QLineEdit()
        self.my_angle_line.setValidator(QDoubleValidator())
        self.my_angle_line.move(80, 20)
        self.my_angle_line.resize(200, 30)
        self.my_angle_lable.move(20, 20)

        self.my_threshold_lable = QLabel()
        self.my_threshold_lable.setText("Threshold:")
        self.my_threshold_line = QLineEdit()
        self.my_threshold_line.setValidator(QDoubleValidator())
        self.my_threshold_line.setText('0.6')
        self.my_threshold_line.move(80, 20)
        self.my_threshold_line.resize(200, 30)
        self.my_threshold_lable.move(20, 20)

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_click)
        self.start_button.resize(200,32)
        self.start_button.move(80, 60) 

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_click)
        self.stop_button.resize(200,32)
        self.stop_button.move(80, 60)  

        self.exit_button = QPushButton('Exit', self)
        self.exit_button.clicked.connect(QCoreApplication.instance().quit)
        self.exit_button.resize(200,32)
        self.exit_button.move(80, 60)
        
        self.my_hlayout1.addWidget(self.my_hight_lable)
        self.my_hlayout1.addSpacing(50)
        self.my_hlayout1.addWidget(self.my_hight_line)
        self.my_hlayout1.addSpacing(50)
        self.my_hlayout1.addWidget(self.start_button)
        self.my_hlayout1.addSpacing(50)

        self.my_hlayout2.addWidget(self.my_angle_lable)
        self.my_hlayout2.addSpacing(50)
        self.my_hlayout2.addWidget(self.my_angle_line)
        self.my_hlayout2.addSpacing(50)
        self.my_hlayout2.addWidget(self.stop_button)
        self.my_hlayout2.addSpacing(50)

        self.my_hlayout3.addWidget(self.my_threshold_lable)
        self.my_hlayout3.addSpacing(50)
        self.my_hlayout3.addWidget(self.my_threshold_line)
        self.my_hlayout3.addSpacing(50)
        self.my_hlayout3.addWidget(self.exit_button)
        self.my_hlayout3.addSpacing(50)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.my_hlayout1)
        self.layout.addLayout(self.my_hlayout2)
        self.layout.addLayout(self.my_hlayout3)
        self.setLayout(self.layout)

        # Detector
        self.net = livox_model(HEIGHT, WIDTH, CHANNELS)
        input_bev_img_pl = self.net.placeholder_inputs(cfg.BATCH_SIZE)
        end_points = self.net.get_model(input_bev_img_pl)
        self.feature_out = np.zeros(end_points['feature_out'].shape, dtype=np.float32)

        # 加载 TRT 模型
        cuda.init()
        self.device = cuda.Device(0)  # enter your GPU id here
        self.ctx = self.device.make_context()
        self.TRT_Model = TRTModel(1*(HEIGHT, WIDTH, CHANNELS), self.feature_out.shape)
        self.graph = tf.get_default_graph()
        self.ctx.pop()
        del self.ctx

        self.marker_pub = rospy.Publisher(
            '/detect_box3d', MarkerArray, queue_size=1)
        self.marker_text_pub = rospy.Publisher(
            '/text_det', MarkerArray, queue_size=1)
        self.pointcloud_pub = rospy.Publisher(
            '/pointcloud', PointCloud2, queue_size=1)
 
        self.mot_tracker = AB3DMOT(max_age=3, min_hits=3)

        self.lidar_sub = rospy.Subscriber('/livox/lidar', PointCloud2, queue_size=1, callback=self.CallBack)

        rospy.on_shutdown(self.stop_click)
        # while not rospy.is_shutdown() and self.start_inference:
        #         rospy.spin()

    def rotation_translation(self, alpha, theta, gama, x, y, z):
        rz = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        rx = np.array([[1, 0, 0], [0, np.cos(gama), -np.sin(gama)], [0, np.sin(gama), np.cos(gama)]])
        return np.matmul(rz, np.matmul(ry, rx)), np.array([[x], [y], [z]])
 
    def start_click(self):
        print('Height: ' + self.my_hight_line.text())
        print('Angle: ' + self.my_angle_line.text())
        print('Threhold: ' + self.my_threshold_line.text())
        if self.my_hight_line.text() and self.my_angle_line.text() and self.my_threshold_line.text():
            self.start_inference = True
            cfg.BOX_THRESHOLD = float(self.my_threshold_line.text())
            self.lidar_rotation_m, self.lidar_translation_v = self.rotation_translation(alpha=0, theta=float(self.my_angle_line.text()) * np.pi / 180, gama=0, 
                         x=0, y=0, z=float(self.my_hight_line.text()))

    def stop_click(self):
        print('STOP')
        self.start_inference = False
    
    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    def get_3d_box(self, box_size, heading_angle, location):
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

    def data2voxel(self, pclist):
        w, h = pclist.shape  
        voxel = preprocess.data2voxel(pclist.reshape((1, w*h)), \
                                      DX, DY, DZ,   \
                                      X_MIN, X_MAX, \
                                      Y_MIN, Y_MAX, \
                                      Z_MIN, Z_MAX, \
                                      overlap, HEIGHT, WIDTH, CHANNELS, LENGTH)
        
        return voxel

    def detect(self, batch_bev_img):
        with self.graph.as_default():
            feature_out = self.TRT_Model.detect(batch_bev_img)

        result = lib_cpp.cal_result(feature_out[0,:,:,:], \
                                    cfg.BOX_THRESHOLD,overlap,X_MIN,HEIGHT, WIDTH, cfg.VOXEL_SIZE[0], cfg.VOXEL_SIZE[1], cfg.VOXEL_SIZE[2], cfg.NMS_THRESHOLD)
        is_obj_list = result[:, 0].tolist()
        obj_cls_list = result[:, 1].tolist()

        result_raw = []
        for i in range(len(is_obj_list)):
            if int(obj_cls_list[i]) == 0 or int(obj_cls_list[i]) == 1 or int(obj_cls_list[i]) == 2:
                if result[i, 3] < 7:
                    cls_name = 10 # "car"
                else:
                    cls_name = 11 #"big car"
            elif int(obj_cls_list[i]) == 3:
                cls_name = 20 # "pedestrian"
            else:
                cls_name = 12 # "cyclelist"
            result_raw.append([cls_name, result[i, 0], result[i, 7], result[i, 4], result[i, 3], result[i, 5], result[i, 8], result[i, 6], result[i, 2]])

        return np.array(result_raw).reshape(-1, 9)

    def post_process(self, dets, info):

        global frame_id
        global det_list_pre
        det_list_cur = []
        t =  timeit.default_timer()

        Now_time =  datetime.now()
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

        save_data = []

        for i in range(len(is_obj_list)):
            box3d_pts_3d = np.ones((8, 4), float)
            # c++ version
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
                                    
            X1, Y1, X2, Y2 = (box3d_pts_3d[0][0] + box3d_pts_3d[1][0])/2, (box3d_pts_3d[0][1] + box3d_pts_3d[1][1])/2, (box3d_pts_3d[2][0] + box3d_pts_3d[2][0])/2,(box3d_pts_3d[2][1] + box3d_pts_3d[2][1])/2

            det_list_cur.append([np.uint8(cls_names[i]), np.uint8(0), np.uint16(ids[i]), 
                                np.uint16(Now_time.year), np.uint16(Now_time.month), np.uint16(Now_time.day), np.uint16(Now_time.hour), np.uint16(Now_time.minute), np.uint16(Now_time.second), np.uint16(Now_time.microsecond / 1000), 
                                np.int16(X1 * 100), np.int16(Y1 * 100), np.int16(X2 * 100), np.int16(Y2 * 100), np.int16(0), np.int16(0), np.uint16(reg_l_list[i] * 100), np.uint16(reg_w_list[i] * 100), np.uint16(reg_h_list[i] * 100), 
                                reg_m_y_list[i] * 100, - reg_m_x_list[i] * 100, - reg_m_z_list[i] * 100, Now_time])

        det_list_cur = self.speed_calculate(det_list_pre, det_list_cur)
        frame_id += 1
        det_list_pre = det_list_cur
        return results

    def speed_calculate(self, det_list_pre, det_list_cur, save_data=True):
        global frame_id
        if len(det_list_pre) != 0:
            id_list_pre = [i[2] for i in det_list_pre]
            id_list_cur = [i[2] for i in det_list_cur]
            for det in det_list_cur:
                id = det[2]
                if id in id_list_pre:
                    dt = (det[-1] - det_list_pre[id_list_pre.index(id)][-1]).total_seconds()
                    vx = (det[-4] - det_list_pre[id_list_pre.index(id)][-4]) / dt
                    vy = (det[-3] - det_list_pre[id_list_pre.index(id)][-3]) / dt

                else:
                    vx, vy = 0, 0
                det[14], det[15] = np.int16(vx), np.int16(vy)

        #write_data_IHI('result.csv', np.array(det_list_cur)[:,0:-4], frame_id)
        return det_list_cur
    
    def CallBack(self, lidar_msg):
        if not self.start_inference: 
            return 
        
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

            points_list.append(point[0])
            points_list.append(point[1])
            points_list.append(point[2])

        points_list = preprocess.rotate_translate_pcd(points_list, self.lidar_rotation_m, self.lidar_translation_v)
        pointcloud_msg = pcl2.create_cloud_xyz32(header, points_list)
        # poincloud to voxel data
        vox = self.data2voxel(points_list)
        vox = np.expand_dims(vox, axis=0)

        # detection
        result_raw = self.detect(vox)
        
        # MOT 3D tracker
        dets_all = {'dets': np.float64(result_raw[:, 2:]), 'info': np.c_[result_raw[:, 0:2], np.zeros((result_raw.shape[0], 5))]}
        trackers = self.mot_tracker.update(dets_all)

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
            marker1.text =  str(int(ob[-1]))# +': ' + ob[0]

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

        
        
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    pyShow = PyGui()
    pyShow.show()
    sys.exit(app.exec_())