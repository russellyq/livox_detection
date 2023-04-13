from livox_dataset import Livox_Dataset
import tensorflow as tf
import numpy as np
import timeit
import sys
sys.path.append('..')
import config.config as cfg
from networks.model import *
import lib_cpp

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

T1 = np.array([[0.0, -1.0, 0.0, 0.0],
               [0.0, 0.0, -1.0, 0.0],
               [1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]]
              )


class Livox_Detector(object):
    def __init__(self, net, sess, ops):
        self.sess = sess
        self.net = net
        self.ops = ops

    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])
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
        #print('3D BBox', corners_3d)
        #print(location)
        #print(heading_angle)
        '''
        L = [location[0], location[1], location[2]]
        corners = np.vstack([x_corners, y_corners, z_corners])
        corners_3d = np.dot(R, corners)
        corners_3d = corners_3d + np.array(L, dtype=np.float32).reshape(3, 1)
        '''
        return corners_3d
    
    def data2voxel(self, pclist):
        data = [i*0 for i in range(HEIGHT*WIDTH*CHANNELS)]
        for line in pclist:
            X = float(line[0])
            Y = float(line[1])
            Z = float(line[2])
            if( Y > Y_MIN and Y < Y_MAX and
                X > X_MIN and X < X_MAX and
                Z > Z_MIN and Z < Z_MAX):
                channel = int((-Z + Z_MAX)/DZ)
                if abs(X)<3 and abs(Y)<3:
                    continue
                if (X > -overlap):
                    pixel_x = int((X - X_MIN + 2*overlap)/DX)
                    pixel_y = int((-Y + Y_MAX)/DY)
                    data[pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel] = 1
                if (X < overlap):
                    pixel_x = int((-X + overlap)/DX)
                    pixel_y = int((Y + Y_MAX)/DY)
                    data[pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel] = 1
        voxel = np.reshape(data, (HEIGHT, WIDTH, CHANNELS))
        return voxel

    def detect(self, batch_bev_img):
        
        feed_dict = {self.ops['input_bev_img_pl']: batch_bev_img}
        feature_out,\
            = self.sess.run([self.ops['end_points']['feature_out'],
                             ], feed_dict=feed_dict)
        result = lib_cpp.cal_result(feature_out[0,:,:,:], \
                                    cfg.BOX_THRESHOLD,overlap,X_MIN,HEIGHT, WIDTH, cfg.VOXEL_SIZE[0], cfg.VOXEL_SIZE[1], cfg.VOXEL_SIZE[2], cfg.NMS_THRESHOLD)
        # result: n*9
        # 1: object
        # 2: heading angle
        # 3: length
        # 4: width
        # 5: x
        # 6: y
        # 7: height
        # 8: z

        is_obj_list = result[:, 0].tolist()
        
        reg_m_x_list = result[:, 5].tolist()
        reg_w_list = result[:, 4].tolist()
        reg_l_list = result[:, 3].tolist()
        obj_cls_list = result[:, 1].tolist()
        reg_m_y_list = result[:, 6].tolist()
        reg_theta_list = result[:, 2].tolist()
        reg_m_z_list = result[:, 8].tolist()
        reg_h_list = result[:, 7].tolist()

        results = []
        for i in range(len(is_obj_list)):
            box3d_pts_3d = np.ones((8, 4), float)
            box3d_pts_3d[:, 0:3] = self.get_3d_box( \
                (reg_l_list[i], reg_w_list[i], reg_h_list[i]), \
                reg_theta_list[i], (reg_m_x_list[i], reg_m_z_list[i], reg_m_y_list[i]))
            
            box3d_pts_3d = np.dot(np.linalg.inv(T1), box3d_pts_3d.T).T  # n*4
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
            results.append([cls_name,
                            box3d_pts_3d[0][0], box3d_pts_3d[1][0], box3d_pts_3d[2][0], box3d_pts_3d[3][0],
                            box3d_pts_3d[4][0], box3d_pts_3d[5][0], box3d_pts_3d[6][0], box3d_pts_3d[7][0],
                            box3d_pts_3d[0][1], box3d_pts_3d[1][1], box3d_pts_3d[2][1], box3d_pts_3d[3][1],
                            box3d_pts_3d[4][1], box3d_pts_3d[5][1], box3d_pts_3d[6][1], box3d_pts_3d[7][1],
                            box3d_pts_3d[0][2], box3d_pts_3d[1][2], box3d_pts_3d[2][2], box3d_pts_3d[3][2],
                            box3d_pts_3d[4][2], box3d_pts_3d[5][2], box3d_pts_3d[6][2], box3d_pts_3d[7][2],
                            is_obj_list[i]])
        return results.tolist()



if __name__ == '__main__':
    num_samples = 14
    
    net = livox_model(HEIGHT, WIDTH, CHANNELS)
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(cfg.GPU_INDEX)):
            input_bev_img_pl = net.placeholder_inputs(cfg.BATCH_SIZE)
            end_points = net.get_model(input_bev_img_pl)

            saver = tf.train.Saver()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            saver.restore(sess, '../model/livoxmodel')
            ops = {'input_bev_img_pl': input_bev_img_pl,  # input
                    'end_points': end_points,  # output
                    }

    livox_detector = Livox_Detector(net, sess, ops)
    results = []
    smaple_matrix = []

    for i in range(num_samples):
        print(i)
        livox_data = Livox_Dataset(idx=i)
        points_list = livox_data.points
        labels = livox_data.anno
        vox = livox_detector.data2voxel(points_list)
        vox = np.expand_dims(vox, axis=0)
        result = livox_detector.detect(vox)
    
