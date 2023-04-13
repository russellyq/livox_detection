import numpy as np

CLASSES = ['car', 'bus', 'truck', 'pedestrian', 'bimo']

RANGE = {'X_MIN': 0,
         'X_MAX': 100.8,
         'Y_MIN': -22.4,
         'Y_MAX': 22.4,
         'Z_MIN': -3.0,
         'Z_MAX': 3.0}

VOXEL_SIZE = [0.2, 0.2, 0.2]
BATCH_SIZE = 1
MODEL_PATH = "model/livoxmodel"

OVERLAP = 11.2

GPU_INDEX = 0
NMS_THRESHOLD = 0.1
BOX_THRESHOLD = 0.2
# BOX_THRESHOLD = 0.9 # self.mot_tracker = AB3DMOT(max_age=3,min_hits=4)
# BOX_THRESHOLD = 0.8 # self.mot_tracker = AB3DMOT(max_age=5,min_hits=5) # (Maha MOT)

tower_confg2 = {
    'alpha': 0*np.pi/180,
    'theta': 60*np.pi/180,
    'gamma': -9*np.pi/180,
    'x': 0,
    'y': 0,
    'z': 5
}

tower_confg = {
    'alpha': 0*np.pi/180,
    'theta': 30*np.pi/180,
    'gamma': 1*np.pi/180,
    'x': 0,
    'y': 0,
    'z': 4
}

CLASS_NAME_TO_ID = {
    'car': 0,
    'bus': 1,
    'truck': 2,
    'bimo': 3,
    'pedestrian': 4,
    'dog': 5
}