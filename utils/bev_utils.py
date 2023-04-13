import numpy as np

def makeBEVMap(pcd, cfg):
    pcd = get_filtered_lidar(pcd, cfg)
    #print(pcd.shape)
    
    DX = cfg.VOXEL_SIZE[0]
    DY = cfg.VOXEL_SIZE[1]
    DZ = cfg.VOXEL_SIZE[2]
    X_MIN = cfg.RANGE['X_MIN']
    X_MAX = cfg.RANGE['X_MAX']
    Y_MIN = cfg.RANGE['Y_MIN']
    Y_MAX = cfg.RANGE['Y_MAX']
    

    overlap = cfg.OVERLAP
    HEIGHT = round((X_MAX - X_MIN+2*overlap) / DX) + 1
    WIDTH = round((Y_MAX - Y_MIN) / DY) + 1
    print(HEIGHT, WIDTH)
    Z_MIN = cfg.RANGE['Z_MIN']
    Z_MAX = cfg.RANGE['Z_MAX']
    CHANNELS = round((Z_MAX - Z_MIN) / DZ)
    DISCRETIZATION = (X_MAX - X_MIN) / HEIGHT
    
    

    DISCRETIZATION = (cfg.RANGE['X_MAX'] - cfg.RANGE['X_MIN']) / HEIGHT 

    # Discretize Feature Map
    pcd = np.copy(pcd)
    pcd[:, 0] = np.int_(np.floor(pcd[:, 0] / DISCRETIZATION))
    pcd[:, 1] = np.int_(np.floor(pcd[:, 1] / DISCRETIZATION) + WIDTH / 2)

    # sort-3times
    sorted_indices = np.lexsort((-pcd[:, 2], pcd[:, 1], pcd[:, 0]))
    pcd = pcd[sorted_indices]
    _, unique_indices, unique_counts = np.unique(pcd[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = pcd[unique_indices]
    print(PointCloud_top.shape)

    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((HEIGHT, WIDTH))
    intensityMap = np.zeros((HEIGHT, WIDTH))
    densityMap = np.zeros((HEIGHT, WIDTH))

    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(Z_MAX - Z_MIN))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, HEIGHT , WIDTH ))
    RGB_Map[2, :, :] = densityMap[:HEIGHT, :WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:HEIGHT, :WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:HEIGHT, :WIDTH]  # b_map

    return RGB_Map

def get_filtered_lidar(lidar, cfg):
    minX = cfg.RANGE['X_MIN']
    maxX = cfg.RANGE['X_MAX']
    minY = cfg.RANGE['Y_MIN']
    maxY = cfg.RANGE['Y_MAX']
    minZ = cfg.RANGE['Z_MIN']
    maxZ = cfg.RANGE['Z_MAX']

    # Remove the point out of range x,y,z
    mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
    lidar = lidar[mask]
    lidar[:, 2] = lidar[:, 2] - minZ
    return lidar