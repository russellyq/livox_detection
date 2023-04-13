import numpy as np
import csv
import os
import open3d
import cv2
import sys
from datetime import datetime
import sqlite3
# write data for IHI
is_first = True

def create_csv_IHI(path):
    with open(path, 'w') as f:
        csv_writer = csv.writer(f)
        #csv_header = ['frame', 'id','obj_type', 'height', 'width', 'length', 'x', 'y', 'z', 'theta', 'time', 'vx', 'vy', 'vz']
        header_list = []
        csv_header = ['scanner ID', 'target ID', 'sequence number', 'transmission_time year', 'transmission_time month', 'transmission_time day', 'transmission_time hour', 'transmission_time minute', 'transmission_time second', 'transmission_time millisecond', 'data identification', 'data length', 'data number', 'status', 
                      'type of object', 'data reliablity','object ID', 'detection_time year', 'detection_time month', 'detection_time day', 'detection_time hour', 'detection_time minute', 'detection_time second', 'detection_time millisecond', '1X', '1Y', '2X', '2Y', 'X speed', 'Y speed', 'length', 'width', 'height']
        csv_writer.writerow(csv_header)
    f.close()

def write_csv_IHI(path, data, frame_id):
    byte_length = np.uint16(0)
    
    Now_time =  datetime.now()
    
    header_list = [np.uint16(0x145D), np.uint16(0x1451), np.uint16(frame_id), 
                np.uint16(Now_time.year), np.uint16(Now_time.month), np.uint16(Now_time.day), np.uint16(Now_time.hour), np.uint16(Now_time.minute), np.uint16(Now_time.second), np.uint16(Now_time.microsecond / 1000), 
                'SD', np.uint16(0), np.uint16(data.shape[0]), np.uint16(0x02)]
    
    # for frame_data in data:
    #     for data_part in frame_data:     
    #         if isinstance(data_part, str):
    #             byte_length += len(data_part.encode())
    #         elif isinstance(data_part, (int, np.integer)):
    #             byte_length += data_part.itemsize
    # print(byte_length)
    
    byte_length = np.uint16(36 * data.shape[0])
    
    for header_part in header_list:
        if isinstance(header_part, str):
            byte_length += len(header_part.encode())
        elif isinstance(header_part, (int, np.integer)):
            byte_length += header_part.itemsize
    byte_length += byte_length.itemsize

    header_list[-3] = np.uint16(byte_length)

    header_part = []
    header_part.append(header_list)

    for i in range(data.shape[0] - 1):
        header_part.append(['', '', '', '', '', '', '', '', '', '', '', '', '', ''])
    
    with open(path, 'a+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(np.concatenate((np.array(header_part), data), axis=1))
    f.close()

def write_data_IHI(path, data, frame_id):
    if not os.path.exists(path):
        create_csv_IHI(path)
    write_csv_IHI(path, data, frame_id)


def create_image_db(path):
    conn = sqlite3.connect(path)
    print("Opened database successfully")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IMAGE
       (FRAME_ID INT,
       IMAGE BLOB);''')
    print("Image Table created successfully")
    conn.commit()
    conn.close()

def create_detection_db(path):
    conn = sqlite3.connect(path)
    print("Opened database successfully")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE DETECTION
       (FRAME_ID INT,
       OBJ_ID INT,
       OBJ_TYPE TEXT,
       X REAL,
       Y REAL,
       Z REAL,
       ROT_Y REAL,
       LENGTH REAL,
       WIDTH REAL,
       HEIGHT REAL);''')
    print("Detection Table created successfully")
    conn.commit()
    conn.close()

def write_img_to_sqlite3(path, cv_img, frame_id):
    # cv_img: cv2 image
    if not os.path.exists(path):
        create_image_db(path)
        create_detection_db(path)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cv_string = cv2.imencode('.jpg', cv_img)[1].tostring()
    img_blob = sqlite3.Binary(cv_string)
    cursor.execute("INSERT INTO IMAGE VALUES (?,?);",(frame_id, img_blob))
    conn.commit()
    conn.close()
    print('Successfully writing image !')

def write_detection_info_to_sqlite3(path, det_data, frame_id):
    # det_data: detection information, (n * 9) list
    if not os.path.exists(path):
        create_image_db(path)
        create_detection_db(path)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    for data in det_data:
        cursor.execute("INSERT INTO DETECTION VALUES (?,?,?,?,?,?,?,?,?,?);",(frame_id, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]))
    conn.commit()
    conn.close()

def read_detection_info_from_sqlite3(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    print("Opened database successfully")
    cursor = c.execute("SELECT frame_id, obj_id, obj_type, x, y, z, rot_y, length, width, height from DETECTION")
    for row in cursor:
        print("frame_id: {}; obj_id: {}; obj_type: {}; X: {}; Y: {}; Z: {}; \
               ROT_Y: {}; length: {}; width: {}; height: {}; ".format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]))
    conn.close()

def read_img_from_sqlite3(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    print("Opened database successfully")
    cursor = c.execute("SELECT frame_id, image from IMAGE")
    for row in cursor:
        print("ID = ", row[0])
        image = row[1]
        image = np.fromstring(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2.imshow('image', image)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    conn.close()

#####################################################
def create_csv(path):
    with open(path, 'w') as f:
        csv_writer = csv.writer(f)
        #csv_header = ['frame', 'id','obj_type', 'height', 'width', 'length', 'x', 'y', 'z', 'theta', 'time', 'vx', 'vy', 'vz']
        csv_header = ['type of object', 'data reliablity','object ID', 'year', 'month', 'day', 'hour', 'second', 'millisecond', '1X', '1Y', '2X', '2Y', 'X speed', 'Y speed', 'length', 'width', 'height']
        csv_writer.writerow(csv_header)
    f.close()

def write_csv(path, data):
    with open(path, 'a+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(data)
    f.close()

def write_data(path, data):
    if not os.path.exists(path):
        create_csv(path)
    write_csv(path, data)

# source_pts = [[7.79321814877791, 21.09639456843, -2.88016767000115],
#                   [24.8805527744841, 2.06188480084171, 1.52956318882623],
#                   [20.4169961194985, 13.5378718257327, 1.42957836609134],
#                     [14.7241213816304, 16.196740540685, -0.323806256969775],
#                     [18.5712427414707, 6.2056048004397, 1.01412774189971],
#                     [13.545773432216, 6.53963421616323, -0.558807134828419],
#                   [17.00333874, 3.614939343, 0.091669806]]
#     target_pts = [[42.5178917836268, 2.66918472144474, 5.64735857582081],
#                   [22.4191062949622, -11.2057634733186, 0.602206601187107],
#                   [33.2819290690458, -8.94275755192158, 4.25182140000439],
#                   [36.5115305930402, -3.62654205404541, 4.77303494933497],
#                   [26.7068431248853, -6.04124669081318, 2.6649967211306],
#                   [28.0742547676313, -0.68245229036983, 3.14217710909098],
#                   [17.66576474, 2.84477177, 0.226770463]]

def pointcloud_register():
    source_pts = [[7.83699059787848, 21.0302072348525, -2.85922748940691],
                  [24.8904038172081, 2.059175526679, 1.54390340400363],
                  [13.5406702361837, 6.52947350610136, -0.573725600269711]
                  ]
    target_pts = [[42.4598188264409, 2.67428702520595, 5.64052804976845],
                  [22.3981317579983, -11.1977512104975, 0.567081744677318],
                  [28.0578263811509, -0.690552564733743, 3.1457059477942]
                  ]
    src_pts_3d = np.array(source_pts)
    dst_pts_3d = np.array(target_pts)

    source_pcd_to_match = open3d.geometry.PointCloud()
    target_pcd_to_match = open3d.geometry.PointCloud()
    source_pcd_to_match.points = open3d.utility.Vector3dVector(src_pts_3d)
    target_pcd_to_match.points = open3d.utility.Vector3dVector(dst_pts_3d)

    corr_np = np.arange(0, len(src_pts_3d))
    corr_np = np.transpose(np.tile(corr_np, (2, 1)))
    correspondence = open3d.utility.Vector2iVector(corr_np)

    estimation=open3d.registration.TransformationEstimationPointToPoint()
    criteria=open3d.registration.RANSACConvergenceCriteria(max_iteration=1000, max_validation=1000)

    result = open3d.registration.registration_ransac_based_on_correspondence(source_pcd_to_match, target_pcd_to_match, correspondence, 1, 
                                                                    estimation_method=estimation, 
                                                                    ransac_n=3,
                                                                    criteria=criteria)
    print(result)
    print(result.transformation)
    rvec = cv2.Rodrigues(np.array(result.transformation[0:3, 0:3]))[0]
    tvec = np.array(result.transformation[0:3, 3]).reshape(3, 1)
    print(rvec)
    print(tvec)

def evaluation_fuction(source_pts, target_pts):
    num = 1
    # source_pts: array (n, 11 + 8*3)
    # frame_id, id, cls_name, h, w, l, x, y, z, theta, t

    source_size = np.float64(source_pts[:, 3])*np.float64(source_pts[:, 4])*np.float64(source_pts[:, 5])
    target_size = np.float64(target_pts[:, 3])*np.float64(target_pts[:, 4])*np.float64(target_pts[:, 5])

    
    # sort with last column: low to high
    source_pts = source_pts[np.argsort(source_size)]
    target_pts = target_pts[np.argsort(target_size)]

    print('True')
    result = open3d.registration.RegistrationResult()
    result.fitness = 0
    result.inlier_rmse = 9999.9

    source_len, target_len = source_pts.shape[0], target_pts.shape[0]
    for i in range(source_len-2):
        for j in range(source_len-1, i+1, -1):
            for t in range(i+1, j):
                #print(np.float64(source_pts[i, 6:9]).shape)
                source_pts_to_match = np.vstack((np.float64(source_pts[i, 6:9]), np.float64(source_pts[t, 6:9]), np.float64(source_pts[j, 6:9])))
                for m in range(target_len-2):
                    for n in range(target_len-1, m+1, -1):
                        for l in range(m+1, n):
                            target_pts_to_match = np.vstack((np.float64(target_pts[m, 6:9]), np.float64(target_pts[l, 6:9]), np.float64(target_pts[n, 6:9])))
                            new_result = icp_with_correspondence(source_pts_to_match, target_pts_to_match)
                            if new_result.fitness == 1 and new_result.inlier_rmse < result.inlier_rmse:
                                result = new_result
                                index = [source_pts[i, 1], source_pts[t, 1], source_pts[j, 1], target_pts[m, 1], target_pts[l, 1], target_pts[n, 1]]
                                np.savetxt('./result/'+str(num)+'.txt', result.transformation)
                                print(num)
                                num += 1
    
    if result.fitness == 1:
        print(result)
        print(result.transformation)
        # rvec = cv2.Rodrigues(np.array(result.transformation[0:3, 0:3]))[0]
        # tvec = np.array(result.transformation[0:3, 3]).reshape(3, 1)
        # print(rvec)
        # print(tvec)
        print(index)
        return result.transformation
    else:
        return None



def icp_with_correspondence(source_pts, target_pts):
    src_pts_3d = np.array(source_pts)
    dst_pts_3d = np.array(target_pts)

    source_pcd_to_match = open3d.geometry.PointCloud()
    target_pcd_to_match = open3d.geometry.PointCloud()
    source_pcd_to_match.points = open3d.utility.Vector3dVector(src_pts_3d)
    target_pcd_to_match.points = open3d.utility.Vector3dVector(dst_pts_3d)

    corr_np = np.arange(0, len(src_pts_3d))
    corr_np = np.transpose(np.tile(corr_np, (2, 1)))
    correspondence = open3d.utility.Vector2iVector(corr_np)

    estimation=open3d.registration.TransformationEstimationPointToPoint()
    criteria=open3d.registration.RANSACConvergenceCriteria(max_iteration=1000, max_validation=1000)

    result = open3d.registration.registration_ransac_based_on_correspondence(source_pcd_to_match, target_pcd_to_match, correspondence, 1, 
                                                                    estimation_method=estimation, 
                                                                    ransac_n=3,
                                                                    criteria=criteria)
    # print(result)
    # print(result.transformation)
    # rvec = cv2.Rodrigues(np.array(result.transformation[0:3, 0:3]))[0]
    # tvec = np.array(result.transformation[0:3, 3]).reshape(3, 1)
    # print(rvec)
    # print(tvec)
    return result