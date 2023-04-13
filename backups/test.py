# # # import preprocess
# # # import numpy as np


# # # x=[1.1,2.2,3.3, 1.1,2.2,3.3]
# # # y=np.identity(3)
# # # z=np.ones((3,1))
# # # a = preprocess.rotate_translate_pcd(x,y,z)
# # # print(a.reshape(-1, 3))

# # import	rospy
# # from	sensor_msgs.msg	import	Image
# # from sensor_msgs.msg import CameraInfo
# # import	cv2, cv_bridge
# # import numpy
# # image = cv2.imread('/home/robot/Downloads/livox-aerial-gound.jpg')
# # # class gray2bgr:
# # #     def __init__(self) -> None:
# # #         self.bridge = cv_bridge.CvBridge()
# # #         self.img_sub = rospy.Subscriber('/rgb_cam/gray_image',Image, self.image_callback)
# # #         self.img_pub = rospy.Publisher('/rgb_cam/image_raw', Image, queue_size=1)
    
# # #     def image_callback(self, img_msg):
# # #         global image
# # #         # img = self.bridge.imgmsg_to_cv2(img_msg, 'mono8')
        
# # #         # color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# # #         # print(color_img.shape)
# # #         # cv2.imshow('main', color_img)
# # #         # cv2.waitKey(3)
# # #         # color_img_msg = self.bridge.cv2_to_imgmsg(color_img, 'bgr8')
# # #         # color_img_msg.header = img_msg.header
# # #         # self.img_pub.publish(color_img_msg)
# # #         color_img_msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
# # #         color_img_msg.header = img_msg.header
# # #         self.img_pub.publish(color_img_msg)


# # # rospy.init_node('gray2color')
# # # gray2bgrnode = gray2bgr()
# # # rospy.spin()
# # import rospy
# # from std_msgs.msg import String
# # bridge = cv_bridge.CvBridge()
# # def talker():
# #     pub = rospy.Publisher('/rgb_cam/image_raw', Image, queue_size=1)
# #     rospy.init_node('talker', anonymous=True)
# #     rate = rospy.Rate(10) # 10hz
# #     while not rospy.is_shutdown():
# #         hello_str = "hello world %s" % rospy.get_time()
# #         rospy.loginfo(hello_str)
# #         color_img_msg = bridge.cv2_to_imgmsg(image, 'bgr8')
# #         pub.publish(color_img_msg)
# #         rate.sleep()

# # if __name__ == '__main__':
# #     try:
# #         talker()
# #     except rospy.ROSInterruptException:
# #         pass

# # from utils.tools import read_img_from_sqlite3, read_detection_info_from_sqlite3
# # read_img_from_sqlite3('test.db')
# # read_detection_info_from_sqlite3('test.db')

# #import the necessary packages
# import numpy as np
# import cv2

# def order_points(pts):
# 	# 初始化坐标点列表，使其遵循一定的顺序：列表第一个值为左上角坐标，列表第二个点是右上角坐标，第三个值为右下角坐标，第四个值为左下角坐标。
# 	rect = np.zeros((4, 2), dtype = "float32")

# 	# 左上角的总和最小，右下角的总和最大
# 	s = pts.sum(axis = 1)
# 	rect[0] = pts[np.argmin(s)]
# 	rect[2] = pts[np.argmax(s)]

# 	# 现在计算点之间的差异：右上角的差异最小，左下角的差异最大。
# 	diff = np.diff(pts, axis = 1)
# 	rect[1] = pts[np.argmin(diff)]
# 	rect[3] = pts[np.argmax(diff)]
# 	# 返回有序坐标
# 	return rect

# def four_point_transform(image, pts):
# 	# obtain a consistent order of the points and unpack them
# 	# individually
# 	rect = order_points(pts)
# 	(tl, tr, br, bl) = rect

# 	# compute the width of the new image, which will be the
# 	# maximum distance between bottom-right and bottom-left
# 	# x-coordiates or the top-right and top-left x-coordinates
# 	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
# 	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
# 	maxWidth = max(int(widthA), int(widthB))

# 	# compute the height of the new image, which will be the
# 	# maximum distance between the top-right and bottom-right
# 	# y-coordinates or the top-left and bottom-left y-coordinates
# 	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
# 	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
# 	maxHeight = max(int(heightA), int(heightB))

# 	# now that we have the dimensions of the new image, construct
# 	# the set of destination points to obtain a "birds eye view",
# 	# (i.e. top-down view) of the image, again specifying points
# 	# in the top-left, top-right, bottom-right, and bottom-left
# 	# order
# 	dst = np.array([
# 		[0, 0],
# 		[maxWidth - 1, 0],
# 		[maxWidth - 1, maxHeight - 1],
# 		[0, maxHeight - 1]], dtype = "float32")

# 	# compute the perspective transform matrix and then apply it
# 	M = cv2.getPerspectiveTransform(rect, dst)
# 	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# 	# return the warped image
# 	return warped

# # USAGE
# # python transform_example.py --image images/example_01.png --coords "[(73, 239), (356, 117), (475, 265), (187, 443)]"
# # python transform_example.py --image images/example_02.png --coords "[(101, 185), (393, 151), (479, 323), (187, 441)]"
# # python transform_example.py --image images/example_03.png --coords "[(63, 242), (291, 110), (361, 252), (78, 386)]"
# def mouse(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, y)
# import numpy as np
# import argparse
# import cv2
# # image = cv2.imread('test.png')
# # cv2.namedWindow('src')
# # cv2.imshow('src', image)
# # cv2.setMouseCallback("src", mouse)
# # k = cv2.waitKey(0)
# # if k == 27:
# #     cv2.destroyAllWindows()
# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--coords",
# 	help = "comma seperated list of source points")
# args = vars(ap.parse_args())

# # load the image and grab the source coordinates (i.e. the list of
# # of (x, y) points)
# # NOTE: using the 'eval' function is bad form, but for this example
# # let's just roll with it -- in future posts I'll show you how to
# # automatically determine the coordinates without pre-supplying them
# image = cv2.imread('test.png')
# pts = np.array(eval(args["coords"]), dtype = "float32")

# # apply the four point tranform to obtain a "birds eye view" of
# # the image
# warped = four_point_transform(image, pts)

# # show the original and warped images

# cv2.imshow("Warped", warped)
# cv2.imwrite('road.png', warped)
# '''执行代码：python transform_example.py --image images/example_03.png --coords "[(63, 242), (291, 110), (361, 252), (78, 386)]"
# '''
# k = cv2.waitKey(0)
# if k == 27:
#     cv2.destroyAllWindows()
import tensorflow as tf
import os
 
model_dir = './model_pb/'
model_name = 'saved_model.pb'
 
def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
 
create_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
with open('node.txt', 'w+') as f:
	for tensor_name in tensor_name_list:
		print(tensor_name)
		f.write(tensor_name)
		f.write('\n')
f.close()
