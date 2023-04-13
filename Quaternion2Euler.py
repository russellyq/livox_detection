from sensor_msgs.msg import Imu
import rospy
import numpy as np
import math

class Converter(object):
    def __init__(self) -> None:
        super().__init__()
        rospy.init_node('Quaternion2Euler', anonymous=True)
        self.imu_sub = rospy.Subscriber('/livox/imu/data', Imu, self.callback)

    def callback(self, imu_msg):
        q = imu_msg.orientation
        # sinr_cosp = float( 2 * (q.w * q.x + q.y * q.z))
        # cosr_cosp = float(1 - 2 * (q.x * q.x + q.y * q.y))
        # roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # sinp = float(2 * (q.w * q.y - q.z * q.x))
        # if (abs(sinp) >= 1):
        #     pitch = float(math.copysign(np.pi / 2, sinp))
        # else:
        #     pitch = np.arcsin(sinp)
        
        # siny_cosp = float(2 * (q.w * q.z + q.x * q.y))
        # cosy_cosp = float(1 - 2 * (q.y * q.y + q.z * q.z))
        # yaw = np.arctan2(siny_cosp, cosy_cosp)

        # print("roll; pitch; yaw; ", roll, pitch, yaw)
        
        t0 = +2.0 * (q.w * q.x + q.y * q.z)
        t1 = +1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (q.w * q.y - q.z * q.x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        Z = math.degrees(math.atan2(t3, t4))
        print(("roll; pitch; yaw", X, Y, Z))
        
        # w = q.w
        # x = q.x
        # y = q.y
        # z = q.z

        # r = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        # p = math.asin(2*(w*y-z*x))
        # y = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))

        # angleR = r*180/math.pi
        # angleP = p*180/math.pi
        # angleY = y*180/math.pi
        # print(("roll; pitch; yaw", angleR, angleP, angleY))

if __name__=="__main__":
    converter = Converter()
    rospy.spin()
