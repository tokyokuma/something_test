#!/usr/bin/env python
import cv2
import math
import numpy as np

img = cv2.imread("/home/kuma/something_test/dataset_izunuma/images/20180808/kinect_color_depth_qhd_01.bag/qhd_depth/frame008106.png", 2)
#cv2.imshow("img", img)
h, w = img.shape

true_bamboo_UTM_x = 507986.129448
true_bamboo_UTM_y = 4285783.0981

true_iron1_UTM_x = 507986.129448
true_iron1_UTM_y = 4285783.0981

#boat position UTM coodinate
UTM_x = 507987.791318454
UTM_y = 4285787.04538982


#bamboo param
bamboo_depth = 1033
bamboo_u = 226
bamboo_v = 257

#iron1 param
iron1_depth = 1847
iron1_u = 266
iron1_v = 206

#commpass heading
compass_deg = 196.87
compass_rad = math.radians(compass_deg)

#The inclination of the camera body (horizontal downward angle) degree to radians
camera_deg = 15
camera_rad = math.radians(camera_deg)

#camera instrict param
fx=521.0852017415114
fy=521.7842912470452
cx=481.64180763282405
cy=277.6362107374241

cam_instrict_mat = np.mat([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])

#Camera coordinates in UTM coordinate system
#XY coordinates of the camera based on the GPS receiver
camera_x_from_GPS = -0.2975
camera_y_from_GPS = 1.55
camera_xy_from_GPS= np.array([camera_x_from_GPS,camera_y_from_GPS])
compass_rot_mat = np.array([[math.cos(compass_rad), math.sin(compass_rad)],
                               [(-1)*math.sin(compass_rad), math.cos(compass_rad)]])


camera_xy_from_GPS_rot = np.dot(compass_rot_mat, camera_xy_from_GPS.T)

print "camera_xy_from_GPS_rot: "
print camera_xy_from_GPS_rot

camera_UTM_x = UTM_x + camera_xy_from_GPS_rot.item(0)
camera_UTM_y = UTM_y + camera_xy_from_GPS_rot.item(1)

print "camera_UTM_x: {} ".format(camera_UTM_x)
print "camera_UTM_y: {}" .format(camera_UTM_y)

#############################ok##########################################


#Depth unit from mm to m conversion
d = float(bamboo_depth) / 1000

#Image coordinate system (x,y) to Camera coordinate system (X,Y,Z)
Z3D = d
X3D = (bamboo_u - cx) * Z3D / fx
Y3D = -((bamboo_v - cy) * Z3D / fy)
print "X3D {}:" .format(X3D)
print "Y3D {}:" .format(Y3D)
print "Z3D {}:" .format(Z3D)

#Convert 3D XYZ to 2D XZ
#Z axis in 2D is oriented in the direction of travel and horizontal
X2D = X3D
Z2D = Z3D * math.cos(camera_rad) + Y3D * math.sin(camera_rad)
ZX_2D = np.array([X2D,Z2D])
print "Z2D {}:" .format(Z2D)
print "X2D {}:" .formateo(X2D)

camera_xy_from_camera_rot = np.dot(compass_rot_mat, ZX_2D.T)

print "camera_xy_from_camera_rot: "
print camera_xy_from_camera_rot
print "camera_x_from_camera_rot: {}" .format(camera_xy_from_camera_rot.item(0))
print "camera_y_from_camera_rot: {}" .format(camera_xy_from_camera_rot.item(1))



estimated_bamboo_UTM_x = camera_UTM_x + camera_xy_from_camera_rot.item(0)
estimated_bamboo_UTM_y = camera_UTM_y + camera_xy_from_camera_rot.item(1)

print "estimated_bamboo_UTM_x: {}" .format(estimated_bamboo_UTM_x)
print "estimated_bamboo_UTM_y: {}" .format(estimated_bamboo_UTM_y)


diff_distance_from_true_bamboo_UTM_x = true_bamboo_UTM_x - estimated_bamboo_UTM_x
diff_distance_from_true_bamboo_UTM_y = true_bamboo_UTM_y - estimated_bamboo_UTM_y

print "diff_distance_from_true_bamboo_position_x: {}" .format(diff_distance_from_true_bamboo_UTM_x)
print "diff_distance_from_true_bamboo_position_y: {}" .format(diff_distance_from_true_bamboo_UTM_y)

#diff_distance_from_true_iron1_UTM_x = true_iron1_UTM_x - estimated_iron1_UTM_x
#diff_distance_from_true_iron1_UTM_y = true_iron1_UTM_y - estimated_iron1_UTM_y

#print "diff_distance_from_true_iron1_position_x: {}" .format(diff_distance_from_true_iron1_UTM_x)
#print "diff_distance_from_true_iron1_position_y: {}" .format(diff_distance_from_true_iron1_UTM_y)


key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
