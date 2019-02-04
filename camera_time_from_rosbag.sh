#!/bin/sh
rm -r ~/camera_header.csv
rm -r ~/camera_secs.csv
rm -r ~/camera_nsecs.csv

echo "start"
rostopic echo -b ~/something_test/rosbag/kinect_color_depth_hd_04.bag -p /kinect2/hd/image_depth_rect/header > ~/camera_header.csv
echo "finish header"
python ~/something_test/python/finish_mp3.py


rostopic echo -b ~/something_test/rosbag/kinect_color_depth_hd_04.bag -p /kinect2/hd/image_depth_rect/header/stamp/secs > ~/camera_secs.csv
echo "finish secs"
python ~/something_test/python/finish_mp3.py

rostopic echo -b ~/something_test/rosbag/kinect_color_depth_hd_04.bag -p /kinect2/hd/image_depth_rect/header/stamp/nsecs > ~/camera_nsecs.csv
echo "finish nsecs"
python ~/something_test/python/finish_mp3.py
