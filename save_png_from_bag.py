#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("image_topic", help="Image topic.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_number", help="image_number.")


    args = parser.parse_args()

    print "Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          args.image_topic, args.output_dir)

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = int(args.image_number)
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        if count % 10 == 0:
            file_num = count / 10
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            #cv2.imwrite(os.path.join(args.output_dir, "frame%06i.png" % file_num), cv_img)
            cv2.imwrite(os.path.join(args.output_dir, "%08i.png" % file_num), cv_img)
            print "Wrote image %i" % file_num
        else:
            pass

        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()
