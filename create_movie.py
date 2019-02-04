#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

fmt = cv2.VideoWriter_fourcc(*'XVID')
fps = 6.0
size = (768*2, 432*2)
writer = cv2.VideoWriter('/home/kuma/Documents/修士論文/動画/speech.avi', fmt, fps, size)


cap_01 = cv2.VideoCapture('/home/kuma/Documents/修士論文/動画/outtest.avi')
cap_02 = cv2.VideoCapture('/home/kuma/Documents/修士論文/動画/qhd01.avi')


while(cap_01.isOpened()):
    ret_01, frame_01 = cap_01.read()
    ret_02, frame_02 = cap_02.read()

    print frame_01.shape
    print frame_02.shape

    im_h = cv2.hconcat([frame_01, frame_02])
    writer.write(im_h)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
