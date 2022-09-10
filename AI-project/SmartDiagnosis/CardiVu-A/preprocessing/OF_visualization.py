import flow_vis as fv
import numpy as np
import cv2
from matplotlib import pyplot as plt

path = r"E:\ryu_pythonProject\2. Cardivu-A\202106\김환진_음주전_2021-06-09-183322-0000/"
file = "10김환진_음주전_2021-06-09-183322-0000.avi"

vidcap = cv2.VideoCapture(path + file)
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))

flow_uv = []
while vidcap.isOpened():
    ret, frame = vidcap.read()
    if not ret:
        break
    flow_uv.append(frame)
    flow = []
    if len(flow_uv) == 2:
        flow_uv[0] = cv2.cvtColor(flow_uv[0], cv2.COLOR_BGR2GRAY)
        flow_uv[1] = cv2.cvtColor(flow_uv[1], cv2.COLOR_BGR2GRAY)
        uflowF = cv2.calcOpticalFlowFarneback(flow_uv[0], flow_uv[1], None, 0.5, 3, 6, 6, 5, 1.2, 0)
        flow_color = fv.flow_to_color(uflowF, convert_to_bgr=False)
        plt.imshow(flow_color)
        plt.show()
        exit(0)
