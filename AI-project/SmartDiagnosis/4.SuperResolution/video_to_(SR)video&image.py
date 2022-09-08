"""

IRIS 90x90 Video to 180x180 [Video & Image]
"""

import numpy as np
import cv2
import os
from ISR.models import RDN

path = r'D:\ryu_pythonProject\2. Cardivu-A\202106/'
save_vid = r'D:\ryu_pythonProject\9. FlowNet\light reflection test IRIS\SR/'
save_img = r'D:\ryu_pythonProject\9. FlowNet\light reflection test IRIS\SR\image/'
file_list = [_ for _ in os.listdir(path) if _.endswith(r".avi")]
print(file_list)

## Weights Load
rdn = RDN(weights='psnr-small')
fps = 100
for i in range(0, len(file_list)):
    sr_video = []
    cap = cv2.VideoCapture(path + file_list[i])
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print('Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    s = 0

    while True:
        ret, frame = cap.read()
        sr_width = frame_width * 2
        sr_height = frame_height * 2

        if not ret:
            print(np.shape(sr_video))
            out = cv2.VideoWriter(save_vid + "(SR)" + file_list[i],
                                  cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'),
                                  fps,
                                  (sr_width, sr_height))
            for n in range(len(sr_video)):
                out.write(sr_video[n])
            out.release()
            cap.release()
            break
        sr_img = rdn.predict(frame)
        frame = cv2.resize(frame, (sr_width, sr_height))
        cv2.imshow('lr_img', frame)
        cv2.imshow('sr_img', sr_img)
        ## Save SR image
        cv2.imwrite(save_img + str(i).rjust(2, '0') + "/(sr_img)" + str(i) + "_" + str(s).rjust(5, '0') + ".jpg", sr_img)
        s += 1
        print(s)
        # sr_video.append(sr_img)
