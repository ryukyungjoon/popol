import numpy as np
import cv2
import cv2.optflow as opt
import flow_vis as fv

path = r"E:\ryu_pythonProject\2. Cardivu-A\re_extraction\IR_IRIS image\After_alcohol\SR/"
# cap = cv2.VideoCapture(path + '김환진_음주후_2021-06-09-211918-0000_L.avi')
cap = cv2.VideoCapture(path + 'bandicam 2021-07-26 11-58-48-373.avi')

ret, frame1 = cap.read()
prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    frame2 = cv2.medianBlur(frame2, 15)
    nexts = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev, nexts, None, 0.3, 2, 14, 2, 5, 1.2, 0)
    flo2 = opt.createOptFlow_SparseToDense()
    flow2 = flo2.calc(prev, nexts, None)

    flow_color = fv.flow_to_color(flow, convert_to_bgr=False)
    flow2_color = fv.flow_to_color(flow2, convert_to_bgr=False)
    cv2.imshow('frame3', flow_color)
    cv2.imshow('frame4', flow2_color)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2', rgb)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)
        cv2.imwrite('opticalvis.png', flow_color)
    prev = nexts

cap.release()
cv2.destroyAllWindows()