import cv2
import flow_vis as fv

path = r"E:\ryu_pythonProject\2. Cardivu-A\re_extraction\IR_IRIS image\Before_alcohol\SR/"


## 53 & 148
## 87 & 233
prev = cv2.imread(path + "(sr_img)0_00087.jpg")
nexts = cv2.imread(path + "(sr_img)0_00087.jpg")

# prev = cv2.medianBlur(prev, 11)
# nexts = cv2.medianBlur(nexts, 11)
#
# prev = cv2.Canny(prev, 0, 50, L2gradient=True)
# nexts = cv2.Canny(nexts, 0, 50, L2gradient=True)
#
# prev = cv2.medianBlur(prev, 3)
# nexts = cv2.medianBlur(nexts, 3)
#
# prev = cv2.resize(prev, (300, 300))
# nexts = cv2.resize(nexts, (300, 300))
# cv2.imshow('prev', prev)
# cv2.imshow('nexts', nexts)

flow = cv2.optflow.createOptFlow_SparseToDense()
flow = flow.calc(prev, nexts, None)
flow_color = fv.flow_to_color(flow, convert_to_bgr=False)
flow_color = cv2.resize(flow_color, (300, 300))
frame3 = cv2.imshow('frame3', flow_color)

flow2 = cv2.calcOpticalFlowFarneback(prev, nexts, None, 0.5, 2, 14, 2, 5, 1.2, 0)
flow_color2 = fv.flow_to_color(flow2, convert_to_bgr=False)
frame2 = cv2.imshow('frame2', flow_color2)

k = cv2.waitKey(30) & 0xff

if k == 27:
    exit(0)
elif k == ord('s'):
    cv2.imwrite('opticalfb.png', frame3)
    cv2.imwrite('opticalvis.png', flow_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
