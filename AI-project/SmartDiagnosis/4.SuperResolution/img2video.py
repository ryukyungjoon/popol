import cv2
import os


srimg_path = r"E:\ryu_pythonProject\2. Cardivu-A\re_extraction\IR_IRIS image\After_alcohol\SR\image\1/"
save = r"E:\ryu_pythonProject\2. Cardivu-A\re_extraction\IR_IRIS image\After_alcohol\SR/"
img_list = os.listdir(srimg_path)
print(img_list)
srimg = []
for i in range(len(img_list)):
    img = cv2.imread(srimg_path + img_list[i], cv2.IMREAD_COLOR)
    srimg.append(img)
    print(i)

fps = int(len(img_list))/180
cap = cv2.VideoWriter(save+"(sr)"+"0.avi", cv2.VideoWriter_fourcc(*'DIVX'),
                      fps, (180, 180))

for n in range(len(srimg)):
    cap.write(srimg[n])
cap.release()
