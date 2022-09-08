import numpy as np
import cv2 as cv

nptmp = np.random.random((1024, 1024)).astype(np.float32)

npMat1 = np.stack([nptmp, nptmp], axis=2)
npMat2 = npMat1

cuMat1 = cv.cuda_GpuMat()
cuMat2 = cv.cuda_GpuMat()
c = cuMat1.upload(npMat1)
m = cuMat2.upload(npMat2)

print(c, m)