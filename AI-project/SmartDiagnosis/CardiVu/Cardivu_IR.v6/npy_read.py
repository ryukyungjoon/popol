import numpy as np

data = np.load(r'E:\ryu_pythonProject\0. Data\HRV\IR\Save_Results/[Timesync]20210730-104303mask1L.npz')
mag = data["mag"]
ang = data["ang"]

print(mag, ang)
print(len(mag), len(ang))
