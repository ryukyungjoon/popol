import numpy as np
import param

class Save:
    def __init__(self):
        pass

    def frame_append(self, OF_mag, OF_ang, LR):
        if LR == "L":
            param.MAG_L.append(OF_mag)
            param.ANG_L.append(OF_ang)
            # param.MOT_L.append(OF_mot)

        elif LR == "R":
            param.MAG_R.append(OF_mag)
            param.ANG_R.append(OF_ang)
            # param.MOT_R.append(OF_mot)

    def save_npz(self, LR):
        if LR == "L":
            print(f'Save to {param.save_path}{param.file_name}{param.mask_loc}_{LR}.npz')
            np.savez(param.save_path + param.file_name + param.mask_loc + "_" + LR + ".npz",
                     mag=param.MAG_L, ang=param.ANG_L)

        elif LR == "R":
            print(f'Save to {param.save_path}{param.file_name}{param.mask_loc}_{LR}.npz')
            np.savez(param.save_path + param.file_name + param.mask_loc + "_" + LR + ".npz",
                     mag=param.MAG_R, ang=param.ANG_R)

    def save_video(self):
        if param.video_save:
            for n in range(len(param.sr_L)):
                param.out_video_L.write(param.sr_L[n])
                param.out_video_R.write(param.sr_R[n])

