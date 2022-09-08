import flow_vis
import pandas as pd
import numpy as np
import flowvid as fv

fv.rgb()

path = r"E:\ryu_pythonProject\2. Cardivu-A\1. data_analysis\test_data\Before\left/"
file = "[f]김환진_음주전_2021-05-17-181830-0000_M_L"
data = pd.read_csv(path+file+".csv")
data = data.drop(['index', 'pupil_size', 'Iris_X', 'Iris_Y', 'diam'], axis=1)
flow_uv = [3.5410074861416656, 2.1482092155982184, 2.6920847847116365, 0.8312735366116123, 0.0193322162428396, 2.2201083941147957, 1.3158034381356474, 2.06629325016655, 2.2064394684696427, 1.2981064497917287, 2.631225154085539, 3.2791073335262615, 3.452120174614043, 3.055634349288645, 3.1663256533676885, 1.914646697706959, 2.4246696980426568, 1.8269362780703517, 1.551657620971485, 2.038492676275501, 1.425143897653134, 1.9728155581541227, 1.8451798908813928, 1.9114846707769138, 2.076674247668903, 1.5033735172007288, 0.0, 0.43662851812121906, 0.0, 30.0, -0.43662851812121906, -0.43662851812121906, -0.1309885554363657, 4.335264314370125, 0.07788579799575246, 0.41810704509236213, 5.0, 10.0, -0.34022124709660967, -0.41810704509236213, -0.03791641460944859, 1.621255967590171, 1.3431891598667562, 0.1578752333997238, 80.0, 10.0, 1.1853139264670325, 1.3431891598667562, 1.0587638045534327, 3.4429015398710545, 2.146414202759964, 0.0, 65.0, 0.0, 2.146414202759964, 2.146414202759964, 1.3951692317939766, 1.749703887174463, 0.5263094871311935, 0.1118664567032249, 20.0, 10.0, 0.4144430304279686, 0.5263094871311935, 0.09407525175591622, 4.090700473167727, 0.904491663324256, 0.01897083123376997, 20.0, 5.0, 0.885520832090486, 0.904491663324256, 0.17994979110316273, 2.20767440320529, 0.0, 1.7364224346932668, 0.0, 80.0, -1.7364224346932668, -1.7364224346932668, -1.3891379477546135, 4.932202943279139, 0.31636728823926674, 1.6995052561391837, 5.0, 40.0, -1.383137967899917, -1.6995052561391837, -0.6639837380437101, 0.8920445668444547, 0.2898618075989998, 0.10153214461108283, 20.0, 20.0, 0.18832966298791698, 0.2898618075989998, 0.0376659325975834, 3.8000578102028446, 0.49567018897053483, 1.0213237814066942, 20.0, 35.0, -0.5256535924361594, -1.0213237814066942, -0.258329285698236, 1.3359226244775453, 0.5784560751235454, 0.17344217826245334, 30.0, 20.0, 0.405013896861092, 0.5784560751235454, 0.13884838688457293, 4.271228278927911, 0.3618796290536125, 1.318054657818324, 15.0, 35.0, -0.9561750287647116, -1.318054657818324, -0.40703718587837157]
flow_uv = np.reshape(flow_uv, (11, 11, 1))
print(np.shape(flow_uv))

flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)

