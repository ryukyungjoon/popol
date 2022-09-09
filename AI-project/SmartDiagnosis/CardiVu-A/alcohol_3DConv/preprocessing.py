import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

data_path = r"E:\ryu_pythonProject\2. Cardivu-A\1. data_analysis\test_data/"

FILELIST = ["[f]문예성_음주전_2021-05-17-181137-0000_M_L",
            "[f]문예성_음주전_2021-05-17-181137-0000_M_R",
            "[f]문예성_음주후2_2021-05-17-203150-0000_M_L",
            "[f]문예성_음주후2_2021-05-17-203150-0000_M_R"]

Before_L = pd.read_csv(data_path + FILELIST[0] + ".csv")
Before_R = pd.read_csv(data_path + FILELIST[1] + ".csv")
After_L = pd.read_csv(data_path + FILELIST[2] + ".csv")
After_R = pd.read_csv(data_path + FILELIST[3] + ".csv")

drop_feature = ['index', 'pupil_size', 'Iris_X', 'Iris_Y', 'diam']

Before_L = Before_L.drop(drop_feature, axis=1)
Before_R = Before_R.drop(drop_feature, axis=1)
After_L = After_L.drop(drop_feature, axis=1)
After_R = After_R.drop(drop_feature, axis=1)

col = Before_L.columns

mms1 = MinMaxScaler(feature_range=(0, 1))
mms1.fit(Before_L)
Before_L = mms1.transform(Before_L)
After_L = mms1.transform(After_L)

mms2 = MinMaxScaler(feature_range=(0, 1))
mms2.fit(Before_R)
Before_R = mms2.transform(Before_R)
After_R = mms2.transform(After_R)

Before_L = pd.DataFrame(Before_L, columns=col)
Before_R = pd.DataFrame(Before_R, columns=col)
After_L = pd.DataFrame(After_L, columns=col)
After_R = pd.DataFrame(After_R, columns=col)

Before_L = Before_L.assign(label=0)
After_L = After_L.assign(label=1)

Before_R = Before_R.assign(label=0)
After_R = After_R.assign(label=1)

L_set = pd.concat([Before_L, After_L], axis=0, ignore_index=True)
R_set = pd.concat([Before_R, After_R], axis=0, ignore_index=True)

data = pd.concat([L_set, R_set], axis=0, ignore_index=True)


data_y, data_x = data['label'], data.drop('label', axis=1)
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# import numpy as np
# train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2)
#
# dt = DecisionTreeClassifier()
# dt.fit(train_x, train_y)
# cv_score = cross_val_score(dt, train_x, train_y, cv=10)
# print(cv_score)
#
# pred = dt.predict(test_x)
# cm = confusion_matrix(test_y, pred)
# print(cm)
# report = classification_report(test_y, pred)
# print(str(report))


##



## 3D convolution Layer
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # if like me you do not have a lot of memory in your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # then these two lines force keras to use your CPU
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from keras.utils import to_categorical
import numpy as np
import cv2



# Split the data into training features/targets
X_train = data[col][:]
targets_train = data["label"][:]
# new_arr = np.array(X_train)
# new_arr = np.resize(new_arr, new_shape=(16, 16, 16, 3))

##
train_x = []
for i in range(len(X_train['OF_1'])):
    nd = np.array(X_train.iloc[i, :])
    nd = np.resize(nd, new_shape=(16, 16, 3))
    if i == 0: print(nd)
    train_x.append(nd)


sample_shape = np.shape(train_x)
print(sample_shape)
# Convert target vectors to categorical targets
targets_train = to_categorical(targets_train).astype(np.integer)

# Create the model
model = Sequential()
model.add(
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
model.summary()
# Fit data to model
history = model.fit(train_x, targets_train,
                    batch_size=128,
                    epochs=40,
                    verbose=1,
                    validation_split=0.3)


