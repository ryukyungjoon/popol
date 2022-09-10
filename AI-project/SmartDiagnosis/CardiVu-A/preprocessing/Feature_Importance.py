import pandas as pd

from sklearn.ensemble import RandomForestClassifier

data_path = r"E:\ryu_pythonProject\2. Cardivu-A\1. data_analysis\test_data/"

BL_path = data_path + "Before/left/"
BR_path = data_path + "Before/right/"
AL_path = data_path + "After/left/"
AR_path = data_path + "After/right/"

Before_L = ["[f]김환진_음주전_2021-05-17-181830-0000_M_L",
            "[f]김환진_음주전_2021-06-09-183322-0000_M_L",
            "[f]문예성_음주전_2021-05-17-181137-0000_M_L",
            "[f]문예성_음주전_2021-06-09-182014-0000_M_L",
            "[f]이동원_음주전_2021-05-17-182348-0000_M_L",
            "[f]이동원_음주전_2021-06-09-182703-0000_M_L",
            "[f]신재민_음주전_2021-06-09-184016-0000_M_L"]

Before_R = ["[f]김환진_음주전_2021-05-17-181830-0000_M_R",
            "[f]김환진_음주전_2021-06-09-183322-0000_M_R",
            "[f]문예성_음주전_2021-05-17-181137-0000_M_R",
            "[f]문예성_음주전_2021-06-09-182014-0000_M_R",
            "[f]이동원_음주전_2021-05-17-182348-0000_M_R",
            "[f]이동원_음주전_2021-06-09-182703-0000_M_R",
            "[f]신재민_음주전_2021-06-09-184016-0000_M_R"]

After_L = ["[f]김환진_음주후_2021-06-09-211918-0000_M_L",
           "[f]김환진_음주후2_2021-05-17-203642-0000_M_L",
           "[f]문예성_음주후_2021-06-09-213245-0000_M_L",
           "[f]문예성_음주후2_2021-05-17-203150-0000_M_L",
           "[f]이동원_음주후_2021-06-09-212501-0000_M_L",
           "[f]이동원_음주후2_2021-05-17-202504-0000_M_L",
           "[f]신재민_음주후_2021-06-09-213742-0000_M_L"]

After_R = ["[f]김환진_음주후_2021-06-09-211918-0000_M_R",
           "[f]김환진_음주후2_2021-05-17-203642-0000_M_R",
           "[f]문예성_음주후_2021-06-09-213245-0000_M_R",
           "[f]문예성_음주후2_2021-05-17-203150-0000_M_R",
           "[f]이동원_음주후_2021-06-09-212501-0000_M_R",
           "[f]이동원_음주후2_2021-05-17-202504-0000_M_R",
           "[f]신재민_음주후_2021-06-09-213742-0000_M_R"]


before = pd.read_csv(BL_path + Before_L[0] + ".csv")
after = pd.read_csv(AL_path + After_L[0] + ".csv")

drop_feature = ['index', 'pupil_size', 'Iris_X', 'Iris_Y', 'diam']
before = before.drop(drop_feature, axis=1)
after = after.drop(drop_feature, axis=1)

before = before.assign(label=0)
after = after.assign(label=1)
train = pd.concat([before, after], axis=0, ignore_index=True)
train_y, train_x = train['label'], train.drop('label', axis=1)

rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf_model.fit(train_x, train_y)
importance = rf_model.feature_importances_
print(importance)
importance = pd.DataFrame(importance)
importance.T.to_csv("./importance.csv")

