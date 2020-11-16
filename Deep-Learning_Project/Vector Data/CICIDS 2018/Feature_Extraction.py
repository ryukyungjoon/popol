from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier, plot_importance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Feature_Extraction:
    def __init__(self):
        pass

    def feature_extraction_1(self, x, y, features):
        RF = RandomForestClassifier(random_state=0, n_estimators=10, n_jobs=-1)
        RFmodel = RF.fit(x, y)
        Importances = RFmodel.feature_importances_
        print(Importances)
        std = np.std([tree.feature_importances_ for tree in RFmodel.estimators_],
                     axis=0)
        indices = np.argsort(Importances)[::-1]

        print('Feature Ranking:')
        for f in range(x.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], Importances[indices[f]]))

        # Drawing Feature Importance Graph
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(x.shape[1]), Importances[indices],
                color="b", yerr=std[indices])
        plt.xticks(range(x.shape[1]), indices, rotation=90)
        plt.xlim([-1, x.shape[1]])
        plt.show()

        xx = pd.DataFrame(x)

        # Feature Importance
        print("indices:", indices)
        for k in range(len(Importances)):
            if Importances[indices[k]] < 0.001:
                f = indices[k]
                xx = xx.drop(features[f], axis=1)

        # Feature Correlation
        corr = xx.rank().corr(method="pearson")
        corr = pd.DataFrame(corr)
        columns = np.full((corr.shape[0],), True, dtype=bool)

        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= 0.95:
                    if columns[j]:
                        columns[j] = False

        selected_columns = xx.columns[columns]
        xx = xx[selected_columns]
        print("XX: {}".format(xx))
        fe_data = pd.concat([xx, y], axis=1)
        remain_features = fe_data.head(0)
        remain_features_num = len(remain_features)
        fe_data = pd.DataFrame(fe_data)
        # fe_data.to_csv("dataset/Feature_Selection/["+str(remain_features_num)+"] RF+Pearson",
        #                header=list(remain_features), index=False)

        return fe_data

    def feature_extraction_2(self, x, y, features):
        xgb = XGBClassifier()
        xgb.fit(x, y)

        # Feature Importance
        Importances = xgb.feature_importances_
        print(Importances)

        new_df = pd.DataFrame(x)

        # plot feature importance
        plot_importance(xgb)
        plt.show()

        for k in range(len(Importances)):
            if Importances[k]<0.001:
                new_df = new_df.drop(k, axis=1)

        print(new_df)