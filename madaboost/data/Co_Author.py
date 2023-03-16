import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data.common.change_rate_data import change_rate_data


def load_data(test_size, new_rate):
    data = pd.read_csv('D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/LEC7/madaboost_23102022/madaboost/data/datasets/Co_Author_100_500_1.csv')
    diag_map = {-1: 1.0, 1: -1.0}
    data['Label'] = data['Label'].map(diag_map)
    X = data.values[:, 0:-1]
    y = data.values[:, 7]
    # X = data.drop(['Lable class'], axis=1).values
    # y = data['Label class'].values
    X, y = change_rate_data(X, y , new_rate = new_rate)
    
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    return X_train,y_train, X_test, y_test
# X_train,y_train, X_test, y_test = load_data(test_size =0.5)
# print(X_train.shape)
# print(X_test.shape)