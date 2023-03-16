import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from keras.utils import to_categorical
#from common.change_rate_data import change_rate_data



def load_data(test_size):
    dataset = pd.read_csv('D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/LEC7/madaboost_23102022/madaboost/data/datasets/transfusion.csv')
    dataset_desc = dataset.describe(include = 'all')
    transfusion_map = {1:1, 0:0}
    dataset['whether he/she donated blood in March 2007'] = dataset['whether he/she donated blood in March 2007'].map(transfusion_map)
    X = dataset.drop(['whether he/she donated blood in March 2007'], axis = 1)
    y = dataset['whether he/she donated blood in March 2007']

    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    train_labels = to_categorical(y_train,num_classes=2)
    test_labels = to_categorical(y_test,num_classes=2)

    return X_train, train_labels, X_test, test_labels
    return X,y

