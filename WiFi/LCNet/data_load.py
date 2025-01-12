import numpy as np
from sklearn.model_selection import train_test_split


def TrainDataset(random_num):
    x = np.load("D:\Dateset\Signal_dataset\SEI Data\WiFi\Feet62_X_train_New.npy")
    y = np.load("D:\Dateset\Signal_dataset\SEI Data\WiFi\Feet62_Y_train_New.npy")
    y = y.astype(np.uint8)
    print(x.shape)

    X_train_label, X_val, Y_train_label, Y_val = train_test_split(x, y, test_size=0.1, random_state=random_num)
    X_train_label = X_train_label.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)

    print(X_train_label.shape, X_val.shape)
    return X_train_label, X_val, Y_train_label, Y_val


def TestDataset():
    x = np.load("D:\Dateset\Signal_dataset\SEI Data\WiFi\Feet62_X_test_New.npy")
    y = np.load("D:\Dateset\Signal_dataset\SEI Data\WiFi\Feet62_Y_test_New.npy")
    x = x.transpose(0, 2, 1)
    y = y.astype(np.uint8)
    print(x.shape, y.shape)
    return x, y

if __name__ == '__main__':
    TrainDataset(30)
    x, y = TestDataset()



