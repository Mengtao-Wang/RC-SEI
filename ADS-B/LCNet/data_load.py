import numpy as np
from sklearn.model_selection import train_test_split



def TrainDataset(random_num):
    x = np.load("D:\Dateset\Signal_dataset\SEI Data\FS-SEI_4800\X_train_10Class.npy")
    y = np.load("D:\Dateset\Signal_dataset\SEI Data\FS-SEI_4800\Y_train_10Class.npy")
    y = y.astype(np.uint8)
    print(x.shape)

    X_train_label, X_val, Y_train_label, Y_val = train_test_split(x, y, test_size=0.1, random_state=random_num)
    X_train_label = X_train_label.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)

    print(X_train_label.shape, X_val.shape)
    return X_train_label, X_val, Y_train_label, Y_val

def TestDataset():
    x = np.load("D:\Dateset\Signal_dataset\SEI Data\FS-SEI_4800\X_test_10Class.npy")
    y = np.load("D:\Dateset\Signal_dataset\SEI Data\FS-SEI_4800\Y_test_10Class.npy")
    x = x.transpose(0, 2, 1)
    print(x.shape)
    y = y.astype(np.uint8)
    return x, y


if __name__ == '__main__':
    TrainDataset(30)
    TestDataset()
