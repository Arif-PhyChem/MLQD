import os
import numpy as np
from sklearn.model_selection import train_test_split; 
def data(Xin:str, Yin: str):

    Xin = Xin + '.npy'
    Yin = Yin + '.npy'
    print('Data: Checking to see whether the input data files', Xin, 'and', Yin, 'exist')
    if os.path.isfile(Xin) == False:
        raise ValueError("Data: file name " + str(Xin) + " does not exist")
    if os.path.isfile(Yin) == False:
        raise ValueError("Data: file name " + str(Yin) + " does not exist")
    
    print('Data: Loading data files', Xin, Yin)
    pwd = os.getcwd()
    X = str(pwd) + '/' + str(Xin)
    Y = str(pwd) + '/' + str(Yin)
    x = np.load(X)
    y = np.load(Y)
    print('Data: splitting data into 80/20 % ratio (training/validation set)')
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=7)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
    x_val  = x_val.reshape(x_val.shape[0], x_val.shape[1],1)
    kernel_choice = []
    for i in range(2, x_train.shape[1], 1):
        kernel_choice.append(i)
    return x_train, y_train, x_val, y_val, kernel_choice

