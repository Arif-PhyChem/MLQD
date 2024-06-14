import os
import numpy as np
from sklearn.model_selection import train_test_split; 
def data(Xin:str, Yin: str, x_val: str, y_val: str):

    Xin = Xin + '.npy'
    Yin = Yin + '.npy'
    print('Data: Checking to see whether the input data files', Xin, 'and', Yin, 'exist')
    if os.path.isfile(Xin) == False:
        raise ValueError("Data: file name " + str(Xin) + " does not exist")
    if os.path.isfile(Yin) == False:
        raise ValueError("Data: file name " + str(Yin) + " does not exist")
    if x_val != None:
        x_val = x_val + '.npy'
        y_val = y_val + '.npy'
        if os.path.isfile(x_val) == False:
            raise ValueError("Data: file name " + str(Xin) + " does not exist")
        if os.path.isfile(y_val) == False:
            raise ValueError("Data: file name " + str(Yin) + " does not exist")
   
    print('Data: Loading data files', Xin, 'and', Yin)
    pwd = os.getcwd()
    X = str(pwd) + '/' + str(Xin)
    Y = str(pwd) + '/' + str(Yin)
    x = np.load(X)
    y = np.load(Y)
    
    if x_val == None:
        print('Data: splitting data into sub-training/validation sets with 80/20 % ratio')
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=7)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
        x_val  = x_val.reshape(x_val.shape[0], x_val.shape[1],1)
    else:
        x_train = x
        y_train = y
        print('Data: Loading data files', x_val, 'and', y_val)
        X = str(pwd) + '/' + str(x_val)
        Y = str(pwd) + '/' + str(y_val)
        x_val = np.load(X)
        y_val = np.load(Y)
        x_train = x_train.reshape(x.shape[0], x.shape[1],1)
        x_val  = x_val.reshape(x_val.shape[0], x_val.shape[1],1)
    
    kernel_size_lim = x_train.shape[1]
    return x_train, y_train, x_val, y_val, kernel_size_lim

