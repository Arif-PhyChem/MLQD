import tensorflow.keras as keras
import pickle
import numpy as np
import data as data
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

def CNN_optim(Xin: str, 
            Yin: str, 
            QDmodelOut: str,
            epochs: int, 
            patience: int):
    x_train, y_train, x_val, y_val, kernel_choice = data.data(Xin, Yin)
    print('=================================================================')
    f = open('best_param.pkl', 'rb')
    hyper_param = pickle.load(f)
    f.close()
    filter_choice = [10,30,50,70,90,110,130,150,170,190]
    dense_choice = [8,16,32,64,128,256,512]
    lr_choice = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    batch_size = [8,16,32,64,128,256,512]

    Conv1D_0 = filter_choice[hyper_param['Conv1D']] 
    Conv1D_1 = filter_choice[hyper_param['Conv1D_1']]
    Conv1D_2 = filter_choice[hyper_param['Conv1D_2']]
    Dense_0 = dense_choice[hyper_param['Dense']]
    Dense_1 = dense_choice[hyper_param['Dense_1']]
    Dense_2 = dense_choice[hyper_param['Dense_2']]
    Batch_size = batch_size[hyper_param['batch_size']] 
    If_0 = hyper_param['if']
    If_1 = hyper_param['if_1']
    Kernel_0 = kernel_choice[hyper_param['kernel_size']]
    Kernel_1 = kernel_choice[hyper_param['kernel_size_1']]
    Kernel_2 = kernel_choice[hyper_param['kernel_size_2']]
    Lr_rate = lr_choice[hyper_param['learning_rate']]

    model = Sequential()
    model.add(Conv1D(Conv1D_0, kernel_size=Kernel_0, activation ='relu', input_shape=(x_train.shape[1],1)))
    model.add(Conv1D(Conv1D_1, kernel_size=Kernel_1, activation = 'relu', padding='same'))
    if If_0 == 1:
        model.add(Conv1D(Conv1D_2, kernel_size=Kernel_2, activation = 'relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(Dense_0, activation = 'relu'))
    model.add(Dense(Dense_1, activation = 'relu'))
    if If_1 == 1: 
        model.add(Dense(Dense_2, activation = 'relu'))
    model.add(Dense(y_train.shape[1], activation='linear'))
    adam = keras.optimizers.Adam(learning_rate=Lr_rate)
    print(model.summary())
    model.compile(loss='mse', optimizer=adam)
    print('=================================================================')
    print('cnn.CNN_optim: Running wth EarlyStopping of patience =', patience)
    
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=2,
        mode="min",
        baseline=None,
        restore_best_weights=True)
    
    print('cnn.CNN_optim: Running with batch size =', Batch_size, 'and epochs =', epochs)
    print('=================================================================')
    model.fit(x_train, y_train,
          batch_size=Batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_val, y_val), callbacks=[es]) 
    QDmodelOut =QDmodelOut + '.keras'
    model.save(QDmodelOut)
    print('=================================================================')
    print('cnn.CNN_optim: OSTL model is saved as "', QDmodelOut ,'"') 

def OSTL_default(Xin: str, 
            Yin: str, 
            QDmodelOut: str,
            epochs: int,
            patience: int):


    x_train, y_train, x_val, y_val, kernel_choice = data.data(Xin, Yin)
    print('=================================================================')
    print('cnn.OSTL_default: Running wth EarlyStopping of patience =', patience)
    print('cnn.OSTL_default: Running with batch size = 16 and epochs =', epochs)
    print('=================================================================')

    model = Sequential()
    model.add(Conv1D(80, kernel_size=3, activation ='relu', input_shape=(x_train.shape[1],1)))
    model.add(Conv1D(110, kernel_size=3, activation = 'relu', padding='same'))
    model.add(Conv1D(80, kernel_size=3, activation = 'relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(y_train.shape[1], activation='linear'))
    adam = keras.optimizers.Adam(learning_rate=10**-3)
    print(model.summary())
    model.compile(loss='mse', optimizer=adam)
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=2,
        mode="min",
        baseline=None,
        restore_best_weights=True)
    model.fit(x_train, y_train,
          batch_size=16,
          epochs=epochs,
          verbose=2,
          validation_data=(x_val, y_val), callbacks=[es])

    QDmodelOut = QDmodelOut + '.keras'
    model.save(QDmodelOut)
    print('=================================================================')
    print('cnn.OSTL_default: OSTL model is saved as "', QDmodelOut, '"') 
def AIQD_default(Xin: str, 
            Yin: str, 
            QDmodelOut: str,
            epochs: int,
            patience: int):

    x_train, y_train, x_val, y_val, kernel_choice = data.data(Xin, Yin)
    print('=================================================================')
    print('cnn.AIQD_default: Running wth EarlyStopping of patience =', patience)
    print('cnn.AIQD_default: Running with batch size = 64 and epochs =', epochs)
    print('=================================================================')

    model = Sequential()
    model.add(Conv1D(80, kernel_size=3, activation ='relu', input_shape=(x_train.shape[1],1)))
    model.add(Conv1D(60, kernel_size=3, activation = 'relu', padding='same'))
    model.add(Conv1D(50, kernel_size=3, activation = 'relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(y_train.shape[1], activation='linear'))
    adam = keras.optimizers.Adam(learning_rate=10**-3)
    print(model.summary())
    model.compile(loss='mse', optimizer=adam)
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=2,
        mode="min",
        baseline=None,
        restore_best_weights=True)
    model.fit(x_train, y_train,
          batch_size=64,
          epochs=epochs,
          verbose=2,
          validation_data=(x_val, y_val), callbacks=[es])
    
    QDmodelOut = QDmodelOut + '.keras'
    model.save(QDmodelOut)
    print('=================================================================')
    print('cnn.AIQD_default: AIQD model is saved as "', QDmodelOut, '"')
