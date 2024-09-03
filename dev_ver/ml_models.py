import os
import keras as keras
import pickle
import numpy as np
import data as data
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv1D, LSTM
from keras.layers import MaxPooling1D
from keras.callbacks import ModelCheckpoint


def CNN(Xin: str, 
            Yin: str, 
            x_val: str,
            y_val: str,
            epochs: int, 
            patience: int,
            systemType: str,
            n_states: int,
            prior: float,
            pinn: str
            ):
    x_train, y_train, x_val, y_val, kernel_choice = data.data(Xin, Yin, x_val, y_val)
    
    optim_param_file = "best_cnn_params.pkl"
    
    print('=================================================================')
    print('ml_model.cnn: Looking for', optim_param_file)
    model = Sequential()
    
    if os.path.isfile(optim_param_file):
        print('ml_models.cnn: loading hyperparameters from', optim_param_file)
        print('=================================================================')
        f = open('best_cnn_params.pkl', 'rb')
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
        Kernel_0 = int(hyper_param['kernel_size'])
        Kernel_1 = int(hyper_param['kernel_size_1'])
        Kernel_2 = int(hyper_param['kernel_size_2'])
        Lr_rate = lr_choice[hyper_param['learning_rate']]

        print('=================================================================')
        print('ml_models.cnn: Running wth EarlyStopping of patience =', patience)
        print('ml_models.cnn: Running with batch size =', Batch_size , 'and epochs =', epochs)
        print('=================================================================')

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
    else:
        print('=================================================================')
        print('ml_models.cnn: '+ str(optim_param_file) +  ' not found, thus training CNN model with the default structure')
        print('=================================================================')
        print('ml_models.cnn: Running wth EarlyStopping of patience =', patience)
        print('ml_models.cnn: Running with batch size = 64 and epochs =', epochs)
        print('=================================================================')
        model.add(Conv1D(80, kernel_size=3, activation ='relu', input_shape=(x_train.shape[1],1)))
        model.add(Conv1D(110, kernel_size=3, activation = 'relu', padding='same'))
        model.add(Conv1D(80, kernel_size=3, activation = 'relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(32, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(y_train.shape[1], activation='linear'))
        adam = keras.optimizers.Adam(learning_rate=10**-3)

    def custom_loss(y_true, y_pred):
        if pinn == 'True':
            print('Running with custom loss: mse + trace penalty term')
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            y_pred -= prior
            trace_penalty = 0.0
            a = 0; b = n_states
            labels = []
            diagonal_idx = [i * (n_states * 2 - i) for i in range(n_states)]
            for kk in range(0, y_pred.shape[-1]//n_states**2):
                trace_t = 0.0
                # calculate trace for each time step
                for idx in diagonal_idx: 
                    trace_t += y_pred[:, kk * n_states**2 + idx]
                trace_penalty += tf.reduce_mean(tf.square(1- trace_t))
            trace_penalty /= y_pred.shape[-1]//n_states**2
                
            if systemType == 'SB':            
                tot_loss = 2.0*mse_loss + 1.0*trace_penalty
            else:
                tot_loss = 2.0*mse_loss + 1.0*trace_penalty
        else:
            tot_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        return tot_loss
    
    model.compile(loss=custom_loss, optimizer=adam)
    
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=2,
        mode="min",
        baseline=None,
        restore_best_weights=True)
  
    models_dir =  "trained_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("Directory",  models_dir, "created sucessfully where the trained models will be saved")
    else:
        print("Directory",  models_dir, "already exists where the trained models will be saved")

    filepath=models_dir+"/"+ str(system_type) + "_cnn_model-{epoch:02d}-tloss-{loss:.3e}-vloss-{val_loss:.3e}.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    if os.path.isfile(optim_param_file):
        model.fit(x_train, y_train,
          batch_size=Batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_val, y_val), callbacks=[callbacks_list, es]) 
    else:
        model.fit(x_train, y_train,
          batch_size=64,
          epochs=epochs,
          verbose=2,
          validation_data=(x_val, y_val), callbacks=[callbacks_list]) 

def LSTM(Xin: str, 
            Yin: str,
            x_val: str,
            y_val: str,
            epochs: int,
            patience: int,
            systemType: str, 
            n_states: int,
            prior: float,
            pinn: str):


    x_train, y_train, x_val, y_val, kernel_choice = data.data(Xin, Yin, x_val, y_val)

    print('=================================================================')
    print('ml_models.lstm: Running wth EarlyStopping of patience =', patience)
    print('ml_models.lstm: Running with batch size = 64 and epochs =', epochs)
    print('=================================================================')

    
    optim_param_file = "best_lstm_params.pkl"
    
    print('=================================================================')
    print('ml_model.lstm: Looking for',optim_param_file)
    
    
    model = Sequential()
    
    if os.path.isfile(optim_param_file):
        print('ml_models.lstm: loading hyperparameters from', optim_param_file)
        print('=================================================================')
        f = open('best_lstm_params.pkl', 'rb')
        hyper_param = pickle.load(f)
        f.close()
        dense_choice = [8,16,32,64,128,256,512]
        lr_choice = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
        batch_size = [8,16,32,64,128,256,512]
        lstm_choice = np.arange(16, 512, 32)

        lstm_units_0 = int(lstm_choice[hyper_param['lstm_units']])
        lstm_units_1 = int(lstm_choice[hyper_param['lstm_units_1']])
        lstm_units_2 = int(lstm_choice[hyper_param['lstm_units_2']])
        Dense_0 = dense_choice[hyper_param['Dense']]
        Dense_1 = dense_choice[hyper_param['Dense_1']]
        Dense_2 = dense_choice[hyper_param['Dense_2']]
        Batch_size = batch_size[hyper_param['batch_size']] 
        If_lstm = hyper_param['if_lstm']
        If_1 = hyper_param['if_1']
        Lr_rate = lr_choice[hyper_param['learning_rate']]

        print('=================================================================')
        print('ml_models.lstm: Running wth EarlyStopping of patience =', patience)
        print('ml_models.lstm: Running with batch size =', Batch_size , 'and epochs =', epochs)
        print('=================================================================')


        if If_lstm == 0:
            model.add(LSTM(lstm_units_0, return_sequences=False, input_shape=(x_train.shape[1],1)))

        if If_lstm == 1:
            model.add(LSTM(lstm_units_0, return_sequences=True, input_shape=(x_train.shape[1],1)))
            model.add(LSTM(lstm_units_1, return_sequences=False))

        if If_lstm == 2:
            model.add(LSTM(lstm_units_0, return_sequences=True, input_shape=(x_train.shape[1],1)))
            model.add(LSTM(lstm_units_1, return_sequences=True))
            model.add(LSTM(lstm_units_2, return_sequences=False))
        
        model.add(Dense(Dense_0, activation = 'relu'))
        model.add(Dense(Dense_1, activation = 'relu'))
        
        if If_1 == 1: 
            
            model.add(Dense(Dense_2, activation = 'relu'))
        
        model.add(Dense(y_train.shape[1], activation='linear'))
        
        adam = keras.optimizers.Adam(learning_rate=Lr_rate)
    
    else:
        print('=================================================================')
        print('ml_models.lstm: '+ str(optim_param_file) +  ' not found, thus training LSTM model with the default structure')
        print('=================================================================')
        print('ml_models.lstm: Running wth EarlyStopping of patience =', patience)
        print('ml_models.lstm: Running with batch size = 64 and epochs =', epochs)
        print('=================================================================')

        model.add(LSTM(112,  return_sequences=False, input_shape=(x_train.shape[1],1)))
        model.add(Dense(80, activation = 'relu'))
        model.add(Dense(336, activation = 'relu'))
        model.add(Dense(304, activation = 'relu'))
        model.add(Dense(y_train.shape[1], activation='linear'))
        adam = keras.optimizers.Adam(learning_rate=10**-3)
    
    print(model.summary())

    def custom_loss(y_true, y_pred):
        if pinn == 'True':
            print('Running with custom loss: mse + trace penalty term')
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            y_pred -= prior
            trace_penalty = 0.0
            diagonal_idx = [i * (n_states * 2 - i) for i in range(n_states)]
            for kk in range(0, y_pred.shape[-1]//n_states**2):
                trace_t = 0.0
                # calculate trace for each time step
                for idx in diagonal_idx: 
                    trace_t += y_pred[:, kk * n_states**2 + idx]
                trace_penalty += tf.reduce_mean(tf.square(1- trace_t))
            trace_penalty /= y_pred.shape[-1]//n_states**2
            
            if systemType == 'SB':
                tot_loss = 2.0*mse_loss + 1.0*trace_penalty
            else:
                tot_loss = 2.0*mse_loss + 1.0*trace_penalty
        else:
            tot_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return tot_loss
    model.compile(loss=custom_loss, optimizer=adam)
    
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=2,
        mode="min",
        baseline=None,
        save_best_only=True, 
        restore_best_weights=True)
    
    models_dir =  "trained_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("Directory",  models_dir, "created sucessfully where the trained models will be saved")
    else:
        print("Directory",  models_dir, "already exists where the trained models will be saved")

    filepath=models_dir+"/"+ str(system_type)+"_lstm_model-{epoch:02d}-tloss-{loss:.3e}-vloss-{val_loss:.3e}.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
  
    if os.path.isfile(optim_param_file):
        
        model.fit(x_train, y_train,
          batch_size=Batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_val, y_val), callbacks=[es])
    else:
        
        model.fit(x_train, y_train,
          batch_size=64,
          epochs=epochs,
          verbose=2,
          validation_data=(x_val, y_val), callbacks=[es])
