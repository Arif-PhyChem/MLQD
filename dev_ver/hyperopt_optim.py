import os
import tensorflow as tf
import keras as keras
import pickle
import numpy as np
import data as data
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import MaxPooling1D
from keras.layers import Conv1D, LSTM
from hyperopt import fmin, hp, Trials, STATUS_OK, tpe
from sklearn.model_selection import train_test_split

def optimize(Xin: str, 
            Yin: str,
            x_val: str, 
            y_val: str,
            epochs: int, 
            max_evals: int,
            systemType: str,
            n_states: int,
            prior: float,
            pinn: str,
            MLmodel: str
            ):

    x_train, y_train, x_val, y_val, kernel_choice = data.data(Xin, Yin, x_val, y_val)
    ######################################################
    print('hyperopt_optim: Optimizing the Neural Network with hyperopt library')
    print('hyperopt_optim: Setting Optimizer to Adam and loss to mean square error (mse)')
    print('hyperopt_optim: We do not optimize the activation function and set it equal to Relu')  
    print('hyperopt_optim: Maximum number of evaluations =', max_evals)
    print('hyperopt_optim: Each evaluation runs for ' + str(epochs) + ' epochs')
    print('=================================================================')
    #####################################################

    if MLmodel == 'cnn':
        space = {'Conv1D': hp.choice('Conv1D', [10,30,50,70,90,110,130,150,170,190]),
            'Conv1D_1': hp.choice('Conv1D_1', [10,30,50,70,90,110,130,150,170,190]),
            'Conv1D_2': hp.choice('Conv1D_2', [10,30,50,70,90,110,130,150,170,190]),
            'Dense': hp.choice('Dense', [8, 16,32,64,128,256,512]),
            'Dense_1': hp.choice('Dense_1', [8, 16,32,64,128,256,512]),
            'Dense_2': hp.choice('Dense_2', [8, 16,32,64,128,256,512]),
            'if': hp.choice('if', [{'layers': 'two', }, {'layers': 'three'}]), 
            'if_1': hp.choice('if_1', [{'layers': 'two', }, {'layers': 'three'}]),
            'kernel_size': hp.uniform('kernel_size', 1, kernel_choice),
            'kernel_size_1': hp.uniform('kernel_size_1', 1, kernel_choice),
            'kernel_size_2': hp.uniform('kernel_size_2', 1, kernel_choice),
            'learning_rate': hp.choice('learning_rate', [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]),
            'batch_size': hp.choice('batch_size', [8, 16,32,64,128,256,512]),
            'activation': 'relu'
        } 

    if MLmodel == 'lstm':
        space = {'Dense': hp.choice('Dense', [8, 16,32,64,128,256,512]),
            'Dense_1': hp.choice('Dense_1', [8, 16,32,64,128,256,512]),
            'Dense_2': hp.choice('Dense_2', [8, 16,32,64,128,256,512]),
            'lstm_units': hp.choice('lstm_units', np.arange(16, 512, 32)),
            'lstm_units_1': hp.choice('lstm_units_1', np.arange(16, 512, 32)),
            'lstm_units_2': hp.choice('lstm_units_2', np.arange(16, 512, 32)),
            'if_1': hp.choice('if_1', [{'layers': 'two', }, {'layers': 'three'}]),
            'if_lstm': hp.choice('if_lstm', [{'layers': 'one'}, {'layers': 'two'}, {'layers': 'three'}]),
            'learning_rate': hp.choice('learning_rate', [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]),
            'batch_size': hp.choice('batch_size', [32, 64,128,256,512]),
            'activation': 'relu'
        } 

    def optimize_model(params):
        model = Sequential()
        if MLmodel == 'cnn':
            model.add(Conv1D(params['Conv1D'], kernel_size=int(params['kernel_size']), input_shape=(x_train.shape[1],1)))
            model.add(Activation(params['activation']))
            model.add(Conv1D(params['Conv1D_1'], kernel_size=int(params['kernel_size_1']), padding='same'))
            model.add(Activation(params['activation']))    
            if params['if']['layers'] == 'three':
                model.add(Conv1D(params['Conv1D_2'], kernel_size=int(params['kernel_size_2']), padding='same'))
                model.add(Activation(params['activation']))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())

        if MLmodel == 'lstm':
            if params['if_lstm']['layers'] == 'one':
                model.add(LSTM(int(params['lstm_units']), return_sequences=False, input_shape=(x_train.shape[1],1)))

            if params['if_lstm']['layers']  == 'two':
                model.add(LSTM(int(params['lstm_units']), return_sequences=True, input_shape=(x_train.shape[1],1)))
                model.add(LSTM(int(params['lstm_units_1']), return_sequences=False))

            if params['if_lstm']['layers']  == 'three':
                model.add(LSTM(int(params['lstm_units']), return_sequences=True, input_shape=(x_train.shape[1],1)))
                model.add(LSTM(int(params['lstm_units_1']), return_sequences=True))
                model.add(LSTM(int(params['lstm_units_2']), return_sequences=False))
        
        model.add(Dense(params['Dense']))
        model.add(Activation(params['activation']))
        model.add(Dense(params['Dense_1']))
        model.add(Activation(params['activation']))
        if params['if_1']['layers'] == 'three':
            model.add(Dense(params['Dense_2']))
            model.add(Activation(params['activation']))
        
        model.add(Dense(y_train.shape[1], activation='linear'))

        adam = keras.optimizers.Adam(learning_rate=params['learning_rate'])

        def custom_loss(y_true, y_pred):
            if pinn == 'True':
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
                    tot_loss = 1.0*mse_loss + 0.5*trace_penalty
            else:
                tot_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            return tot_loss
        
        model.compile(loss=custom_loss, optimizer=adam)
        model.fit(x_train, y_train,
                  batch_size=params['batch_size'],
                  epochs=epochs,
                  verbose=2,
                  validation_data=(x_val, y_val))
        loss= model.evaluate(x_val, y_val, verbose=0)
        return {'loss': loss, 'status': STATUS_OK, 'model': model}

    #if __name__ == '__main__':
    trials = Trials()
    best_run = fmin(optimize_model,
                        space,
                        algo=tpe.suggest,
                        max_evals=max_evals,
                        trials=trials)
    if MLmodel == 'cnn':
        f = open("best_cnn_params.pkl", "wb")
    if MLmodel == 'lstm':
        f = open("best_lstm_params.pkl", "wb")
    pickle.dump(best_run, f)
    f.close()
