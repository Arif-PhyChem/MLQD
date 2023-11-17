import os
import tensorflow.keras as keras
import pickle
import numpy as np
import data as data
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Conv1D
from hyperopt import fmin, hp, Trials, STATUS_OK, tpe
from sklearn.model_selection import train_test_split

def optimize(Xin: str, 
            Yin: str,
            x_val: str, 
            y_val: str,
            epochs: int, 
            max_evals: int):

    x_train, y_train, x_val, y_val, kernel_choice = data.data(Xin, Yin, x_val, y_val)
    ######################################################
    print('hyperopt_optim: Optimizing the Neural Network with hyperopt library')
    print('hyperopt_optim: Setting Optimizer to Adam and loss to mean square error (mse)')
    print('hyperopt_optim: We do not optimize the activation function and set it equal to Relu')  
    print('hyperopt_optim: Maximum number of evaluations =', max_evals)
    print('hyperopt_optim: Each evaluation runs for ' + str(epochs) + ' epochs')
    print('=================================================================')
    #####################################################
    space = {'Conv1D': hp.choice('Conv1D', [10,30,50,70,90,110,130,150,170,190]),
            'Conv1D_1': hp.choice('Conv1D_1', [10,30,50,70,90,110,130,150,170,190]),
            'Conv1D_2': hp.choice('Conv1D_2', [10,30,50,70,90,110,130,150,170,190]),
            'Dense': hp.choice('Dense', [8, 16,32,64,128,256,512]),
            'Dense_1': hp.choice('Dense_1', [8, 16,32,64,128,256,512]),
            'Dense_2': hp.choice('Dense_2', [8, 16,32,64,128,256,512]),
            'if': hp.choice('if', [{'layers': 'two', }, {'layers': 'three'}]), 
            'if_1': hp.choice('if_1', [{'layers': 'two', }, {'layers': 'three'}]),
            'kernel_size': hp.choice('kernel_size', kernel_choice),
            'kernel_size_1': hp.choice('kernel_size_1', kernel_choice),
            'kernel_size_2': hp.choice('kernel_size_2', kernel_choice),
            'learning_rate': hp.choice('learning_rate', [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]),
            'batch_size': hp.choice('batch_size', [8, 16,32,64,128,256,512]),
            'activation': 'relu'
        } 
    def optimize_model(params):
        model = Sequential()
        model.add(Conv1D(params['Conv1D'], kernel_size=params['kernel_size'], input_shape=(x_train.shape[1],1)))
        model.add(Activation(params['activation']))
        model.add(Conv1D(params['Conv1D_1'], kernel_size=params['kernel_size_1'], padding='same'))
        model.add(Activation(params['activation']))    
        if params['if']['layers'] == 'three':
            model.add(Conv1D(params['Conv1D_2'], kernel_size=params['kernel_size_2'], padding='same'))
            model.add(Activation(params['activation']))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(params['Dense']))
        model.add(Activation(params['activation']))
        model.add(Dense(params['Dense_1']))
        model.add(Activation(params['activation']))
        if params['if_1']['layers'] == 'three':
            model.add(Dense(params['Dense_2']))
            model.add(Activation(params['activation']))

        model.add(Dense(y_train.shape[1], activation='linear'))

        adam = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(loss='mse', optimizer=adam)
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
    f = open("best_param.pkl", "wb")
    pickle.dump(best_run, f)
    f.close()
