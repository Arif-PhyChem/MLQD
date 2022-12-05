import os
import keras
import pickle
import numpy as np
from hyperas import optim
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Activation
from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.convolutional import Conv1D
from hyperas.distributions import choice, uniform
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import train_test_split


print('*****************************************')
print('hyperopt_optim: splitting data into 80/20 % ratio (training/validation set)')
def data():
    X = str(os.getcwd()) + '/' + 'x_optim.npy'
    Y = str(os.getcwd()) + '/' + 'y_optim.npy'
    x = np.load(X)
    y = np.load(Y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=7)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
    x_val  = x_val.reshape(x_val.shape[0], x_val.shape[1],1)
    kernel_choice = []
    for i in range(2, x_train.shape[1], 1):
        kernel_choice.append(i)
    return x_train, y_train, x_val, y_val, kernel_choice
######################################################
print('hyperopt_optim: Optimizing the Neural Network with hyperopt library')
print('hyperopt_optim: Setting Optimizer to Adam and loss to mean square error (mse)')
print('hyperopt_optim: We do not optimize the activation function and set it equal to Relu')  
print('hyperopt_optim: Maximum number of evaluations = 200')
print('hyperopt_optim: Each evaluation runs for 100 epochs')
print('*****************************************')
#####################################################
def optimize_model(x_train, y_train, x_val, y_val, kernel_choice):
    model = Sequential()
    model.add(Conv1D({{choice([10,20, 30, 40, 50, 60,70, 80, 90, 100])}}, kernel_size={{choice(kernel_choice)}}, input_shape=(x_train.shape[1],1)))
    model.add(Activation({{choice(['relu'])}}))
    model.add(Conv1D({{choice([10,20, 30,40,50,60,70, 80, 90, 100])}}, kernel_size={{choice(kernel_choice)}}, padding='same'))
    model.add(Activation({{choice(['relu'])}}))    
    if ({{choice(['two', 'three'])}}) == 'three':
        model.add(Conv1D({{choice([10,20,30,40,50,60,70, 80])}}, kernel_size={{choice(kernel_choice)}}, padding='same'))
        model.add(Activation({{choice(['relu'])}}))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense({{choice([16,32,64,128, 256, 512])}}))
    model.add(Activation({{choice(['relu'])}}))
    model.add(Dense({{choice([8,16,32, 64, 128, 256,512])}}))
    model.add(Activation({{choice(['relu'])}}))
    if ({{choice(['two', 'three'])}}) == 'three':
        model.add(Dense({{choice([8,16, 32, 64, 128, 256,512])}}))
        model.add(Activation({{choice(['relu'])}}))
    model.add(Dense(y_train.shape[1], activation='linear'))


    adam = keras.optimizers.Adam(learning_rate={{choice([10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}})
    model.compile(loss='mse', optimizer=adam)
    model.fit(x_train, y_train,
              batch_size={{choice([16,32,64,128,256,512])}},
              epochs=100,
              verbose=2,
              validation_data=(x_val, y_val))
    loss= model.evaluate(x_val, y_val, verbose=0)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(optimize_model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=200,
                                        trials=Trials()) 
    x_train, y_train, x_val, y_val, kernel_choice = data()
    f = open("best_param.pkl", "wb")
    pickle.dump(best_run, f)
    f.close()
