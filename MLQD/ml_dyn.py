import os
import re
import subprocess
import numpy as np
import time as proc_time
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
############################################################
def KRR(Xin: np.ndarray,
        time: float,
        time_step: float,
        QDmodelIn: str,
        traj_output_file: str):
    
    QDmodelIn = re.split(r'.unf', QDmodelIn)[0]
    print(QDmodelIn)
    tm = Xin.shape[1]
    tmm = (tm-1)*time_step
    t = np.arange(0,time+tmm+ time_step, time_step)
    nsteps = len(t)
    y = Xin;
    y = np.resize(y, (y.shape[1], y.shape[0]))
    a = 1;
    #
    print('ml_dyn.KRR: Running dynamics with KRR model using MLatom in the backend ......')
    print('ml_dyn.KRR: The output of MLatom will be saved as "kkr_dyn_output"')
    #
    QDmodelIn = str(os.getcwd()) + '/' + QDmodelIn + '.unf'
    ti = proc_time.time()
    for i in range(tm, len(t)):
        np.savetxt('input.dat', Xin)
        arg = ['rm', '-f', 'y_est.dat']
        subprocess.run(arg, check=True)
        args = ['mlatom', 'useMlmodel', 'MlmodelIn='+ str(QDmodelIn) + ' XfileIn=input.dat', 'YestFile=y_est.dat', 'debug']
        with open('kkr_dyn_output', "w") as output:
            subprocess.run(args, check=True, stdout=output)
        y_est = np.loadtxt('y_est.dat')
        y = np.resize(y, (y.shape[0]+1, y.shape[1]))
        y[y.shape[0]-1, 0] = y_est
        Xin[0,:] = y[a:y.shape[0], 0]
        a += 1;
    arg = ['rm', '-f', 'y_est.dat']
    subprocess.run(arg, check=True)
    np.save(traj_output_file, np.c_['-1',t[:], y[:]])
    print('ml_dyn.KRR: Dynamics is saved in a file  "' + traj_output_file + '"')
    print('ml_dyn.KRR: Time taken =', proc_time.time() - ti, "sec")
    

def OSTL(Xin: np.ndarray,  
        n_states: int, 
        time: float,  
        time_step: float, 
        QDmodelIn: str,
        systemType: str,
        traj_output_file: str):
   
    QDmodelIn = str(os.getcwd()) + '/' + QDmodelIn 
    model = keras.models.load_model(QDmodelIn)
    #Show the model architecture
    model.summary()
    #
    print('ml_dyn.OSTL: Running dynamics with OSTL approach ......')
    #
    if systemType == 'FMO':
        time_range = np.arange(0,time*1000,time_step) # fs
    elif systemType == 'SB':
        time_range = np.arange(0,time,time_step) 
    time_range = np.append(time_range, time_range[time_range.shape[0]-1] + time_step, axis=None)
    nsteps = len(time_range)
    y = np.zeros((nsteps, n_states**2), dtype=float)
    y1 = np.zeros((nsteps, n_states**2), dtype=complex)
    #
    # normalizing the time feature using logistic function 
    #
    ti = proc_time.time()
    x_pred = Xin[:]
    x_pred = x_pred.reshape(1, Xin.shape[1],1) # reshape the input 
    yhat = model.predict(x_pred, verbose=0)
    a = 0; b = n_states**2;
    for i in range(0, nsteps):
        y[i,:] = yhat[0, a:b]
        a = a + n_states**2
        b = b + n_states**2
    for i in range(0, nsteps):
        # grab poplation
        a = n_states; b = n_states - 1
        c = 0; d = 0  
        for j in range(0, n_states):
            y1[i,c] = y[i,d]
            e = d + 1
            f = e + 1
            g = c + 1
            h = g + n_states - 1
            for k in range(0, b):
                y1[i,g] = y[i,e] + 1j * y[i,f]
                y1[i,h] = y[i,e] - 1j * y[i,f]
                e += 2
                f += 2
                g += 1
                h += n_states
            d += a + b
            a -= 1
            b -= 1
            c += n_states + 1
    np.save(traj_output_file, np.c_['-1',time_range, y1])
    print('ml_dyn.OSTL: Dynamics is saved in a file  "' + traj_output_file + '"')
    print('ml_dyn.OSTL: Time taken =', proc_time.time() - ti, "sec")

def AIQD(Xin: np.ndarray,  
        n_states: int, 
        time: float,  
        time_step: float, 
        numLogf: int, 
        LogCa: float,
        LogCb: float, 
        LogCc: float, 
        LogCd: float,
        QDmodelIn: str,
        systemType: str,
        traj_output_file: str):
    
    def logistic(x, c):
        
        return LogCa/(1 + LogCb * np.exp(-(x-c)/LogCd)) 
    
    QDmodelIn = str(os.getcwd()) + '/' + QDmodelIn
    model = keras.models.load_model(QDmodelIn)
    #Show the model architecture
    model.summary()
    #
    print('ml_dyn.AIQD: Running dynamics with AI-QD approach ......')
    #
    if systemType == 'FMO':
        time_range = np.arange(0,time*1000,time_step) # fs
    elif systemType == 'SB':
        time_range = np.arange(0,time,time_step) 
    time_range = np.append(time_range, time_range[time_range.shape[0]-1] + time_step, axis=None)
    if systemType == 'FMO':
        time_step = time_step/1000 # convert to ps
    t = np.arange(0,time,time_step)  
    t = np.append(t, t[t.shape[0]-1] + time_step, axis=None) # don't use time + time_step, has a bug, sometimes gives one extra value
    nsteps = len(t)
    y = np.zeros((nsteps, n_states**2), dtype=float)
    y1 = np.zeros((nsteps, n_states**2), dtype=complex)
    tt = np.zeros((nsteps, numLogf), dtype=float)
    u = 0
    for i in t:
        c = LogCc
        for j in range(0, numLogf): 
            tt[u,j]=logistic(i, c)
            c += 5.0
        u += 1
    #
    # normalizing the time feature using logistic function 
    #
    ti = proc_time.time()
    Xin_dim = Xin.shape[1]
    Xin = np.resize(Xin, (Xin.shape[0], Xin.shape[1] + numLogf))
    i = 0
    for tm in tt:
        Xin[0, Xin_dim:Xin.shape[1]] = tm[:]
        x_pred = Xin[:]
        x_pred = x_pred.reshape(1, Xin.shape[1],1) # reshape the input 
        yhat = model.predict(x_pred, verbose=0)
        y[i,:] = yhat
        i += 1
    for i in range(0, nsteps):
        # grab poplation
        a = n_states; b = n_states - 1
        c = 0; d = 0  
        for j in range(0, n_states):
            y1[i,c] = y[i,d]
            e = d + 1
            f = e + 1
            g = c + 1
            h = g + n_states - 1
            for k in range(0, b):
                y1[i,g] = y[i,e] + 1j * y[i,f]
                y1[i,h] = y[i,e] - 1j * y[i,f]
                e += 2
                f += 2
                g += 1
                h += n_states
            d += a + b
            a -= 1
            b -= 1
            c += n_states + 1
    np.save(traj_output_file, np.c_['-1',time_range, y1])
    print('ml_dyn.AIQD: Dynamics is saved in a file  "' + traj_output_file + '"')
    print('ml_dyn.AIQD: Time taken =', proc_time.time() - ti, "sec")


