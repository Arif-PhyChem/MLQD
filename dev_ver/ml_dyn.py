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
        init_time: float,
        time_step: float,
        QDmodelIn: str,
        traj_output_file: str):
    
    QDmodelIn = re.split(r'.unf', QDmodelIn)[0]
    tm = Xin.shape[1]
    if init_time != 0:
        t = init_time + time_step
    else:
        t = init_time
    tt = t
    for i in range(0, tm + int(time/time_step)-1):
        tt += time_step
        t = np.append(t, tt)
    y = Xin;
    y = np.resize(y, (y.shape[1], y.shape[0]))
    a = 1;
    #
    print('ml_dyn.KRR: Running dynamics with KRR model using MLatom in the backend ......')
    print('ml_dyn.KRR: The output of MLatom will be saved as "krr_dyn_output"')
    
    path_to_model = str(os.getcwd()) + '/' + QDmodelIn + '.unf'
    #
    # check whether the model exists in the current directory
    #
    if os.path.isfile(path_to_model):
        QDmodelIn = str(os.getcwd()) + '/' + QDmodelIn + '.unf'
    else:
        QDmodelIn = QDmodelIn + '.unf' 
    ti = proc_time.time()
    for i in range(0, int(time/time_step)):
        np.savetxt('input.dat', Xin)
        arg = ['rm', '-f', 'y_est.dat']
        subprocess.run(arg, check=True)
        args = ['mlatom', 'useMlmodel', 'MlmodelIn='+ str(QDmodelIn) + ' XfileIn=input.dat', 'YestFile=y_est.dat', 'debug']
        with open('krr_dyn_output', "w") as output:
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
    
def RCDYN(n_states: int, 
        Xin: np.ndarray,
        time: float,
        time_step: float,
        cons_trace: str,
        QDmodelIn: str,
        traj_output_file: str,
        prior
        ):

    if cons_trace == 'False':
        path_to_model = str(os.getcwd()) + '/' + QDmodelIn
        #
        # check whether the model exists in the current directory
        #
        if os.path.isfile(path_to_model):
            QDmodelIn = str(os.getcwd()) + '/' + QDmodelIn
        else:
            QDmodelIn = QDmodelIn
        model = keras.models.load_model(QDmodelIn, compile=False)
        #Show the model architecture
        model.summary()
    else:
        QDmodelIn_0 = re.split(r',\s+', QDmodelIn)[0]
        path_to_model = str(os.getcwd()) + '/' + QDmodelIn_0
        if os.path.isfile(path_to_model):
            QDmodelIn_0 = str(os.getcwd()) + '/' + QDmodelIn_0
        else:
            QDmodelIn_0 = QDmodelIn_0
        print('==========================')
        print('         Model # 1        ')
        print('==========================')
        model_0 = keras.models.load_model(QDmodelIn_0, compile=False)
        #Show the model architecture
        model_0.summary()
        
        QDmodelIn_1 = re.split(r',\s+', QDmodelIn)[1]
        path_to_model = str(os.getcwd()) + '/' + QDmodelIn_1
        if os.path.isfile(path_to_model):
            QDmodelIn_1 = str(os.getcwd()) + '/' + QDmodelIn_1
        else:
            QDmodelIn_1 = QDmodelIn_1
        print('==========================')
        print('         Model # 2        ')
        print('==========================')
        model_1 = keras.models.load_model(QDmodelIn_1, compile=False)
        #Show the model architecture
        model_1.summary()
    tm = Xin.shape[0]
    time_range=0
    tt = time_range
    for i in range(0, tm + int(time/time_step)-1):
        tt += time_step
        time_range = np.append(time_range, tt)
    a = 1;
    #
    # Changing Xin to the required format
    #
    xlength = tm
    labels = []
    m = 0; 
    a = 0; b = n_states
    for i in range(0, n_states):
        for j in range(a, b):
            labels.append(j)
            m += 1
        a += n_states + 1 
        b += n_states
    divider = n_states + 1
    m = 0
    x = np.zeros((1, xlength*n_states**2), dtype=float)
    yy = np.zeros((Xin.shape[0], Xin.shape[1]), dtype=complex)
    yy[:,:] = Xin[:, 0:n_states**2] # excluding the 1st column of time
    k = 0
    for j in range(0, xlength):
        q = 0
        for p in labels:
            if p%divider == 0:
                x[m,q+k] = yy[j,p].real
                q += 1
            else:
                x[m,q+k] = yy[j,p].real
                q += 1
                x[m,q+k] = yy[j,p].imag
                q += 1
        k += n_states**2
    print('ml_dyn.RCDYN: Running recursive dynamics with CNN model......')
    y = np.zeros((len(time_range), n_states**2), dtype=complex)
    y[0:xlength,:] = Xin;
    ti = proc_time.time()
    for i in range(xlength, len(time_range)):
        if cons_trace == 'True':
            x_pred = x
            x_pred = x_pred.reshape(1, x.shape[1],1) # reshape the input 
            yhat_0 = model_0.predict(x_pred, verbose=0)
            yhat_1 = model_1.predict(x_pred, verbose=0)
            yhat_1 -= prior  # subtract the prior
            y0 = np.zeros((1,n_states), dtype=float)
            y1 = np.zeros((1,n_states), dtype=float)
            a = n_states; b = n_states - 1
            c = 0
            # grab diagonal terms
            for j in range(0,n_states): 
                y0[0,j] = yhat_0[0,c]
                y1[0,j] = yhat_1[0,c]
                c += a + b
                a -= 1
                b -= 1
                # Difference in the result of two models
            dy = 0.0
            trace = np.sum(y0[:])
            for j in range(0, n_states):
                dy = dy + abs(y0[0,j] - y1[0,j])
            for j in range(0,n_states):
                # weight
                w = abs(y0[0,j] - y1[0,j])/dy
                # correction
                y0[0,j] = y0[0,j] + w * (1.0 - trace)
            # replace the diagonal values with the corrected ones
            a = n_states; b = n_states - 1
            c = 0
            for j in range(0,n_states): 
                yhat_0[0,c] = y0[0,j]
                c += a + b
                a -= 1
                b -= 1
            for j in range(0, n_states**2):
                x = np.delete(x,0)
            
            x=np.append(x,yhat_0)
            x=x.reshape(1,x.shape[0])
            a = n_states; b = n_states - 1
            c = 0; d = 0 
            for j in range(0, n_states):
                y[i,c] = yhat_0[0,d]
                e = d + 1
                f = e + 1
                g = c + 1
                h = g + n_states - 1
                for k in range(0, b):
                    y[i,g] = yhat_0[0,e] + 1j * yhat_0[0,f]
                    y[i,h] = yhat_0[0,e] - 1j * yhat_0[0,f]
                    e += 2
                    f += 2
                    g += 1
                    h += n_states
                d += a + b
                a -= 1
                b -= 1
                c += n_states + 1
        else:
            x_pred = x
            x_pred = x_pred.reshape(1, x.shape[1],1) # reshape the input 
            yhat = model.predict(x_pred, verbose=0) - prior
            for j in range(0, n_states**2):
                x = np.delete(x,0)
            x=np.append(x,yhat)
            x=x.reshape(1,x.shape[0])
            a = n_states; b = n_states - 1
            c = 0; d = 0 
            for j in range(0, n_states):
                y[i,c] = yhat[0,d]
                e = d + 1
                f = e + 1
                g = c + 1
                h = g + n_states - 1
                for k in range(0, b):
                    y[i,g] = yhat[0,e] + 1j * yhat[0,f]
                    y[i,h] = yhat[0,e] - 1j * yhat[0,f]
                    e += 2
                    f += 2
                    g += 1
                    h += n_states
                d += a + b
                a -= 1
                b -= 1
                c += n_states + 1
    np.save(traj_output_file, np.c_['-1',time_range[:], y[:]])
    print('ml_dyn.RCDYN: Dynamics is saved in a file  "' + traj_output_file + '"')
    print('ml_dyn.RCDYN: Time taken =', proc_time.time() - ti, "sec")

def OSTL(Xin: np.ndarray,  
        n_states: int, 
        time: float,  
        time_step: float, 
        QDmodelIn: str,
        systemType: str,
        traj_output_file: str):
  
    path_to_model = str(os.getcwd()) + '/' + QDmodelIn
    #
    # check whether the model exists in the current directory
    #
    if os.path.isfile(path_to_model):
        QDmodelIn = str(os.getcwd()) + '/' + QDmodelIn
    else:
        QDmodelIn = QDmodelIn

    model = keras.models.load_model(QDmodelIn, compile=False)
    #Show the model architecture
    model.summary()
    #
    print('ml_dyn.OSTL: Running dynamics with OSTL approach ......')
    #
    time_range = 0
    tt = time_range
    for i in range(0, int(time/time_step)-1):
        tt += time_step
        time_range = np.append(time_range, tt)
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
    if len(yhat[0, :]) < nsteps*n_states**2:
        raise ValueError('ml_dyn.OSTL: The model can only predict dynamics to the time length it was trained on. Please keep time equal to the training time length')
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
    if nsteps < y1.shape[0]:
        t1 = time_range([0])
            
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
    
    path_to_model = str(os.getcwd()) + '/' + QDmodelIn
    #
    # check whether the model exists in the current directory
    #
    if os.path.isfile(path_to_model):
        QDmodelIn = str(os.getcwd()) + '/' + QDmodelIn
    else:
        QDmodelIn = QDmodelIn
    
    model = keras.models.load_model(QDmodelIn)
    #Show the model architecture
    model.summary()
    #
    print('ml_dyn.AIQD: Running dynamics with AI-QD approach ......')
    #
    time_range = 0
    tt = time_range
    for i in range(0, int(time/0.005)-1):
        tt += time_step
        time_range = np.append(time_range, tt)
    nsteps = len(time_range)
    y = np.zeros((nsteps, n_states**2), dtype=float)
    y1 = np.zeros((nsteps, n_states**2), dtype=complex)
    tt = np.zeros((nsteps, numLogf), dtype=float)
    u = 0
    for i in time_range:
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


