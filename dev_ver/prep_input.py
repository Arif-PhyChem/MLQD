import os
import glob
import re
import numpy as np

def KRR(Xin: str,
        Yin: str,
        dataCol: int, 
        xlength: int, 
        dtype: str, 
        dataPath: str):
    all_files = {}
    j = 0
    print('=================================================================')
    print('prep_input.KRR: Grabbing data from "', dataPath, '" directory') 
    for files in glob.glob(dataPath+'/*'):
        file_name = os.path.basename(files)
        all_files[file_name] = np.load(files)
        j += 1
    xx = all_files[file_name]
    length = xx[:,0].shape[0]
    file_count = j
    print('prep_input.KRR: Number of trajectories =', file_count)
    m = (length - xlength) * file_count
    x = np.zeros((int(m), xlength), dtype=float)
    y = np.zeros((int(m),1), dtype=float)
    m = 0
    for files in glob.glob(dataPath+'/*'):
        file_name = os.path.basename(files)
        df = all_files[file_name]
        if dtype == 'imag':
            y1 = df[:,dataCol].imag
        else:
            y1 = df[:,dataCol].real
        for i in range(0, (length - xlength)):
            x[m,:] = y1[i:xlength+i]
            y[m,0] = y1[i+xlength]
            m += 1
    np.savetxt(Xin, x) # the input is saved as Xin
    np.savetxt(Yin, y) # the target values are saved as Yin
    print('prep_input.KRR: The input and target values are saved as (txt files), i.e.,', Xin, 'and', Yin, ',respectively')

def RCDYN(Xin: str,
        Yin: str,
        n_states: int,
        dataCol: int, 
        xlength: int,
        time: float,
        time_step: float,
        dataPath: str,
        prior: float):
    all_files = {}
    j = 0
    print('=================================================================')
    print('prep_input.RCDYN: Grabbing data from "', dataPath, '" directory') 
    for files in glob.glob(dataPath+'/*'):
        file_name = os.path.basename(files)
        all_files[file_name] = np.load(files)
        j += 1
    xx = all_files[file_name]
    tot_length = xx[0:int(round((time+time_step)/time_step) + xlength),:].shape[0]
    file_count = j
    print('prep_input.RCDYN: Number of trajectories =', file_count)
    m = (tot_length - xlength) * file_count
    
    labels = []
    a = 0; b = n_states
    for i in range(0, n_states):
        for j in range(a, b):
            labels.append(j)
        a += n_states + 1 
        b += n_states
    divider = n_states + 1
   
    if dataCol == None:
        x = np.zeros((int(m), xlength*n_states**2), dtype=float)
        y = np.zeros((int(m), n_states**2), dtype=float)
        yy = np.zeros((xlength+1, n_states**2), dtype=complex)
    else:
        if (dataCol-1)%divider == 0:                # for diagonal terms, only real terms are considered
            x = np.zeros((int(m), xlength), dtype=float)  
            y = np.zeros((int(m), 1), dtype=float)
        else:
            x = np.zeros((int(m), xlength * 2), dtype=float)  # off-diagonal terms, real and imag 
            y = np.zeros((int(m), 2), dtype=float)
    m = 0
    for files in glob.glob(dataPath+'/*'):
        file_name = os.path.basename(files)
        df = all_files[file_name]
        if dataCol != None:
            for i in range(0, (tot_length - xlength)):
                yy = df[:, dataCol]
                if (dataCol -1)%divider == 0:
                    x[m,:] = yy[i:xlength+i].real
                    y[m,0] = yy[i+xlength].real
                else:
                    x[m,0:xlength] = yy[i:xlength+i].real
                    y[m,0] = yy[i+xlength].real
                    x[m,xlength:] = yy[i:xlength+i].imag
                    y[m,1] = yy[i+xlength].imag
                m += 1
        if dataCol == None:
            for i in range(0, (tot_length - xlength)):
                yy[:,:] = df[i:xlength+i+1, 1:n_states**2+1] # excluding the 1st column of time
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
                q = 0
                for p in labels:
                    if p%divider == 0:
                        y[m,q] = yy[j+1,p].real
                        q += 1
                    else:
                        y[m,q] = yy[j+1,p].real
                        q += 1
                        y[m,q] = yy[j+1,p].imag
                        q += 1
                m += 1
    np.save(Xin, x) # the input is saved as Xin
    np.save(Yin, y - prior) # the target values are saved as Yin

def OSTL(Xin: str,
        Yin: str,
        systemType: str, 
        n_states: int,
        time: float,
        time_step: float, 
        energyNorm: float,
        DeltaNorm: float,
        gammaNorm: float, 
        lambNorm: float, 
        tempNorm: float, 
        dataPath: str):
    all_files = {}
    j = 0
    print('=================================================================')
    print('prep_input.OSTL: Grabbing data from "', dataPath, '" directory')
    print('prep_input.OSTL: It is assumed that the data is in the same naming format and the same datatype as were adopted in our QD3SET-1 dataset \
, otherwise training files will not be successfully generated')
    for files in glob.glob(dataPath+'/*np[yz]'):
        file_name = os.path.basename(files)
        all_files[file_name] = np.load(files)
        j += 1
    file_count =  j
    print('prep_input.OSTL: Number of trajectories =', file_count)
    # create empty list
    gamma = np.zeros((file_count), dtype=float)
    lamb = np.zeros((file_count), dtype=float)
    temp = np.zeros((file_count), dtype=float)  # Inv-temperature (beta) in the case of SB model
    initial = np.zeros((file_count), dtype=int)
    epsilon = np.zeros((file_count), dtype=float)
    Delta = np.zeros((file_count), dtype=float)
    j = 0
    for files in glob.glob(dataPath+'/*np[yz]'):
        #
        # extract the values of gamma, lambda and temperature from the file name
        #
        file_name = os.path.basename(files)
        if systemType == 'SB': 
            x = re.split(r'-', file_name)
            y = re.split(r'_', x[1])
            epsilon[j] = y[0]
            y = re.split(r'_', x[2])
            Delta[j] = y[0]
            y = re.split(r'_', x[3])
            lamb[j] = y[0]
            y = re.split(r'_', x[4])
            gamma[j] = y[0]
            y = re.split(r'.n', x[5])
            temp[j] = y[0]
        else:
            x = re.split(r'_', file_name)
            y = re.split(r'-', x[1])
            initial[j] = y[1]
            y = re.split(r'-', x[2]) # extracting value of gamma
            gamma[j] = y[1] 
            y = re.split(r'-', x[3]) # extract value of lambda 
            lamb[j] = y[1]
            y = re.split(r'-', x[4])
            x = re.split(r'.npy', y[1]) # extract value of temperature
            temp[j] = x[0]

        j = j + 1
#
    print('prep_input.OSTL: Normalizing gamma, lambda and temperature using the following',
            'normalizing factors in their respective order:', gammaNorm, lambNorm, tempNorm)
#
    j=0
    for i in lamb:
        lamb[j] = i/lambNorm
        j=j+1
    j=0
    for i in gamma:
        gamma[j] = i/gammaNorm
        j=j+1
    j=0
    for i in temp:
        temp[j] = i/tempNorm
        j=j+1
    j=0
    if systemType == 'SB':
        for i in epsilon:
            epsilon[j] = i/energyNorm
            j=j+1
        j=0
        for i in Delta:
            Delta[j] = i/DeltaNorm
            j=j+1

    t = np.arange(0,time, time_step)
    t = np.append(t, t[t.shape[0]-1] + time_step, axis=None)  # don't use time + time_step, has a bug, sometimes gives one extra value
    nsteps = len(t)
    xx = all_files[file_name]
    length = xx[:,0].shape[0]
    if len(t) > length:
        raise ValueError('Train_ml.ostl: The given time is larger than the time of the training data')
    if systemType == 'SB':
        x = np.zeros((file_count, 5), dtype=float)
    else:
        x = np.zeros((file_count, 4), dtype=float)
    y = np.zeros((file_count, n_states**2 * length), dtype=float)
    yy = np.zeros((1, n_states**2), dtype=complex)
    labels = np.zeros((1, int((n_states**2-n_states)/2 + n_states)), dtype = int)
    labels = []
    a = 0; b = n_states
    for i in range(0, n_states):
        for j in range(a, b):
            labels.append(j)
        a += n_states + 1 
        b += n_states 
    divider = n_states + 1
    m = 0
    f = 0
    for files in glob.glob(dataPath+'/*np[yz]'):
        file_name = os.path.basename(files)
        df = all_files[file_name]
        if systemType == 'FMO':
            if initial[f] == 1:   
                init_index = 0.1  # use 0.1 as a label for initial excitation on site-1
            elif initial[f] == 6:
                init_index = 0.6  # use 0.6 as a label for initial excitation on site-6
            else:
                init_index = 0.8  # use 0.8 as a lebel for initial excitation on site-8
            x[f,0] = init_index
            x[f,1] = gamma[f]
            x[f,2] = lamb[f]
            x[f,3] = temp[f]
        if systemType == 'SB':
            x[m,0] = epsilon[f]
            x[m,1] = Delta[f]
            x[m,2] = gamma[f]
            x[m,3] = lamb[f]
            x[m,4] = temp[f]
        q = 0
        for i in range(0, nsteps):
            yy[0,:] = df[i, 1:n_states**2+1] # excluding the 1st column of time
            for p in labels:
                if p%divider == 0:
                    y[m,q] = yy[0,p].real
                    q += 1
                else:
                    y[m,q] = yy[0,p].real
                    q += 1
                    y[m,q] = yy[0,p].imag
                    q += 1
        m += 1
        f += 1
    np.save(Xin, x) # the input is saved as Xin
    np.save(Yin, y) # the target values are saved as Yin
    print('prep_input.OSTL: The input and target values are saved as', Xin + '.npy', 'and', Yin + '.npy', ', respectively.')

def AIQD(Xin: str,
        Yin: str,
        systemType: str, 
        n_states: int, 
        time: float, 
        time_step: float, 
        numLogf: int,
        LogCa: float,
        LogCb: float, 
        LogCc: float, 
        LogCd: float,
        energyNorm: float,
        DeltaNorm: float,
        gammaNorm: float, 
        lambNorm: float, 
        tempNorm:float, 
        dataPath: str):
    all_files = {}
    j = 0
    print('=================================================================')
    print('prep_input.AIQD: Grabbing data from "', dataPath, '" directory')
    print('prep_input.AIQD: It is assumed that the data is in the same naming format and the same datatype as were adopted in our QD3SET-1 dataset \
, otherwise training files will not be successfully generated')
    for files in glob.glob(dataPath+'/*np[yz]'):
        file_name = os.path.basename(files)
        all_files[file_name] = np.load(files)
        j += 1
    file_count =  j
    print('prep_input.AIQD: Number of trajectories =', file_count)
    # create empty list
    gamma = np.zeros((file_count), dtype=float)
    lamb = np.zeros((file_count), dtype=float)
    temp = np.zeros((file_count), dtype=float) # Inv-temperature (beta) in the case of SB model
    initial = np.zeros((file_count), dtype=int)
    epsilon = np.zeros((file_count), dtype=float)
    Delta = np.zeros((file_count), dtype=float)
    j = 0
    for files in glob.glob(dataPath+'/*np[yz]'):
        #
        # extract the values of gamma, lambda and temperature from the file name
        #
        file_name = os.path.basename(files)
        if systemType == 'SB': 
            x = re.split(r'-', file_name)
            y = re.split(r'_', x[1])
            epsilon[j] = y[0]
            y = re.split(r'_', x[2])
            Delta[j] = y[0]
            y = re.split(r'_', x[3])
            lamb[j] = y[0]
            y = re.split(r'_', x[4])
            gamma[j] = y[0]
            y = re.split(r'.n', x[5])
            temp[j] = y[0]
        else:
            x = re.split(r'_', file_name)
            y = re.split(r'-', x[1])
            initial[j] = y[1]
            y = re.split(r'-', x[2]) # extracting value of gamma
            gamma[j] = y[1] 
            y = re.split(r'-', x[3]) # extract value of lambda 
            lamb[j] = y[1]
            y = re.split(r'-', x[4])
            x = re.split(r'.npy', y[1]) # extract value of temperature
            temp[j] = x[0]

        j = j + 1
#
    print('prep_input.AIQD: Normalizing gamma, lambda and temperature using the following',
            'normalizing factors in their respective order:', gammaNorm, lambNorm, tempNorm)
#
    j=0
    for i in lamb:
        lamb[j] = i/lambNorm
        j=j+1
    j=0
    for i in gamma:
        gamma[j] = i/gammaNorm
        j=j+1
    j=0
    for i in temp:
        temp[j] = i/tempNorm
        j=j+1
    if systemType == 'SB':
        j=0
        for i in epsilon:
            epsilon[j] = i/energyNorm
            j=j+1
        j=0
        for i in Delta:
            Delta[j] = i/DeltaNorm
            j=j+1

    #xx = all_files[file_name]
    #length = xx[:,0].shape[0]
    t = np.arange(0,time, time_step)
    t = np.append(t, t[t.shape[0]-1] + time_step, axis=None)  # don't use time + time_step, has a bug, sometimes gives one extra value
    nsteps = len(t)
    length = nsteps
    if systemType == 'SB':
        x = np.zeros((file_count*nsteps, 5+numLogf), dtype=float)
    else:
        x = np.zeros((file_count*nsteps, 4+numLogf), dtype=float)
    y = np.zeros((nsteps, n_states**2), dtype=float)
    tt = np.zeros((nsteps, numLogf), dtype=float)
    def logistic(x, c):    
        return LogCa/(1 + LogCb * np.exp(-(x-c)/LogCd)) 
    u = 0
    for i in t:
        c = LogCc
        for j in range(0, numLogf): 
            tt[u,j]=logistic(i, c)
            c += 5.0
        u += 1
    y = np.zeros((file_count*length, n_states**2), dtype=float)
    yy = np.zeros((1, n_states**2), dtype=complex)
    labels = np.zeros((1, int((n_states**2-n_states)/2 + n_states)), dtype = int)
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
    f = 0
    for files in glob.glob(dataPath+'/*np[yz]'):
        file_name = os.path.basename(files)
        df = all_files[file_name]
        for i in range(0, length):
            if systemType == 'FMO':
                if initial[f] == 1:   
                    init_index = 0.1  # use 0.1 as a label for initial excitation on site-1
                elif initial[f] == 6:
                    init_index = 0.6  # use 0.6 as a label for initial excitation on site-6
                else:
                    init_index = 0.8  # use 0.8 as a lebel for initial excitation on site-8
                x[m,0] = init_index
                x[m,1] = gamma[f]
                x[m,2] = lamb[f]
                x[m,3] = temp[f]
                x[m,4:] = tt[i,:]
            if systemType == 'SB':
                x[m,0] = epsilon[f]
                x[m,1] = Delta[f]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:] = tt[i,:]
            yy[0,:] = df[i, 1:n_states**2+1] # excluding the 1st column of time
            q = 0
            for p in labels:
                if p%divider == 0:
                    y[m,q] = yy[0,p].real
                    q += 1
                else:
                    y[m,q] = yy[0,p].real
                    q += 1
                    y[m,q] = yy[0,p].imag
                    q += 1
            m += 1
        f += 1
    np.save(Xin, x) # the input is saved as Xin
    np.save(Yin, y) # the target values are saved as Yin
    print('prep_input.AIQD: The input and target values are saved as', Xin + '.npy', 'and', Yin + '.npy', ', respectively.')

    
      
