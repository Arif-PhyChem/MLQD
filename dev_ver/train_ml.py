import os
import subprocess
import numpy as np
import ml_models as ml_models
import hyperopt_optim as optim
import time as proc_time
import keras as keras
import prep_input as prep_input
############################################################
def KRR(Xin: str,
        Yin: str,
        QDmodelOut: str,
        prepInput: str, 
        dataCol: int, 
        xlength: int, 
        dtype: str, 
        dataPath: str,
        hyperParam: str,
        krrSigma: float,
        krrLamb: float):
    #
    if prepInput == 'True':
        print('train_ml.KRR: preparing training data for KRR model')
        prep_input.KRR(Xin,
                    Yin, 
                    dataCol, 
                    xlength, 
                    dtype, 
                    dataPath)
    print('=================================================================')
    print('Train_ml.KRR: Training KRR model with Gaussian kernel using MLatom in the backend ......')
    print('Train_ml.KRR: KRR model will be created and saved as "' + str(QDmodelOut) + '"')
    #
    print('Train_ml.KRR: Checking to see whether the input data files', Xin, 'and', Yin, 'exists')
    if os.path.isfile(Xin) == False:
        raise ValueError("Train_ml.KRR: file name " + str(Xin) + " does not exist")
    if os.path.isfile(Yin) == False:
        raise ValueError("Train_ml.KRR: file name ", str(Xin), " does not exist")
    print('=================================================================')
    ti = proc_time.time()
    QDmodelOut = QDmodelOut + '.unf'
    arg = ['rm', '-f', QDmodelOut + '*']
    subprocess.run(arg, check=True)
    if hyperParam == 'True':
        args = ['mlatom', 'createMLmodel', 'MLmodelOut='+ str(QDmodelOut) + ' XfileIn=' + str(Xin), 'Yfile=' + str(Yin), 
                    'kernel=Gaussian', 'sigma=opt', 'lgSigmaL=-25', 'lgSigmaH=25', 'lambda=opt', 'lgLambdaL=-30.0', 'sampling=random']
        with open('krr_train_output', "w") as output:
            subprocess.run(args, check=True, stdout=output)
    else:
        args = ['mlatom', 'createMLmodel', 'MLmodelOut='+ str(QDmodelOut) + ' XfileIn=' + str(Xin), 'Yfile=' + str(Yin), 
                    'kernel=Gaussian', 'sigma='+ str(krrSigma), 'lambda=' + str(krrLamb)]
        with open('krr_train_output', "w") as output:
            subprocess.run(args, check=True, stdout=output)
    print('Train_ml.KRR: The output of MLatom can be found as "krr_train_output" file',
    '(please check to ensure that MLatom execution was successful)') 
    print('Train_ml.KRR: Time taken =', proc_time.time() - ti, "sec")

def RCDYN(Xin: str,
        Yin: str,
        x_val: str,
        y_val: str,
        systemType: str,
        n_states: int,
        dataCol: int, 
        xlength: int,
        time: float,
        time_step: float,
        ostl_steps: int, 
        gammaNorm: float,
        lambNorm: float, 
        tempNorm: float,
        dataPath: str,  # optional if prepInput is False
        prior: float,
        pinn: str,
        MLmodel: str,
        hyperParam: str,
        OptEpochs: int,
        TrEpochs: int,
        max_evals: int, 
        patience: int,
        prepInput: str):

    if prepInput == 'True':
        print('train_ml.RCDYN: preparing training data for RCDYN model')
        prep_input.RCDYN(Xin,
                        Yin,
                        systemType,
                        n_states,
                        dataCol,
                        xlength,
                        time,
                        time_step,
                        ostl_steps, 
                        gammaNorm,
                        lambNorm, 
                        tempNorm,
                        dataPath,
                        prior)
    if hyperParam == 'True':
        print('=================================================================')
        print('Train_ml.RCDYN: Going for hyperopt optimization of ' + str(MLmodel))
        print('=================================================================')
        t1 = proc_time.time()
        optim.optimize(Xin, Yin, x_val, y_val, OptEpochs, max_evals, systemType, n_states, prior, pinn, MLmodel)
        print('Train_ml.RCDYN: Time taken for optimization =', proc_time.time() - t1, "sec")
        print('Train_ml.RCDYN: Training model with the optimized hyper_parameters')
        print('=================================================================')
    
    t1 = proc_time.time()
    if MLmodel == 'cnn':
        ml_models.CNN(Xin, Yin, x_val, y_val, TrEpochs, patience, systemType, n_states, prior, pinn)
    if MLmodel == 'lstm':
        ml_models.LSTM(Xin, Yin, x_val, y_val, TrEpochs, patience, systemType, n_states, prior, pinn)

    print('Train_ml.RCDYN: Time taken for training =', proc_time.time() - t1 , "sec")
            
def OSTL(Xin: str,
        Yin: str,
        x_val: str,
        y_val: str,
        systemType: str,
        n_states: int,
        time: float,
        time_step: float,
        energyNorm: float,
        DeltaNorm: float,
        gammaNorm: float,
        lambNorm: float, 
        tempNorm: float,
        dataPath: str,  # optional if prepInput is False
        prior: float,
        pinn: str,
        QDmodelOut: str,
        hyperParam: str,
        OptEpochs: int,
        TrEpochs: int,
        max_evals: int, 
        patience: int,
        prepInput: str):
    
    if prepInput == 'True':
        print('train_ml.OSTL: preparing training data for OSTL model')
        prep_input.OSTL(Xin,
                        Yin,
                        systemType, 
                        n_states,
                        time,
                        time_step,
                        energyNorm,
                        DeltaNorm,
                        gammaNorm, 
                        lambNorm, 
                        tempNorm, 
                        dataPath)

    if hyperParam == 'True':
        print('=================================================================')
        print('Train_ml.OSTL: Going for hyperopt optimization')
        print('=================================================================')
        t1 = proc_time.time()
        optim.optimize(Xin, Yin, x_val, y_val, OptEpochs, max_evals, n_states, prior, pinn)
        print('Train_ml.OSTL: Time taken for optimization =', proc_time.time() - t1, "sec")
        print('Train_ml.OSTL: Training CNN model with the optimized hyper_parameters')
        print('=================================================================')
    
    t1 = proc_time.time()
    if MLmodel == 'cnn':
        ml_models.CNN(Xin, Yin, x_val, y_val, TrEpochs, patience, n_states, prior)
    if MLmodel == 'lstm':
        ml_models.LSTM(Xin, Yin, x_val, y_val, TrEpochs, patience, n_states, prior)

    print('Train_ml.OSTL: Time taken for training =', proc_time.time() - t1 , "sec")

def AIQD(Xin: str,
        Yin: str,
        x_val: str, 
        y_val: str,
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
        tempNorm: float,
        QDmodelOut: str,
        hyperParam: str,
        OptEpochs: int,
        TrEpochs: int, 
        max_evals: int, 
        patience: int,
        prepInput: str,
        dataPath: str,
        prior: float,
        pinn: str):

    if prepInput == 'True':
        print('train_ml.AIQD: preparing training data for AIQD model')
        prep_input.AIQD(
                        Xin,
                        Yin,
                        systemType, 
                        n_states,
                        time,
                        time_step, 
                        numLogf,
                        LogCa,
                        LogCb, 
                        LogCc, 
                        LogCd,
                        energyNorm,
                        DeltaNorm,
                        gammaNorm, 
                        lambNorm, 
                        tempNorm, 
                        dataPath
                        )

    if hyperParam == 'True':
        print('=================================================================')
        print('Train_ml.AIQD: Going for hyperopt optimization of CNN')
        print('=================================================================')
        t1 = proc_time.time()
        optim.optimize(Xin, Yin, x_val, y_val, OptEpochs, max_evals, n_states, prior, pinn)
        print('Train_ml.AIQD: Time taken for optimization =', proc_time.time() - t1, "sec")
        print('Train_ml.AIQD: Training CNN model with the optimized hyper_parameters')
        print('=================================================================')
    t1 = proc_time.time()
    if MLmodel == 'cnn':
        ml_models.CNN(Xin, Yin, x_val, y_val, TrEpochs, patience, n_states, prior, pinn)
    if MLmodel == 'lstm':
        ml_models.LSTM(Xin, Yin, x_val, y_val, TrEpochs, patience, n_states, prior, pinn)

    print('Train_ml.AIQD: Time taken for training =', proc_time.time() - t1 , "sec")
