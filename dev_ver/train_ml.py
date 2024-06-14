import os
import subprocess
import numpy as np
import cnn as cnn
import hyperopt_optim as optim
import time as proc_time
import tensorflow.keras as keras
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
        n_states: int,
        dataCol: int, 
        xlength: int,
        time: float,
        time_step: float,
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
        print('train_ml.RCDYN: preparing training data for RCDYN model')
        prep_input.RCDYN(Xin,
                        Yin,
                        n_states,
                        dataCol,
                        xlength,
                        time,
                        time_step,
                        dataPath,
                        prior)

    if hyperParam == 'True':
        print('=================================================================')
        print('Train_ml.RCDYN: Going for hyperopt optimization of CNN')
        print('=================================================================')
        t1 = proc_time.time()
        optim.optimize(Xin, Yin, x_val, y_val, OptEpochs, max_evals, n_states, prior, pinn)
        print('Train_ml.RCDYN: Time taken for optimization =', proc_time.time() - t1, "sec")
        print('Train_ml.RCDYN: Training CNN model with the optimized hyper_parameters')
        print('=================================================================')
        t2 = proc_time.time()
        cnn.CNN_optim(Xin, Yin, x_val, y_val, QDmodelOut, TrEpochs, patience, n_states, prior, pinn)
        print('Train_ml.RCDYN: Time taken for training =', proc_time.time() - t2, "sec")
        print('Train_ml.RCDYN: Total Time (optimization + training) =', proc_time.time() - t1 , "sec")
    else:
        optim_param_file = "best_param.pkl"
        print('=================================================================')
        print('Train.ml_RCDYN: Looking for',optim_param_file)
        t1 = proc_time.time()
        if os.path.isfile(optim_param_file):
            print('Train.ml_RCDYN: loading hyperparameters from', optim_param_file)
            print('=================================================================')
            cnn.CNN_optim(Xin, Yin, x_val, y_val, QDmodelOut, TrEpochs, patience, n_states,  prior, pinn)
            print('Train_ml.OSTL: Time taken for training =', proc_time.time() - t1 , "sec")
        else:
            print('=================================================================')
            print('Train.ml_RCDYN: '+ str(optim_param_file) +  ' not found, thus training CNN model with the default structure')
            print('=================================================================')
            cnn.OSTL_default(Xin, Yin, x_val, y_val, QDmodelOut, TrEpochs, patience, n_states, prior, pinn)
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
        print('Train_ml.OSTL: Going for hyperopt optimization of CNN')
        print('=================================================================')
        t1 = proc_time.time()
        optim.optimize(Xin, Yin, x_val, y_val, OptEpochs, max_evals, n_states, prior, pinn)
        print('Train_ml.OSTL: Time taken for optimization =', proc_time.time() - t1, "sec")
        print('Train_ml.OSTL: Training CNN model with the optimized hyper_parameters')
        print('=================================================================')
        t2 = proc_time.time()
        cnn.CNN_optim(Xin, Yin, x_val, y_val, QDmodelOut, TrEpochs, patience, n_states, prior, pinn)
        print('Train_ml.OSTL: Time taken for training =', proc_time.time() - t2, "sec")
        print('Train_ml.OSTL: Total Time (optimization + training) =', proc_time.time() - t1 , "sec")
    else:
        optim_param_file = "best_param.pkl"
        print('=================================================================')
        print('Train.ml_OSTL: Looking for',optim_param_file)
        print('=================================================================')
        t1 = proc_time.time()
        if os.path.isfile(optim_param_file):
            print('Train.ml_OSTL: loading hyperparameters from', optim_param_file)
            cnn.CNN_optim(Xin, Yin, x_val, y_val, QDmodelOut, TrEpochs, patience, n_states, prior, pinn)
            print('Train_ml.OSTL: Time taken for training =', proc_time.time() - t1 , "sec")
        else:
            print('=================================================================')
            print('Train.ml_OSTL: '+ str(optim_param_file) +  ' not found, thus training CNN model with the default structure')
            print('=================================================================')
            cnn.OSTL_default(Xin, Yin, x_val, y_val, QDmodelOut, TrEpochs, patience, n_states, prior, pinn)
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
        print('Train_ml.OSTL: Time taken for optimization =', proc_time.time() - t1, "sec")
        print('Train_ml.AIQD: Training CNN model with the optimized hyper_parameters')
        print('=================================================================')
        t2 = proc_time.time()
        cnn.CNN_optim(Xin, Yin, x_val, y_val, QDmodelOut, TrEpochs, patience, n_states, prior, pinn)
        print('Train_ml.AIQD: Time taken for training =', proc_time.time() - t2, "sec")
        print('Train_ml.AIQD: Total Time (optimization + training) =', proc_time.time() - t1 , "sec")

    else:
        optim_param_file = "best_param.pkl"
        print('=================================================================')
        print('Train.ml_AIQD: Looking for',optim_param_file)
        print('=================================================================')
        t1 = proc_time.time()
        if os.path.isfile(optim_param_file):
            print('Train.ml_AIQD: loading hyperparameters from', optim_param_file)
            cnn.CNN_optim(Xin, Yin, x_val, y_val, QDmodelOut, TrEpochs, patience, n_states, prior, pinn)
            print('Train_ml.AIQD: Time taken for training =', proc_time.time() - t1, "sec")
        else:
            print('=================================================================')
            print('Train.ml_OSTL: '+ str(optim_param_file) +  ' not found, thus training CNN model with the default structure')
            print('=================================================================')
            cnn.AIQD_default(Xin, Yin, x_val, y_val, QDmodelOut, TrEpochs,  patience, n_states, prior, pinn)
            print('Train_ml.AIQD: Time taken for training =', proc_time.time() - t1, "sec")
