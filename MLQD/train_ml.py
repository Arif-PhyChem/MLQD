import os
import cnn
import keras
import prep_input
import subprocess
import numpy as np
import time as proc_time
############################################################
def KRR(Xin: str,
        Yin: str,
        QDmodelOut: str,
        prepInput: bool, 
        dataCol: int, 
        xlength: int, 
        dtype: str, 
        dataPath: str,
        hyperParam: bool,
        krrSigma: float,
        krrLamb: float):
    #
    if bool(prepInput) == True:
        print('train_ml.KRR: preparing training data for KRR model')
        prep_input.KRR(Xin,
                    Yin, 
                    dataCol, 
                    xlength, 
                    dtype, 
                    dataPath)
    print('*****************************************')
    print('Train_ml.KRR: Training KRR model with Gaussian kernel using MLatom in the backend ......')
    print('Train_ml.KRR: KRR model will be created and saved as "' + str(QDmodelOut) + '"')
    #
    print('Train_ml.KRR: Checking to see whether the input data files', Xin, 'and', Yin, 'exists')
    if os.path.isfile(Xin) == False:
        raise ValueError("Train_ml.KRR: file name " + str(Xin) + " does not exist")
    if os.path.isfile(Yin) == False:
        raise ValueError("Train_ml.KRR: file name ", str(Xin), " does not exist")
    print('*****************************************')
    ti = proc_time.time()
    QDmodelOut = QDmodelOut + '.unf'
    arg = ['rm', '-f', QDmodelOut + '*']
    subprocess.run(arg, check=True)
    if bool(hyperParam) == True:
        args = ['mlatom', 'createMLmodel', 'MLmodelOut='+ str(QDmodelOut) + ' XfileIn=' + str(Xin), 'Yfile=' + str(Yin), 
                    'kernel=Gaussian', 'sigma=opt', 'lgSigmaL=-25', 'lgSigmaH=25', 'lambda=opt', 'lgLambdaL=-30.0', 'sampling=random']
        with open('kkr_train_output', "w") as output:
            subprocess.run(args, check=True, stdout=output)
    if bool(hyperParam) == False:
        args = ['mlatom', 'createMLmodel', 'MLmodelOut='+ str(QDmodelOut) + ' XfileIn=' + str(Xin), 'Yfile=' + str(Yin), 
                    'kernel=Gaussian', 'sigma='+ str(krrSigma), 'lambda=' + str(krrLamb)]
        with open('kkr_train_output', "w") as output:
            subprocess.run(args, check=True, stdout=output)
    print('Train_ml.KRR: The output of MLatom can be found as "kkr_train_output" file',
    '(please check to ensure that MLatom execution was successful)') 
    print('Train_ml.KRR: Time taken =', proc_time.time() - ti, "sec")

def OSTL(Xin: str,
        Yin: str, 
        systemType: str,
        n_states: int,
        energyNorm: float,
        DeltaNorm: float,
        gammaNorm: float,
        lambNorm: float, 
        tempNorm: float,
        dataPath: str,  # optional in prepInput is False
        QDmodelOut: str,
        hyperParam: bool,
        patience: int,
        prepInput: bool,
        mlqdDr: str):

    if bool(prepInput) == True:
        print('train_ml.OSTL: preparing training data for OSTL model')
        prep_input.OSTL(Xin,
                        Yin,
                        systemType, 
                        n_states,
                        energyNorm,
                        DeltaNorm,
                        gammaNorm, 
                        lambNorm, 
                        tempNorm, 
                        dataPath)

    if bool(hyperParam) == True:
        print('*****************************************')
        print('Train_ml.OSTL: Going for hyperopt optimization of CNN.', 
                'If you are using Jupyter Notebook, you may not see the output of it.')
        print('*****************************************')
        arg = ['cp', str(Xin) + '.npy', 'x_optim.npy']
        subprocess.run(arg, check=True)
        arg = ['cp', str(Yin) + '.npy', 'y_optim.npy']
        subprocess.run(arg, check=True)
        t1 = proc_time.time()
        arg = ['python', str(mlqdDr) + '/hyperopt_optim.py']
        subprocess.run(arg, check=True)
        print('Train_ml.OSTL: Time taken for optimization =', proc_time.time() - t1, "sec")
        arg = ['rm', '-f', '*_optim.npy']
        subprocess.run(arg, check=True)
        print('*****************************************')
        print('Train_ml.OSTL: Training CNN model with the optimized hyper_parameters')
        print('*****************************************')
        t2 = proc_time.time()
        cnn.CNN_optim(Xin, Yin, QDmodelOut, patience)
        print('Train_ml.OSTL: Time taken for training =', proc_time.time() - t2, "sec")
        print('Train_ml.OSTL: Total Time (optimization + training) =', proc_time.time() - t1 , "sec")
    else:
        print('*****************************************')
        print('Train.ml_OSTL: Training CNN model with the default structure')
        print('*****************************************')
        t1 = proc_time.time()
        cnn.OSTL_default(Xin, Yin, QDmodelOut, patience)
        print('Train_ml.OSTL: Time taken for training =', proc_time.time() - t1 , "sec")

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
        tempNorm: float,
        QDmodelOut: str,
        hyperParam: bool,
        patience: int,
        prepInput: bool,
        dataPath: str,
        mlqdDr: str):

    if bool(prepInput) == True:
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

    if bool(hyperParam) == True:
        print('*****************************************')
        print('Train_ml.AIQD: Going for hyperopt optimization of CNN.',
                'If you are using Jupyter Notebook, you may not see the output of it.')
        print('*****************************************')
        arg = ['cp', str(Xin) + '.npy', 'x_optim.npy']
        subprocess.run(arg, check=True)
        arg = ['cp', str(Yin) + '.npy', 'y_optim.npy']
        subprocess.run(arg, check=True)
        t1 = proc_time.time()
        arg = ['python', str(mlqdDr) +  '/hyperopt_optim.py']
        subprocess.run(arg, check=True)
        arg = ['rm', '-f', '*_optim.npy']
        subprocess.run(arg, check=True)
        print('*****************************************')
        print('Train_ml.AIQD: Training CNN model with the optimized hyper_parameters')
        print('*****************************************')
        t2 = proc_time.time()
        cnn.CNN_optim(Xin, Yin, QDmodelOut, patience)
        print('Train_ml.AIQD: Time taken for training =', proc_time.time() - t2, "sec")
        print('Train_ml.AIQD: Total Time (optimization + training) =', proc_time.time() - t1 , "sec")

    else:
        print('*****************************************')
        print('Train.ml_AIQD: Training CNN model with the default structure')
        print('*****************************************')
        t1 = proc_time.time()
        cnn.AIQD_default(Xin, Yin, QDmodelOut, patience)
        print('Train_ml.AIQD: Time taken for training =', proc_time.time() - t1, "sec")
