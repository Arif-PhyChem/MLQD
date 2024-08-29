import os
import random 

def loadparam(*args, **param):
    mlqdDr = args[0]
    systemType = args[1]
    QDmodelType = args[2]
    if param.get('prepInput') is not None:
        prepInput = param.get('prepInput')
        if type(prepInput) != str:
            raise Exception('prepInput should be string')
        if prepInput == 'True':
            print('Setting option "prepInput" to ' + str(prepInput))
        else:
            print('You have chosen not to prepare the input files, othewise you should pass "True" to prepInput')
    else:
        prepInput = 'False'
        print('The is running with default "prepInput" option', prepInput)
    
    if param.get('MLmodel') is not None:
        MLmodel = param.get('MLmodel')
        if type(MLmodel) != str:
            raise Exception('MLmodel should be string')
        print('Setting "MLmodel" to ' +  str(MLmodel))
    else:
        MLmodel = 'cnn'
        print('Setting "MLmodel" to default option ' +  str(MLmodel))

    if param.get('XfileIn') is not None:
        Xin = param.get('XfileIn')
        if type(Xin) != str:
            raise ValueError('The input XfileIn "' + str(Xin) + '" should be string')
        print('Xfilein is', Xin)
    else:
        Xin = "x_data_" + str(random.random()) 
        print('Setting XfileIn to dafault name', Xin)
    if param.get('YfileIn') is not None:
        Yin = param.get('YfileIn')
        if type(Yin) != str: 
           raise ValueError('The target YfileIn "' + str(Yin) + '" should be string')
        print('YfileIn is', Yin)
    else:
        Yin = "y_data_" + str(random.random())
        print('Setting Yfilein to dafault name', Yin)
    
    if QDmodelType != 'KRR':
        if param.get('XvalIn') is not None:
            Xval = param.get('XvalIn')
            print('X file for validation to be used is ' + str(Xval))
        else:
            Xval = None
        if param.get('YvalIn') is not None:
            Yval = param.get('YvalIn')
            print('Y file for validation to be used is ' + str(Yval))
        else:
            Yval = None
    else:
        Xval = None
        Yval = None

    if param.get('hyperParam') is not None:
        hyperParam = param.get('hyperParam')
        if type(hyperParam) != str: 
            raise ValueError('You can only pass string to "hyperParam"')
        if hyperParam == 'True':
            print('You have chosen to optimize the hyper parameters of the model')
        else:
            print('You have chosen not to optimize the hyper parameters of the model, otherwise you should pass "True" to hyperParam')
    else:
        hyperParam = 'False'  # donot optimize
        print('You have chosen not to optimize the hyper parameters of the model, so it will run with the dafault hyper parameters')
    print('=================================================================')
    if QDmodelType != 'KRR':
        if param.get('patience') is not None:
            patience = param.get('patience')
            if type(patience) != int: 
                raise ValueError('"patience" value can be only integer')
            print('Setting patience for early stopping to', patience)
        else:
            patience = 20  # for early stopping
            print('Running with the dafualt value of "patience" =', patience)
        if param.get('TrEpochs') is not None:
            TrEpochs = param.get('TrEpochs')
            if type(TrEpochs) != int: 
                raise ValueError('"TrEpochs" value can be only integer')
            print('Setting number of epochs for training to', TrEpochs)
        else:
            TrEpochs = 100  # for CNN training or optimization 
            print('Running with the dafualt value of "Training epochs" =', TrEpochs)
        
        if hyperParam == 'True':
            if param.get('OptEpochs') is not None:
                OptEpochs = param.get('OptEpochs')
                if type(OptEpochs) != int: 
                    raise ValueError('"OptEpochs" value can be only integer')
                print('Setting number of epochs in hyperopt optimization to', OptEpochs)
            else:
                OptEpochs = 100  # for CNN training or optimization 
                print('Running with the dafualt value of "Optimization epochs" =', OptEpochs)
            if param.get('max_evals') is not None:
                max_evals = param.get('max_evals')
                if type(max_evals) != int: 
                    raise ValueError('"max_evals" value can be only integer')
                print('Setting maximum number of evaluations to', max_evals)
            else:
                max_evals = 100  # for hyperopt optimization 
                print('Running with the dafualt value of maximum evaluations "max_evals" =', max_evals)
        else:
            OptEpochs = None
            max_evals = None

    else:
        patience = None
        OptEpochs = None
        TrEpochs = None
        max_evals = None

    if QDmodelType == 'RCDYN' or QDmodelType == 'OSTL' or QDmodelType == 'AIQD':
        if QDmodelType != 'RCDYN': 
            if systemType == 'SB':
                if param.get('energyNorm') is not None:
                    energyNorm = param.get('energyNorm')
                    print('Setting energy difference normalizer "energyNorm" to ' + str(energyNorm))
                else:
            	    energyNorm = 1.0
            	    print('Running with the default value of energy difference normalizer; energyNorm = 1.0 ')
                if param.get('DeltaNorm') is not None:
                    DeltaNorm = param.get('DeltaNorm')
                    print('Setting tunneling matrix element normalizer "DeltaNorm" to ' + str(DeltaNorm))
                else:
            	    DeltaNorm = 1.0
            	    print('Running with the default value of tunneling matrix element normalizer; DeltaNorm = 1.0 ')
            else:
                energyNorm = None
                DeltaNorm = None
        else:
            energyNorm = None
            DeltaNorm = None
        if param.get('gammaNorm') is not None:
            gammaNorm = param.get('gammaNorm')
            print('Setting gamma normalizeer "gammaNorm" to ' + str(gammaNorm))
        else:
            if systemType == 'SB':
        	    gammaNorm = 10.0
        	    print('Running with the default value of gamma normalizeer; gammaNorm = 10 ')
            else:
                gammaNorm = 500.0
                print('Running with the default value of gamma normalizeer; gammaNorm = 500 ')
        if param.get('lambNorm') is not None:
            lambNorm = param.get('lambNorm')
            print('Setting lambda normalizer "lambNorm" to ' + str(lambNorm))
        else:
            if systemType == 'SB':
        	    lambNorm = 1.0
        	    print('Running with the default value of lambda normalizer; lambNorm = 1.0 ')
            else:
        	    lambNorm = 510.0
        	    print('Running with the default value of lambda normalizer; lambNorm = 520 ')
        if param.get('tempNorm') is not None:
            tempNorm = param.get('tempNorm')
            print('Setting temperature (or inverse temperature) normalizer "tempNorm" to ' + str(tempNorm))
        else:
            if systemType == 'SB':
        	    tempNorm = 1.0
        	    print('Running with the default value of temperature (or inverse temperature) normalizer; tempNorm = 1.0 ')
            else:
        	    tempNorm = 510.0
        	    print('Running with the default value of temperature (or inverse temperature) normalizer; tempNorm = 510 ')
        print('=================================================================')
    else:
        energyNorm = None
        DeltaNorm = None
        gammaNorm = None
        lambNorm = None
        tempNorm = None
    if prepInput == 'True':
        if param.get('dataPath') is not None:
             dataPath = param.get('dataPath')
             if type(dataPath) != str: 
                 raise ValueError('The provided datapath "' + str(dataPath) + '" should be string')
             if os.path.isdir(dataPath):
                 pass
             else:
                 raise ValueError('Datapath "' + dataPath +'" does not exist')
        else:
            raise ValueError('Please provide Datapath using keyword "dataPath"')
    else:
        dataPath = None
###############################################################################
#                                 Specific to KRR
###############################################################################
    if (QDmodelType == 'KRR') or (QDmodelType == 'RCDYN'):
        if param.get('xlength') is not None:
            xlength = param.get('xlength')
            if type(xlength) != int: 
                raise ValueError('length of x-input should be integer')
            print('Setting length of each row in input file "xlength" to', xlength)
        else:
            xlength = 81
            print('Setting length of x-input "xlength" to default value', xlength)
    else:
        xlength = None
    if (QDmodelType == 'KRR'):
        if hyperParam != 'True':
            if param.get('krrSigma') is not None:
                krrSigma = param.get('krrSigma')
                print('Setting hyperparameter Sigma for Guassian kernel "krrSigma" to ', krrSigma)
            else:
                krrSigma = 4.0 
                print('Setting hyperparameter Sigma for Gaussian kernel "krrSigma" to default value', krrSigma)
            if param.get('krrLamb') is not None:
                krrLamb = param.get('krrLamb')
                print('Setting hyperparameter Lambda for KRR "krrLamb" to ', krrLamb)
            else:
                krrLamb = 0.00000000047875 
                print('Setting hyperparameter Lambda for KRR "krrLamb" to default value', krrLamb)
        else:
            krrSigma = None
            krrLamb = None
        print('=================================================================')
    else:
        krrSigma = None
        krrLamb = None
    if QDmodelType == 'AIQD':
        if param.get('numLogf') is not None:
            numLogf = int(param.get('numLogf'))
            if type(numLogf) !=int:
                raise ValueError('numLogf should be integer')
            print('Setting number of Logistic functions to ' + str(numLogf))
        else:
            numLogf = 1
            print('Setting number of Logistic functions to its default value " numLogf = 1 "')
        if param.get('LogCa') is not None:
            LogCa = float(param.get('LogCa'))
            print('Setting "a" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to ' + str(LogCa))
        else:
            LogCa = 1
            print('Setting "a" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to its default value "1"')
        if param.get('LogCb') is not None:
            LogCb = float(param.get('LogCb'))
            print('Setting "b" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to ' + str(LogCb))
        else:
            LogCb = 15
            print('Setting "b" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to its default value "15"')
        if param.get('LogCc') is not None:
            LogCc = float(param.get('LogCc'))
            print('Setting "c" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to ' + str(LogCc))
        else:
            LogCc = -1.0
            print('Setting "b" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to its default value "-1"')
        if param.get('LogCd') is not None:
            LogCd = float(param.get('LogCd'))
            print('Setting "c" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to ' + str(LogCd))
        else:
            LogCd = 1.0
            print('Setting "d" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to its default value "1"')
    else:
        numLogf = None
        LogCa = None
        LogCb = None
        LogCc = None
        LogCd = None

    return MLmodel, prepInput, energyNorm, DeltaNorm, gammaNorm, lambNorm, tempNorm, Xin, Yin, Xval, Yval, hyperParam, patience, OptEpochs, TrEpochs, max_evals, dataPath, xlength, krrSigma, krrLamb, numLogf, LogCa, LogCb, LogCc, LogCd 
