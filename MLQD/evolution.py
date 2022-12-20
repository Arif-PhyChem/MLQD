import os 
import re
import sys
import pickle
import random
import numpy as np
import datetime
from pathlib import Path
import ml_dyn as mld
import lic
import plot
import hp
import train_ml as train_ml

fmo_7_init_sites = [1, 6]
fmo_8_init_sites = [1, 6, 8]
QDmodel = ['createQDmodel', 'useQDmodel']
datatype = ['real', 'imag']
class quant_dyn:
    """ Assign values to the parameters"""
    def __init__(self, **param):
        lic.statement()
        if len(param) == 0:
            hp.help()
        else:
            print('=================================================================')
            print("MLQD is started at", datetime.datetime.now())
            print('=================================================================')
            source_path = Path(__file__).resolve()
            self._mlqdDr = source_path.parent
            if param.get('systemType') is not None:
                self._systemType = param.get('systemType')
                if type(self._systemType) != str:
                    raise Exception('systemType should be string')
                print('Setting "systemType" to ' +  str(self._systemType))
            else:
                raise Exception('Please provide system type "systemType = FMO or SB"')
            if param.get('QDmodel') is not None:
                self._QDmodel = param.get('QDmodel')
                if type(self._QDmodel) != str:
                    raise Exception('QDmodel should be string')
                if self._QDmodel not in QDmodel:
                    raise ValueError('QDmodel should be one of these: "createQDmodel", "useQDmodel"')
                print('MLQD is running with the option QDmodel = ', self._QDmodel)
            else:
                self._QDmodel = 'useQDmodel'
                print('The is running with the dafault option QDmodel = ', self._QDmodel)
            if param.get('QDmodelType') is not None:
                self._QDmodelType = param.get('QDmodelType')
                if type(self._QDmodelType) != str:
                    raise Exception('QDmodelType should be string')
                print('Setting ML Model Type "QDmodelType" to ' + str(self._QDmodelType))
            else:
            	self._QDmodelType = 'OSTL'
            	print('As the ML model type is not provided, the ML-QD is running with the \n dafault option QDmodelType = OSTL"')
            if self._QDmodel == 'ceateQDmodel' and self._QDmodelType == 'KRR':
                pass
            else:
                if param.get('n_states') is not None:
                    self._n_states = param.get('n_states')
                    if type(self._n_states) != int:
                        raise Exception(' Number of states "n_states" shoul be integer')
                    if self._systemType == 'SB': 
                        if self._n_states != 2:
                            raise ValueError('In spin-boson, n_states should be equal to 2')
                    print('Setting number of states "n_states" to ' + str(self._n_states))
                else:
                    if self._systemType == 'FMO':
                        self._n_states = 7   # FMO-7
                        print('Running with the default number of states (FMO); n_states: 7, please check if it is the case!')
                    if self._systemType == 'SB':
                        self._n_states = 2
                        print('Running with the default number of states (SB); n_states: 2')
########    ##################################################################################### 
            if self._QDmodel == 'createQDmodel':
                if param.get('prepInput') is not None:
                    self._prepInput = param.get('prepInput')
                    if type(self._prepInput) != str:
                        raise Exception('prepInput should be string')
                    if self._prepInput == 'True':
                        print('Setting option "prepInput" to ' + str(self._prepInput))
                    else:
                        print('You have chosen not to prepare the input files, othewise you should pass "True" to prepInput')
                else:
                    self._prepInput = 'False'
                    print('The is running with default "prepInput" option', self._prepInput)
                if param.get('QDmodelOut') is not None:
                    self._QDmodelOut = param.get('QDmodelOut')
                    if type(self._QDmodelOut) != str:
                        raise Exception('QDmodelOut should be string')
                    print('Model will be saved as ' + str(self._QDmodelOut))
                else:
                    self._QDmodelOut = str(self._QDmodelType) + "_" + "model_for_" + str(self._systemType) + "_" + str(random.random())
                    print('The model will be saved as', self._QDmodelOut)
            else:
                self._prepInput = None
########    ###################################################################################
            if self._QDmodel == 'useQDmodel' or (self._QDmodelType == 'AIQD' and self._QDmodel == 'createQDmodel' and self._prepInput == 'True'):
               if param.get('time') is not None:
                   self._time = param.get('time')
                   print('Setting propagation time "time" to ' + str(self._time))
               else:
                   if self._systemType == 'FMO':
                       self._time = 50 # ps 
                       print('Running with the dafault propagation time; time: 50 ps')
                   elif self._systemType == 'SB':
                       self._time = 20  # a. u.
                       print('Running with the the dafault propagation time:', self._time, '(a.u.)')                
                   else:
                       raise Exception('Provide propagation time using parameter "time"')

               if param.get('time_step') is not None:
                   self._time_step = param.get('time_step')
                   print('Setting time_step to ' + str(self._time_step))
               else:
                   if self._systemType == 'FMO':
                       self._time_step = 5.0 # fs
                       print('Running with the dafault time-step: time_step: 5.0 fs ')
                   elif self._systemType == 'SB':
                       self._time_step = 0.05  # (a.u.)
                       print('Running with the dafault time-step:', self._time_step)
            else:
                self._time = None
                self._time_step = None
########    ######################################################################################
            if self._QDmodel == 'useQDmodel':
                if param.get('QDmodelIn') is not None:
                    self._QDmodelIn = param.get('QDmodelIn')
                    print('Using the trained model "'+ str(self._QDmodelIn) + '" for dynamics prediction')
                else:
                	raise ValueError('please provide a trained model using the option "QDmodelIn"')
                if param.get('QDtrajOut') is not None:
                    self._traj_output_file = param.get('QDtrajOut')
                    if type(self._traj_output_file) != str:
                        raise Exception('QDtrajOut should be string')
                        print('Trajectory will be save as', str(self._traj_output_file))
                else:
                    self._traj_output_file = str(self._systemType) + "_" + str(self._QDmodelType) + "_QD_" + str(random.random()) + ".npy"
                    print('Trajectory will be save as', str(self._traj_output_file))
########    ######################################################################################
            if self._QDmodelType != 'KRR':
                if self._QDmodel == 'useQDmodel':
                    if self._systemType == 'FMO':
                        if param.get('initState') is not None:
                            self._initState = param.get('initState')
                            if type(self._initState) != int:
                                raise Exception('Initial state "initState" shoul be integer')
                            print('Setting initial state "initState" to ' + str(self._initState))
                        else:
                	        self._initState = 1
                	        print('Running with the default value of initial state; initState = 1')
########    ######################################################################################
                if self._QDmodelType == 'OSTL' or self._QDmodelType == 'AIQD':
                    if self._QDmodel == 'useQDmodel':
                        if param.get('XfileIn') is None:
                            if self._systemType == 'SB':
                                if param.get('energyDiff') is not None:
                                    self._energyDiff = param.get('energyDiff')
                                    print('Setting energy difference between two states "energyDiff" to ' + str(self._energyDiff))
                                else:
                	                self._energyDiff = 0.0
                	                print('As energy difference is not provided, the ML-QD is running with the dafault option energyDiff = 0.0')
                                if param.get('Delta') is not None:
                                    self._Delta = param.get('Delta')
                                    print('Setting tunneling matrix element of the two states "Delta" to ' + str(self._Delta))
                                else:
                	                self._Delta = 1.0
                	                print('As tunneling is not provided, the ML-QD is running with the dafault option Delta = 1.0')

                            if param.get('gamma') is not None:
                                self._gamma = param.get('gamma')
                                print('Setting cutt-off frequency "gamma" to ' + str(self._gamma))
                            else:
                                if self._systemType == 'FMO':
                            	    self._gamma = 500
                            	    print('Running with the default value of cutt-off frequency; gamma = 500')
                                if self._systemType == 'SB':
                            	    self._gamma = 10
                            	    print('Running with the default value of cutt-off frequency; gamma = 10')
                            

                            if param.get('lamb') is not None:
                                self._lamb = param.get('lamb')
                                print('Setting system-bath coupling strength "lambda" to ' + str(self._lamb))
                            else:
                                if self._systemType == 'FMO':
                    	            self._lamb = 520
                    	            print('Running with the default value of system-bath coupling strength; lamb = 520')
                                if self._systemType == 'SB':
                    	            self._lamb = 1.0
                    	            print('Running with the default value of system-bath coupling strength; lamb = 0.1')
                            

                            if param.get('temp') is not None:
                                self._temp = param.get('temp')
                                print('Setting temperature (or inverse temperature) value "temp" to ' + str(self._temp))
                            else:
                                if self._systemType == 'FMO':
                    	            self._temp = 510
                    	            print('Running with the default temperature (or inverse temperature) value temp = 510')
                                if self._systemType == 'SB':
                                    self._temp = 1.0
                                    print('Running with the default temperature (or inverse temperature) value temp = 1.0')
                            print('=================================================================')
                            self._name = re.split(r'.hdf5', self._QDmodelIn)[0] + ".pkl"
                            print('Reading normalization constants from', self._name)
                            print('=================================================================')
                            f = open(self._name, 'rb')   # Load normalization parameters
                            norm_param = pickle.load(f)
                            f.close()
                            self._energyNorm = norm_param['energyNorm']
                            self._DeltaNorm = norm_param['DeltaNorm']
                            self._lambNorm = norm_param['lambNorm']
                            self._gammaNorm = norm_param['gammaNorm']
                            self._tempNorm = norm_param['tempNorm']
                            print('Setting energy difference normalizer "energyNorm" to ' + str(self._energyNorm))
                            print('Setting tunneling matrix element normalizer "DeltaNorm" to ' + str(self._DeltaNorm))
                            print('Setting gamma normalizeer "gammaNorm" to ' + str(self._gammaNorm))
                            print('Setting lambda normalizer "lambNormalizer" to ' + str(self._lambNorm))
                            print('Setting temperature (or inverse temperature) normalizer "tempNorm" to ' + str(self._tempNorm))
                    if self._QDmodel == 'createQDmodel':
                        if self._systemType == 'SB':
                            if param.get('energyNorm') is not None:
                                self._energyNorm = param.get('energyNorm')
                                print('Setting energy difference normalizer "energyNorm" to ' + str(self._energyNorm))
                            else:
                        	    self._energyNorm = 1.0
                        	    print('Running with the default value of energy difference normalizer; energyNorm = 1.0 ')
                            if param.get('DeltaNorm') is not None:
                                self._DeltaNorm = param.get('DeltaNorm')
                                print('Setting tunneling matrix element normalizer "DeltaNorm" to ' + str(self._DeltaNorm))
                            else:
                        	    self._DeltaNorm = 1.0
                        	    print('Running with the default value of tunneling matrix element normalizer; DeltaNorm = 1.0 ')
                        else:
                            self._energyNorm = None
                            self._DeltaNorm = None
                        if param.get('gammaNorm') is not None:
                            self._gammaNorm = param.get('gammaNorm')
                            print('Setting gamma normalizeer "gammaNorm" to ' + str(self._gammaNorm))
                        else:
                            if self._systemType == 'FMO':
                        	    self._gammaNorm = 500.0
                        	    print('Running with the default value of gamma normalizeer; gammaNorm = 500 ')
                            if self._systemType == 'SB':
                        	    self._gammaNorm = 10.0
                        	    print('Running with the default value of gamma normalizeer; gammaNorm = 10 ')
                        if param.get('lambNorm') is not None:
                            self._lambNorm = param.get('lambNorm')
                            print('Setting lambda normalizer "lambNormalizer" to ' + str(self._lambNorm))
                        else:
                            if self._systemType == 'FMO':
                        	    self._lambNorm = 510.0
                        	    print('Running with the default value of lambda normalizer; lambNorm = 520 ')
                            if self._systemType == 'SB':
                        	    self._lambNorm = 1.0
                        	    print('Running with the default value of lambda normalizer; lambNorm = 1.0 ')
                        if param.get('tempNorm') is not None:
                            self._tempNorm = param.get('tempNorm')
                            print('Setting temperature (or inverse temperature) normalizer "tempNorm" to ' + str(self._tempNorm))
                        else:
                            if self._systemType == 'FMO':
                        	    self._tempNorm = 510.0
                        	    print('Running with the default value of temperature (or inverse temperature) normalizer; tempNorm = 510 ')
                            if self._systemType == 'SB':
                        	    self._tempNorm = 1.0
                        	    print('Running with the default value of temperature (or inverse temperature) normalizer; tempNorm = 1.0 ')
                    if self._QDmodel == 'useQDmodel':
                        if self._systemType == 'FMO':
                            if self._n_states == 7:
                	            if self._initState not in fmo_7_init_sites:
                	    	        raise ValueError('The initial State initState for FMO should be 1 or 6')
                            if self._n_states == 8:
                                if self._initState not in fmo_8_init_sites:
                                    raise ValueError('The initial State initState for FMO should be 1, 6 or 8')
######################################################################################################
#                                        Run or Train QD
######################################################################################################
            if self._QDmodel == 'useQDmodel':
                if param.get('XfileIn') is not None:
                    X = param.get('XfileIn')
                    if type(X) == str: 
                        if os.path.isfile(X):
                            self._Xin = np.loadtxt(X)
                            self._Xin = np.reshape(self._Xin, (-1, self._Xin.shape[0]))
                            print('Reading from the input file XfileIn "' + str(X) + '" ..........')
                        else:
                            raise ValueError('The input x-file "' + str(X) + '" does not exist')
                    else:
                        self._Xin = X
                        self._Xin = np.array(X)
                        self._Xin = np.reshape(self._Xin, (-1, self._Xin.shape[0]))
                        print('The input is =', X)
                else:
                    if self._QDmodelType == 'KRR':
                        raise ValueError('please the input shot trajectory through XfileIn')
                    if self._QDmodelType == 'OSTL' or self._QDmodelType == 'AIQD': 
                        if self._systemType == 'FMO':
                            X = np.zeros((1,4), dtype = float)
                            X[0,0] = self._initState/10
                            X[0,1] = self._gamma/self._gammaNorm
                            X[0,2] = self._lamb/self._lambNorm
                            X[0,3] = self._temp/self._tempNorm
                        if self._systemType == 'SB':
                            X = np.zeros((1,5), dtype = float)
                            X[0,0] = self._energyDiff/self._energyNorm
                            X[0,1] = self._Delta/self._DeltaNorm
                            X[0,2] = self._gamma/self._gammaNorm
                            X[0,3] = self._lamb/self._lambNorm
                            X[0,4] = self._temp/self._tempNorm
                        self._Xin = X
########    #########################################################################
            if self._QDmodel == 'createQDmodel':
                if param.get('XfileIn') is not None:
                    self._Xin = param.get('XfileIn')
                    if type(self._Xin) != str:
                        raise ValueError('The input XfileIn "' + str(self._Xin) + '" should be string')
                    print('Xfilein is', self._Xin)
                else:
                    self._Xin = "x_data_" + str(random.random()) 
                    print('Setting XfileIn to dafault name', self._Xin)
                if param.get('YfileIn') is not None:
                    self._Yin = param.get('YfileIn')
                    if type(self._Yin) != str: 
                       raise ValueError('The target YfileIn "' + str(self._Yin) + '" should be string')
                    print('YfileIn is', self._Yin)
                else:
                    self._Yin = "y_data_" + str(random.random())
                    print('Setting Yfilein to dafault name', self._Yin)
                if param.get('hyperParam') is not None:
                    self._hyperParam = param.get('hyperParam')
                    if type(self._hyperParam) != str: 
                        raise ValueError('You can only pass string to "hyperParam"')
                    if self._hyperParam == 'True':
                        print('You have chosen to optimize the hyper parameters of the model')
                    else:
                        print('You have chosen not to optimize the hyper parameters of the model, otherwise you should pass "True" to hyperParam')
                else:
                    self._hyperParam = 'False'  # donot optimize
                    print('You have chosen not to optimize the hyper parameters of the model, so it will run with the dafault hyper parameters')
                if self._QDmodelType == 'OSTL' or self._QDmodelType == 'AIQD':
                    if param.get('patience') is not None:
                        self._patience = param.get('patience')
                        if type(self._patience) != int: 
                            raise ValueError('"patience" value can be only integer')
                        print('Setting patience for early stopping to', self._patience)
                    else:
                        self._patience = 20  # for early stopping
                        print('Running with the dafualt value of "patience" =', self._patience)

                    if param.get('epochs') is not None:
                        self._epochs = param.get('epochs')
                        if type(self._epochs) != int: 
                            raise ValueError('"epochs" value can be only integer')
                        print('Setting number of epochs to', self._epochs)
                    else:
                        self._epochs = 100  # for CNN training or optimization 
                        print('Running with the dafualt value of "epochs" =', self._epochs)
                    if self._hyperParam == 'True':
                        if param.get('max_evals') is not None:
                            self._max_evals = param.get('max_evals')
                            if type(self._max_evals) != int: 
                                raise ValueError('"max_evals" value can be only integer')
                            print('Setting maximum number of evaluations to', self._max_evals)
                        else:
                            self._max_evals = 100  # for hyperopt optimization 
                            print('Running with the dafualt value of maximum evaluations "max_evals" =', self._max_evals)
                    else:
                        self._max_evals = None 
                if self._prepInput == 'True':
                    if param.get('dataPath') is not None:
                         self._dataPath = param.get('dataPath')
                         if type(self._dataPath) != str: 
                             raise ValueError('The provided datapath "' + str(self._dataPath) + '" should be string')
                         if os.path.isdir(self._dataPath):
                             pass
                         else:
                             raise ValueError('Datapath "' + self._dataPath +'" does not exist')
                    else:
                        if self._systemType == 'SB':
                            self._dataPath = str(self._mlqdDr) + '/Jupyter_Notebooks/sb_data'
                        if self._systemType == 'FMO':
                            self._dataPath = str(self._mlqdDr) + '/Jupyter_Notebooks/fmo_data'
                else:
                    self._dataPath = None
                if self._QDmodelType == 'KRR':
                    if param.get('xlength') is not None:
                        self._xlength = param.get('xlength')
                        if type(self._xlength) != int: 
                            raise ValueError('length of x-input should be integer')
                        print('Setting length of each row in input file "xlength" to', self._xlength)
                    else:
                        self._xlength = 81
                        print('Setting length of x-input "xlength" to default value', self._xlength)

                    if self._hyperParam != 'True':
                        if param.get('krrSigma') is not None:
                            self._krrSigma = param.get('krrSigma')
                            print('Setting hyperparameter Sigma for Guassian kernel "krrSigma" to ', self._krrSigma)
                        else:
                            self._krrSigma = 4.0 
                            print('Setting hyperparameter Sigma for Gaussian kernel "krrSigma" to default value', self._krrSigma)
                        if param.get('krrLamb') is not None:
                            self._krrLamb = param.get('krrLamb')
                            print('Setting hyperparameter Lambda for KRR "krrLamb" to ', self._krrLamb)
                        else:
                            self._krrLamb = 0.00000000047875 
                            print('Setting hyperparameter Lambda for KRR "krrLamb" to default value', self._krrLamb)
                    else:
                        self._krrSigma = None
                        self._krrLamb = None
            if self._QDmodelType == 'KRR':
                if param.get('dataCol') is not None:
                    self._dataCol = param.get('dataCol')
                    if type(self._dataCol) != int: 
                        raise ValueError('dataCol should be integer')
                        print('is grabbing column #', self._dataCol, 'as was passed through "dataCol"')
                else:
                    self._dataCol = 1
                    print('Setting data column "dataCol" to default value ', self._dataCol)
                if param.get('dtype') is not None:
                    self._dtype = param.get('dtype')
                    if type(self._dtype) != str: 
                        raise ValueError('data type "dtype" should be string')
                    print('data type "dtype" is ', self._dtype)
                    if self._dtype not in datatype:
                        raise ValueError('data type "dtype" should be "real" or "imag"')
                else:
                    self._dtype = 'real'
                    print('Setting data type "dtype" to default type ', self._dtype)
            else: 
                self._dataCol = None
                self._dtype = None
            print('=================================================================')
            
            if self._QDmodelType == 'KRR':
                if self._QDmodel == 'useQDmodel':
                    mld.KRR(
                            self._Xin,
                            self._time,
                            self._time_step,
                            self._QDmodelIn,
                            self._traj_output_file
                            )
                if self._QDmodel == 'createQDmodel':
                    train_ml.KRR(
                                self._Xin,
                                self._Yin,
                                self._QDmodelOut,
                                self._prepInput,
                                self._dataCol,
                                self._xlength,
                                self._dtype,
                                self._dataPath,
                                self._hyperParam,
                                self._krrSigma,
                                self._krrLamb
                                )
            if self._QDmodelType == 'OSTL':
                if self._QDmodel == 'useQDmodel':
                    mld.OSTL(
                    	self._Xin,
                    	self._n_states,
                    	self._time,
                    	self._time_step,
                    	self._QDmodelIn,
                        self._systemType,
                        self._traj_output_file,
                        )
                if self._QDmodel == 'createQDmodel':
                    self._name = self._QDmodelOut + ".pkl"
                    norm_param = {'energyNorm': self._energyNorm, 'DeltaNorm': self._DeltaNorm, 
                                'gammaNorm': self._gammaNorm, 'lambNorm': self._lambNorm, 'tempNorm': self._tempNorm}
                    f = open(self._name, "wb")
                    pickle.dump(norm_param, f)
                    f.close()
                    print('Normalization constants are dumped at', self._name)
                    print('=================================================================')
                    train_ml.OSTL(
                                self._Xin,
                                self._Yin, 
                                self._systemType,
                                self._n_states,
                                self._energyNorm,
                                self._DeltaNorm,
                                self._gammaNorm,
                                self._lambNorm, 
                                self._tempNorm,
                                self._dataPath,
                                self._QDmodelOut,
                                self._hyperParam,
                                self._epochs, 
                                self._max_evals,
                                self._patience,
                                self._prepInput)
            if self._QDmodelType == 'AIQD':
                if self._QDmodel == 'useQDmodel':
                    print('Reading Logistic function parameters from', self._name)
                    self._numLogf = norm_param['numLogf']
                    self._LogCa = norm_param['LogCa']
                    self._LogCb = norm_param['LogCb']
                    self._LogCc = norm_param['LogCc']
                    self._LogCd = norm_param['LogCd']
                    print('Setting number of Logistic functions to', self._numLogf)
                    print('Setting "a" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to ' + str(self._LogCa))
                    print('Setting "b" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to ' + str(self._LogCb))
                    print('Setting "c" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to ' + str(self._LogCc))
                    print('Setting "d" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to ' + str(self._LogCd))
                    print('=================================================================')
                    mld.AIQD(
                    	self._Xin,
                    	self._n_states,
                    	self._time,
                    	self._time_step,
                        self._numLogf, 
                        self._LogCa,
                        self._LogCb,
                        self._LogCc,
                        self._LogCd,
                    	self._QDmodelIn,
                        self._systemType,
                        self._traj_output_file 
                        )
                if self._QDmodel == 'createQDmodel':
                    if param.get('numLogf') is not None:
                        self._numLogf = int(param.get('numLogf'))
                        if type(self._numLogf) !=int:
                            raise ValueError('numLogf should be integer')
                        print('Setting number of Logistic functions to ' + str(self._numLogf))
                    else:
            	        self._numLogf = 1
            	        print('Setting number of Logistic functions to its default value " numLogf = 1 "')
                    if param.get('LogCa') is not None:
                        self._LogCa = int(param.get('LogCa'))
                        print('Setting "a" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to ' + str(self._LogCa))
                    else:
            	        self._LogCa = 1
            	        print('Setting "a" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to its default value "1"')
                    if param.get('LogCb') is not None:
                        self._LogCb = int(param.get('LogCb'))
                        print('Setting "b" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to ' + str(self._LogCb))
                    else:
            	        self._LogCb = 15
            	        print('Setting "b" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to its default value "15"')
                    if param.get('LogCc') is not None:
                        self._LogCc = int(param.get('LogCc'))
                        print('Setting "c" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to ' + str(self._LogCc))
                    else:
                        self._LogCc = -1.0
                        print('Setting "b" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to its default value "-1"')
                    if param.get('LogCd') is not None:
                        self._LogCd = int(param.get('LogCd'))
                        print('Setting "c" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to ' + str(self._LogCd))
                    else:
                        self._LogCd = 1.0
                        print('Setting "d" in Logistic function f(t) = a/(1 + b * np.exp(-(t-c)/d))) to its default value "1"')

                    self._name = self._QDmodelOut + ".pkl"
                    norm_param = {'energyNorm': self._energyNorm, 'DeltaNorm': self._DeltaNorm, 
                            'gammaNorm': self._gammaNorm, 'lambNorm': self._lambNorm, 
                            'tempNorm': self._tempNorm, 'numLogf': self._numLogf, 
                            'LogCa': self._LogCa, 'LogCb': self._LogCb,
                            'LogCc': self._LogCc, 'LogCd': self._LogCd}
                    f = open(self._name, "wb")
                    pickle.dump(norm_param, f)
                    f.close()
                    print('=================================================================')
                    print('Normalization constants and constants for Logistic-function are dumped at', self._name) 
                    print('=================================================================')
                    train_ml.AIQD(
                        self._Xin,
                        self._Yin,
                        self._systemType,
                    	self._n_states,
                    	self._time,
                    	self._time_step,
                        self._numLogf, 
                        self._LogCa,
                        self._LogCb,
                        self._LogCc,
                        self._LogCd,
                        self._energyNorm,
                        self._DeltaNorm,
                        self._gammaNorm,
                        self._lambNorm, 
                        self._tempNorm,
                    	self._QDmodelOut,
                        self._hyperParam,
                        self._epochs,
                        self._max_evals,
                        self._patience,
                        self._prepInput,
                        self._dataPath)

########    ########################################################
#                    parameters for plotting 
########    ########################################################
            print('=================================================================')
            if self._QDmodel == 'useQDmodel':
               if param.get('pltNstates') is not None:
                   self._pltNstates = param.get('pltNstates')
               else:
                   self._pltNstates = self._n_states # Number of states to be plotted 
               if param.get('xlim') is not None:
                   self._xlim = param.get('xlim')
               else:
                   self._xlim = None # plot the full length dynamics
               if param.get('refTraj') is not None:
                   self._refTraj = param.get('refTraj')
                   print('Using the following reference trajectory for plotting:', self._refTraj)
                   plot.plot(self._n_states,
                           self._QDmodelType,
                           self._traj_output_file, 
                           self._refTraj, 
                           self._dataCol, 
                           self._dtype,
                           self._pltNstates,
                           self._xlim)
               else: 
                   print('No reference trajectory was provided, so dynamics is not plotted.',
                           'You can provide reference trajectory with "refTraj"')
            print('=================================================================')
            print("MLQD is ended at", datetime.datetime.now())

