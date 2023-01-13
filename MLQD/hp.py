__initState__='initState (int): Initial state with Initial Excitation. It is only required when propagating dynamics with OSTL or AIQD method for FMO complex. Default value is 1'
__n_states__='n_states (int): The number of states or sites. Default values are 2 for spin-boson model and 7 for FMO complex'
__QDmodel__='QDmodel (str): You can pass "createQDmodel" to train a QD model or "useQDmodel" to propagate dynamics with the already trained QD model. Default option is "useQDmodel"'
__systemType__='systemType (str): System type, you can pass "SB" for spin-boson model and "FMO" for FMO complex'
__QDmodelOut__='QDmodelOut (str): MLQD will save the trained QD model with this name. If you don not pass it, MLQD will pick a random name'
__QDmodelIn__='QDmodelIn (str): Pass the name of the trained QD model and MLQD will use it to predict dynamics'
__QDtrajOut__='QDtrajOut (str): MLQD will save the predicted trajectory with this name. If you do not pass a name, MLQD will pick a random name'
__QDmodelType__='QDmodelType (str): Type of QD model. You can pass "KRR" for kernel ridge regression method, "OSTL" for OSTL approach and "AIQD" for AI-QD approach. The default option is OSTL'
__prepInput__='prepInput (str): Choose wether you wanna prepare the input X and Y files from the data. You can pass "True" or "False". Default option is "False". For OSTL and AIQD, your data files should be in the same naming and file format as in out QD3SET-1 dataset' 
__time__='time: (float): Propagation time in picoseconds (ps) for FMO complex and in atomic units (a.u.) for spin-boson model'
__time_step__='time_step (float): Time-step for time-propagation. Default values are 0.05 a.u. (for spin-boson model) and 5 fs for FMO complex'
__hyperParam__='hyperParam (str): Default option is "False". You can pass "True" (to optimize the hyperparameters) or "False" to not optimize the hyperparameters and run with the default structure'
__patience__='patience (int): Patience for early stopping in CNN training'
__OptEpochs__='OptEpochs (int): The number of epochs for optimzation. Default value is 100'
__TrEpochs__='TrEpochs (int): The number of epochs for training. Default value is 100'
__max_evals='max_evals (int): The number of maximum evaluations in hyperopt optimization. Default value is 100'
__XfileIn__='XfileIn (str): The prepared X file will be saved at the provided file name, if not provided, MLQD will use the default name. In the case of KRR-based prediction of dynamics, this option provides the input short-time dynamics'  
__YfileIn__='YfileIn (str): The prepared Y file will be saved at the provided file name. If not provided, MLQD will use the dafault name' 
__datapath__='dataPath (str): MLQD will access data with this path and prepare the X and Y files'
__numLogf__='numLogf (int): The number of Logistic functions for the normalization of time dimension. Default value is 1.0'    
__LogCa__='LogCa (float): Coefficient "a" in the logistic function, default value is 1.0'
__LogCb__='LogCb (float): Coefficient "b" in the logistic function, default value is 15.0'
__LogCc__='LogCc (float): Coefficient "c" in the logistic function, default value is -1.0'
__LogCd__='LogCd (float): Coefficient "d" in the logistic function, default values is 1.0'
__energyDiff__='energyDiff (float): Energy difference between the two states in the case of spin-boson model. Default value is 1.0'
__Delta__='Delta (float): The tunneling matrix element in the case of spin-boson model. Default value is 1.0' 
__gamma__='gamma (int): Characteristic frequency. Default values are 10 (spin-boson model) and 500 (FMO complex)'
__lamb__='lamb (int): System-bath coupling strength or the reorganization energy. Default values are 1 (spin-boson model) and 520 (FMO complex)'
__temp__='temp (int): Temperature or inverse temperature. Default values are 1 (spin-boson model) and 510 (FMO complex)' 
__energyNorm__='energyNorm (float): Normalizer for the energy difference between the states in the case of spin-boson model. Default value is 1.0'
__DeltaNorm__='DeltaNorm (float): Normalizer for the tunneling matrix element in the case of spin-boson model. Default value is 1.0'
__gammaNorm__='gammaNorm (float): Normalizer for Characteristic frequency. Default values are 500 (FMO complex) and 10 (spin-boson model)' 
__lambNorm__='lambNorm (float): Normalizer for System-bath coupling strength. Default values are 520 (FMO complex) and 1 (SB model)' 
__tempNorm__='tempNorm (float): Normalizer for temperature or inverse temperature. Default values are 510 (FMO complex) and 1 (spin-boson model)'
__xlim__='xlim (float): The xaxis limit while plotting' 
__pltNstates__='pltNstates (int): The number of states to be plotted. Default option is to plot all states'
__refTraj__='refTraj (str): MLQD will plot the predicted dynamics against this reference trajectory. If not provided, MLQD will ignore plotting'
_dataCol__='dataCol (int): The column number (counting starts from 0); As KRR is a single output model, and it is possible that you have many columns in your data files, using dataCol option, MLQD is able to grab only the mentioned column and train a KRR model on it. For plotting, the same option is used to grab the concerend column from the reference trajectory'
__dtype__='dtype (str): specify the data type. You can pass "real" or "imag". Default is "real". In KRR training, MLQD will extract the real or imaginary (imag) part of the column passed with "dataCol". For plotting, the same option is used the grab the concerned data type from the column of the reference trajectory'
__xlength__='xlength (int): Length of the input short-time trajectory. It is the number of time steps in the data you pass with the "dataCol". Default value is 81'
__krrSigma__='krrSigma (float): If we pass "False" to hyperParam, then we need to provide a value for the hyperparameter Sigma in Gaussian kernel. Otherwise the model will run with the default value 4.0'
__krrLamb__='krrLamb (float): If we pass "False" to hyperParam, then we need to provide a value for the hyperparameter Lambda in KRR. Otherwise the model will run with the default value 0.00000001'
def help():
    print('\n=================================================================')
    print('                            Manual                               ')
    print('=================================================================')
    i=0
    i+=1; print(str(i) + ']===>',__QDmodel__)
    i+=1; print(str(i) + ']===>',__systemType__)
    i+=1; print(str(i) + ']===>',__QDmodelOut__)
    i+=1; print(str(i) + ']===>',__QDmodelIn__)
    i+=1; print(str(i) + ']===>',__QDtrajOut__)
    i+=1; print(str(i) + ']===>',__QDmodelType__)
    i+=1; print(str(i) + ']===>',__prepInput__)
    i+=1; print(str(i) + ']===>',__time__)
    i+=1; print(str(i) + ']===>',__time_step__)
    i+=1; print(str(i) + ']===>', __hyperParam__)
    i+=1; print(str(i) + ']===>', __XfileIn__)
    i+=1; print(str(i) + ']===>', __YfileIn__)
    i+=1; print(str(i) + ']===>', __datapath__)
    print('\n------------------------------------------------------------------')
    print('                Specific to AIQD and OSTL approach                ')
    print('------------------------------------------------------------------')
    i+=1; print(str(i) + ']===>', __initState__)
    i+=1; print(str(i) + ']===>', __n_states__)
    i+=1; print(str(i) + ']===>', __patience__)
    i+=1; print(str(i) + ']===>', __OptEpochs__)
    i+=1; print(str(i) + ']===>', __TrEpochs__)
    i+=1; print(str(i) + ']===>', __max_evals)
    i+=1; print(str(i) + ']===>', __energyDiff__)
    i+=1; print(str(i) + ']===>', __Delta__)
    i+=1; print(str(i) + ']===>', __gamma__)
    i+=1; print(str(i) + ']===>', __lamb__) 
    i+=1; print(str(i) + ']===>', __temp__) 
    i+=1; print(str(i) + ']===>',__energyNorm__)
    i+=1; print(str(i) + ']===>',__DeltaNorm__)
    i+=1; print(str(i) + ']===>', __gammaNorm__)
    i+=1; print(str(i) + ']===>', __lambNorm__)
    i+=1; print(str(i) + ']===>', __tempNorm__)
    print('\n------------------------------------------------------------------')
    print('                     Specific to AIQD approach                    ')
    print('------------------------------------------------------------------')
    i+=1; print(str(i) + ']===>', __numLogf__)
    i+=1; print(str(i) + ']===>', __LogCa__)
    i+=1; print(str(i) + ']===>', __LogCb__)
    i+=1; print(str(i) + ']===>', __LogCc__)
    i+=1; print(str(i) + ']===>', __LogCd__)
    print('\n------------------------------------------------------------------')
    print('                    Specific to KRR approach        ')
    print('------------------------------------------------------------------')
    i+=1; print(str(i) + ']===>',__xlength__)
    i+=1; print(str(i) + ']===>',__krrSigma__)
    i+=1; print(str(i) + ']===>',__krrLamb__)
    print('\n------------------------------------------------------------------')
    print('                 Specific to KRR approach and Potting  ')
    print('------------------------------------------------------------------')
    i+=1; print(str(i) + ']===>', _dataCol__)
    i+=1; print(str(i) + ']===>', __dtype__)
    print('\n------------------------------------------------------------------')
    print('                        Specific to Plotting         ')
    print('------------------------------------------------------------------')
    i+=1; print(str(i) + ']===>', __xlim__) 
    i+=1; print(str(i) + ']===>', __pltNstates__)
    i+=1; print(str(i) + ']===>', __refTraj__)
    print('=================================================================')

