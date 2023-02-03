def loadparam(*args, **param):
    systemType = args[0]
    if param.get('XfileIn') is None:
        if systemType == 'SB':
            if param.get('energyDiff') is not None:
                energyDiff = param.get('energyDiff')
                print('Setting energy difference between two states "energyDiff" to ' + str(energyDiff))
            else:
                energyDiff = 0.0
                print('As energy difference is not provided, the ML-QD is running with the dafault option energyDiff = 0.0')
            if param.get('Delta') is not None:
                Delta = param.get('Delta')
                print('Setting tunneling matrix element of the two states "Delta" to ' + str(Delta))
            else:
                Delta = 1.0
                print('As tunneling is not provided, the ML-QD is running with the dafault option Delta = 1.0')
        else:
            energyDiff = None
            Delta = None
    
        if param.get('gamma') is not None:
            gamma = param.get('gamma')
            print('Setting cutt-off frequency "gamma" to ' + str(gamma))
        else:
            if systemType == 'FMO':
        	    gamma = 500
        	    print('Running with the default value of cutt-off frequency; gamma = 500')
            if systemType == 'SB':
        	    gamma = 10
        	    print('Running with the default value of cutt-off frequency; gamma = 10')
        
        if param.get('lamb') is not None:
            lamb = param.get('lamb')
            print('Setting system-bath coupling strength "lambda" to ' + str(lamb))
        else:
            if systemType == 'FMO':
                lamb = 520
                print('Running with the default value of system-bath coupling strength; lamb = 520')
            if systemType == 'SB':
                lamb = 1.0
                print('Running with the default value of system-bath coupling strength; lamb = 0.1')
        
    
        if param.get('temp') is not None:
            temp = param.get('temp')
            print('Setting temperature (or inverse temperature) value "temp" to ' + str(temp))
        else:
            if systemType == 'FMO':
                temp = 510
                print('Running with the default temperature (or inverse temperature) value temp = 510')
            if systemType == 'SB':
                temp = 1.0
                print('Running with the default temperature (or inverse temperature) value temp = 1.0')
        print('=================================================================')
        print('Reading normalization constants from', name)
        if param.get('QDmodelIn') is not None:
            QDmodelIn = param.get('QDmodelIn')
        else:
            raise ValueError(str(QDmodelIn) + '.pkl does not exist')
        name = re.split(r'.hdf5', QDmodelIn)[0] + ".pkl"
        print('=================================================================')
        f = open(name, 'rb')   # Load normalization parameters
        norm_param = pickle.load(f)
        f.close()
        energyNorm = norm_param['energyNorm']
        DeltaNorm = norm_param['DeltaNorm']
        lambNorm = norm_param['lambNorm']
        gammaNorm = norm_param['gammaNorm']
        tempNorm = norm_param['tempNorm']
        print('Setting energy difference normalizer "energyNorm" to ' + str(energyNorm))
        print('Setting tunneling matrix element normalizer "DeltaNorm" to ' + str(DeltaNorm))
        print('Setting gamma normalizeer "gammaNorm" to ' + str(gammaNorm))
        print('Setting lambda normalizer "lambNormalizer" to ' + str(lambNorm))
        print('Setting temperature (or inverse temperature) normalizer "tempNorm" to ' + str(tempNorm))
    else:
        energyDiff = None
        Delta = None
        gamma = None
        lamb = None
        temp = None
        energyNorm = None
        DeltaNorm = None
        lambNorm = None
        gammaNorm = None
        tempNorm = None

        return energyDiff, Delta, gamma, lamb, temp, energyNorm, DeltaNorm, gammaNorm, lambNorm, tempNorm 
