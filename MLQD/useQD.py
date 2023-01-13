def loadparam(*args, **param):
    systemType = args[0]
    if param.get('XfileIn') is None:
        if systemType == 'SB':
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
        else:
            energyDiff = None
            Delta = None
    
        if param.get('gamma') is not None:
            self._gamma = param.get('gamma')
            print('Setting cutt-off frequency "gamma" to ' + str(self._gamma))
        else:
            if systemType == 'FMO':
        	    self._gamma = 500
        	    print('Running with the default value of cutt-off frequency; gamma = 500')
            if systemType == 'SB':
        	    self._gamma = 10
        	    print('Running with the default value of cutt-off frequency; gamma = 10')
        
        if param.get('lamb') is not None:
            self._lamb = param.get('lamb')
            print('Setting system-bath coupling strength "lambda" to ' + str(self._lamb))
        else:
            if systemType == 'FMO':
                self._lamb = 520
                print('Running with the default value of system-bath coupling strength; lamb = 520')
            if systemType == 'SB':
                self._lamb = 1.0
                print('Running with the default value of system-bath coupling strength; lamb = 0.1')
        
    
        if param.get('temp') is not None:
            self._temp = param.get('temp')
            print('Setting temperature (or inverse temperature) value "temp" to ' + str(self._temp))
        else:
            if systemType == 'FMO':
                self._temp = 510
                print('Running with the default temperature (or inverse temperature) value temp = 510')
            if systemType == 'SB':
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
