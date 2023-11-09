def loadparam(**param):
    if param.get('systemType') is not None:
        systemType = param.get('systemType')
        if type(systemType) != str:
            raise Exception('systemType should be string')
        print('Setting "systemType" to ' +  str(systemType))
    else:
        raise Exception('Please provide system type "systemType = FMO or SB"')
    if param.get('QDmodel') is not None:
        QDmodel = param.get('QDmodel')
        if type(QDmodel) != str:
            raise Exception('QDmodel should be string')
        if QDmodel not in QDmodel:
            raise ValueError('QDmodel should be one of these: "createQDmodel", "useQDmodel"')
        print('MLQD is running with the option QDmodel = ', QDmodel)
    else:
        QDmodel = 'useQDmodel'
        print('The is running with the dafault option QDmodel = ', QDmodel)
    if param.get('QDmodelType') is not None:
        QDmodelType = param.get('QDmodelType')
        if type(QDmodelType) != str:
            raise Exception('QDmodelType should be string')
        print('Setting ML Model Type "QDmodelType" to ' + str(QDmodelType))
    else:
    	QDmodelType = 'OSTL'
    	print('As the ML model type is not provided, the ML-QD is running with the \n dafault option QDmodelType = OSTL"')
    if QDmodel == 'ceateQDmodel' and QDmodelType == 'KRR':
        n_states = None
    else:
        if param.get('n_states') is not None:
            n_states = param.get('n_states')
            if type(n_states) != int:
                raise Exception(' Number of states "n_states" shoul be integer')
            if param.get('systemType') == 'SB': 
                if n_states != 2:
                    raise ValueError('In spin-boson, n_states should be equal to 2')
            print('Setting number of states "n_states" to ' + str(n_states))
        else:
            if param.get('systemType') == 'FMO':
                n_states = 7   # FMO-7
                print('Running with the default number of states (FMO); n_states: 7, please check if it is the case!')
            if param.get('systemType') == 'SB':
                n_states = 2
                print('Running with the default number of states (SB); n_states: 2')
    return systemType, QDmodel, QDmodelType, n_states
