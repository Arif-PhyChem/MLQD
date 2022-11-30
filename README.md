# MLQD- A Python package for Machine Learning-based Qauntum Dissipative Dynamics
We provide three Machine Learning (ML) methods in our package MLQD
* **Kernel Ridge Regression (KRR)-based recursive (iterative) Quantum Dissipative Dynamics method:** [Speeding up quantum dissipative dynamics of open systems with kernel methods](https://iopscience.iop.org/article/10.1088/1367-2630/ac3261 "Named link title")  which outperforms NN models as we have shown in this comparative study [A comparative study of different machine learning methods for dissipative quantum dynamics](https://dx.doi.org/10.1088/2632-2153/ac9a9d "Named link title")
* **AIQD non-recursive  (non-iterative) approach:** [Predicting the future of excitation energy transfer in light-harvesting complex with artificial intelligence-based quantum dynamics](https://doi.org/10.1038/s41467-022-29621-w "Named link title") 
* **The blazingly fast OSTL non-recursive (non-iterative) approach:** [One-Shot Trajectory Learning of Open Quantum Systems Dynamics]( https://doi.org/10.1021/acs.jpclett.2c01242 "Named link title")

**MLQD provides**

* Propagation of dynamics with the existing trained models
* Training a convolutional neural networks (CNN) model and KRR model on the data
* Transforming data into input files X and Y
* Optimization of the hyperparameters in CNN and KRR models  

## Dynamics Propagation <a name="propagation"></a>
(***Go to example folder for ready made scripts***)
We provide already trained QD models which can be found here [coming soon], you can download them and test the code. If You wanna train your own model, then go to [Model training on your own data](#training).
First import ```quant_dyn``` class from ```evolution.py``` 
``` from evolution import quant_dyn ```

* **KRR model:**
For KRR model, You need to provide the following parameters
```
        param={ 
        'time': 20,                     # float: Propagation time in picoseconds (ps)
        'time_step': 0.1,               # float: Time-step for time-propagation (you are restricted to the time-step used in the training data)
        'QDmodel': 'useQDmodel',        # string: In MLQD, the dafault option is useQDmodel tells the MLQD to propagate dynamics with an existing trained model
        'MLmodelType': 'OSTL',          # string:  In MLQD, passing the type of model we wanna use (KRR, AIQD, or OSTL). The default option is OSTL
         'XfileIn': 'x_input',          # str (name of a file) or name of an array or list:  A short time trajectory (equal to the length the input-model was                                                 trained on). Here x_input is a txt file where this short-time trajectory is saved. You can also just define a list or an                                             array and pass the name of the array (XfileIn = x_input).  In x-input file, the data should be row wise.  
        'systemType': 'SB',             # str:  (Not optional)  Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'QDmodelIn': 'KRR_SB_model',    # str: (Not optional for useQDmodel), provide the name of the trained ML                                            model
        }
```

* **AIQD model:**
        *Providing input file, an array or a list of input parameters (In this case, the user needs to normalized the data him/her-self.*

```
        param={ 
        'initState': 1,                 # Int:  Initial state with Initial Excitation case (only required in FMO complex case, Default is '1')
        'n_states': 2,                  # Int:  Number of states (SB) or sites (FMO), default 2 (SB) and 7 (FMO).
        'time': 20,                     # float: Propagation time in picoseconds (ps)
        'time_step': 0.1,               # float: Time-step for time-propagation (you are not restricted to the time-step used in the training data, however better                                             stick to that for good accuracy)
        'QDmodel': 'useQDmodel',        # string: In MLQD, the dafault option is useQDmodel tells the MLQD to propagate dynamics with an existing trained model
        'MLmodelType': 'AIQD',          # string:  In MLQD, passing the type of model we wanna use, here AIQD. The default option is OSTL
         'XfileIn': 'x_input',          # str (name of a file) or name of an array or list:  A short time trajectory (equal to the length the input-model was trained on). Here                                                 x_input is a txt file where this short-time trajectory is saved. You can also just define a list or an array and pass the                                           name of the array (XfileIn = x_input).  In x-input file, the data should be row wise.  
        'systemType': 'SB',             # str:  (Not optional)  Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'QDmodelIn': 'KRR_SB_model',    # str:  (Not Optional for useQDmodel), provide the name of the trained ML                                            model
        }
```
*If a user just wants to providing just parameters

```
        param={ 
        'initState': 1,                 # Int:  Initial state with Initial Excitation case (only required in FMO complex case, Default is '1')
        'n_states': 2,                  # Int:  Number of states (SB) or sites (FMO), default 2 (SB) and 7 (FMO).
        'time': 20,                     # float: Propagation time in picoseconds (ps)
        'time_step': 0.1,               # float: Time-step for time-propagation (you are not restricted to the time-step used in the training data, however better                                             stick to that for good accuracy)
        'QDmodel': 'useQDmodel',        # st: In MLQD, the dafault option is useQDmodel tells the MLQD to propagate dynamics with an existing trained model
        'MLmodelType': 'AIQD',          # st:  In MLQD, passing the type of model we wanna use, here AIQD. The default option is OSTL
        'systemType': 'SB',             # str:  (Not optional)  Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'QDmodelIn': 'KRR_SB_model',    # str:  (Not Optional for useQDmodel), provide the name of the trained ML                                            model
        }
```


## Model training on your own data <a name="training"></a>
