# MLQD- A Python package for Machine Learning-based Quantum Dissipative Dynamics <a name="Top"></a>
In MLQD, we provide three Machine Learning (ML) methods for propagation Qauntum Dissipative Dynamics 
* **Kernel Ridge Regression (KRR)-based recursive (iterative) Quantum Dissipative Dynamics method:** Here is the corresponding article $\boldsymbol{\rightarrow}$ [Speeding up quantum dissipative dynamics of open systems with kernel methods](https://iopscience.iop.org/article/10.1088/1367-2630/ac3261 "Named link title"). Recently, we have performed a comparative study where KKR method outperforms NN models, here is the article $\boldsymbol{\rightarrow}$ [A comparative study of different machine learning methods for dissipative quantum dynamics](https://dx.doi.org/10.1088/2632-2153/ac9a9d "Named link title")
* **AIQD non-recursive  (non-iterative) approach:** Here is the corresponding article $\boldsymbol{\rightarrow}$ [Predicting the future of excitation energy transfer in light-harvesting complex with artificial intelligence-based quantum dynamics](https://doi.org/10.1038/s41467-022-29621-w "Named link title") 
* **The blazingly fast OSTL non-recursive (non-iterative) approach:** Here is the corresponding article $\boldsymbol{\rightarrow}$ [One-Shot Trajectory Learning of Open Quantum Systems Dynamics]( https://doi.org/10.1021/acs.jpclett.2c01242 "Named link title")

**MLQD provides**

* Propagation of dynamics with the existing trained models [[Click here](#propagation)]
* Training convolutional neural networks (CNN) and KRR models on the data [[Click here](#training)]
* Transformation of data into input files X and Y [[Click here](#preparation)] and direct training with out transformation [[Click here](#nopreparation)]
* Optimization of the hyperparameters in CNN and KRR models [[Click here](#preparation)]
* Auto-plotting

**For Licence statement, please [[Click here](#licence)]**


### Installation and dependencies

Download the GitHub repository and go to Jupyter Notebooks folder to run the notebooks. 
Do not change the name of the MLQD folder as the system will look for the MLQD folder. 

*Some dependencies:* 

Create a conda environment 

```conda create --name mlqd```

Activate the environment

```conda activate mlqd```

Install the following required dependencies

* tensorflow  ```conda install -c conda-forge tensorflow``` 

* hyperopt ```conda install -c conda-forge hyperopt```

* matplotlib ```conda install -c conda-forge matplotlib```

* MLatom ```pip install MLatom```

## User-Manual

MLQD provides User-Manual. To get to that, we need to import ```quant_dyn``` class from ```evolution.py``` from MLQD folder

``` from evolution import quant_dyn ```

and then passing now parameters to quant_dyn will print the manual, i.e.,  ``` quant_dyn()``` 

## Dynamics Propagation <a name="propagation"></a> [[Go to Top](#Top)]
(***Go to examples folder for ready made scripts***)
We provide already trained QD models which can be found here [coming soon], you can download them and test the code. If you want to train your own model, then go to [Training on your own data](#training) section. First of all, we need to import ```quant_dyn``` class from ```evolution.py``` .

``` from evolution import quant_dyn ```

* **KRR model:**

For KRR model, You need to provide the following parameters. Just to emphsize, MLQD is using MLatom package [http://mlatom.com/] for KRR in the backend.
```
        param={ 
        'time': 20,                     # float: Propagation time in picoseconds (ps)  for FMO complex and in (a.u.) for spin-boson model
        'time_step': 0.1,               # float: Time-step for time-propagation (you are restricted to the time-step used in the training data). Default for KRR is 0.1
        'QDmodel': 'useQDmodel',        # str: In MLQD, the dafault option is useQDmodel tells the MLQD to propagate dynamics with an existing trained model
        'QDmodelType': 'KRR',           # str: The type of model we wanna use (KRR, AIQD, or OSTL). Here KRR and the default option is OSTL
        'XfileIn': 'x_input',           # str: Name of a txt file where a short time trajectory (equal to the length the input-model was trained on) is saved. In x-input file, the data should be row wise.  
        'systemType': 'SB',             # str: (Not optional) Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'QDmodelIn': 'KRR_SB_model',    # str: (Not optional for useQDmodel), provide the name of the trained ML model
        'QDtrajOut': 'Qd_trajectory'    # str: (Optional), File name where the trajectory should be saved
        }
```

* **AIQD model:**

For each time-step, the AIQD approach predicts the corresponding reduced density matrix in the following format 
$$\mathcal{R}[\rho_{11}(t)], \mathcal{R}[\rho_{12}(t)], \mathcal{I}[\rho_{12}(t)] \dots, \mathcal{R}[\rho_{1N}(t)], \mathcal{I}[\rho_{1N}(t)], \mathcal{R}[\rho_{22}(t)], \dots, \mathcal{R}[\rho_{2N}(t)], \mathcal{I}[\rho_{2N}(t)], \mathcal{R}[\rho_{33}(t)], \dots, \mathcal{R}[\rho_{3N}(t)],\mathcal{I}[\rho_{3N}(t)],\dots, \dots,   \mathcal{R}[\rho_{NN}(t)]$$

where $N$ is the dimension of the reduced density matrix and $\mathcal{R}$ and $\mathcal{I}$ represent the real and imaginary parts of the off-diagonal terms, respectively.  

I. **Case-1:** If a user wants to provide parameters for propagation in a file, in the shape of an array or in the form of a list. (In this case, the user needs to normalized the data him/herself). AIQD uses a logistic function to normalize the dimension of time, i.e.,  $$f(t) = a/(1 + b \exp(-(t + c)/d))$$ where $a, b, c$ and  $d$ are constants.  Check out the Supplementary Figure 3 of our AIQD papar [Predicting the future of excitation energy transfer in light-harvesting complex with artificial intelligence-based quantum dynamics](https://doi.org/10.1038/s41467-022-29621-w "Named link title") 

```
        param={ 
        'n_states': 2,                          # int:  Number of states (SB) or sites (FMO), default 2 (SB) and 7 (FMO).
        'time': 20,                             # float: Propagation time in picoseconds (ps) for FMO complex and in (a.u.) for spin-boson model
        'time_step': 0.05,                      # float: Time-step for time-propagation (you are not restricted to the time-step used in the training data, however better  stick to that for good accuracy). Default values are 0.1 (KRR SB), 0.05 (AIQD and OSTL for spin-boson model) and 0.005ps for FMO complex.
        'QDmodel': 'useQDmodel',                # str: In MLQD, the dafault option is useQDmodel tells the MLQD to propagate dynamics with an existing trained model
        'QDmodelType': 'AIQD',                  # str: Type of model we wanna use, here AIQD. The default option is OSTL
        'XfileIn': 'x_input',                   # str: Input parameters should be in the same format as the model was trained on. Here "x_input" can be a txt file ('XfileIn': 'x_input'). It can be a list or an array and in this case you need to pass the name of the array or list (XfileIn = x_input). 
        'numLogf': 1,                           # int: Number of Logistic function for the normalization of time dimension. Default value is 1.0.    
        'LogCa' : 1.0,                          # float: Coefficient "a" in the logistic function, default values is 1.0 (you may not provide it)
        'LogCb' : 15.0,                         # float: Coefficient "b" in the logistic function, default values is 15.0 (you may not provide it)
        'LogCc' : -1.0,                         # float: Coefficient "a" in the logistic function, default values is -1.0 (you may not provide it)
        'LogCd' : 1.0,                          # float: Coefficient "d" in the logistic function, default values is 1.0 (you may not provide it)
        'systemType': 'SB',                     # str: (Not optional) Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'QDmodelIn': 'AIQD_SB_model.hdf5',      # str: (Not Optional for useQDmodel), provide the name of the trained ML model
        'QDtrajOut': 'Qd_trajectory'            # str: (Optional), File name where the trajectory should be saved
        }
```
   II. **Case-2:** A user can also just provide simulation parameters (Characteristic frequency, System-bath coupling strengt, Temperature etc.) and MLQD will predict the correspinding dynamics. 

```
        param={ 
        'initState': 1,                         # int:  Initial state with Initial Excitation case (only required in FMO complex case, Default is '1')
        'n_states': 8,                          # Int:  Number of states (SB) or sites (FMO). Default is 2 (SB) and 7 (FMO).
        'time': 50,                             # float: Propagation time in picoseconds (ps)  for FMO complex and in (a.u.) for spin-boson model
        'time_step': 0.005,                     # float: Time-step for time-propagation (you are not restricted to the time-step used in the training data, however better stick to that for good accuracy) Default values are 0.1 (KRR SB), 0.05 (AIQD and OSTL for spin-boson model) and 0.005ps for FMO complex
        'numLogf': 10,                          # int: Number of Logistic function for the normalization of time dimension. Default value is 1.0.   
        'LogCa' : 1.0,                          # float: Coefficient "a" in the logistic function, default values is 1.0 (you may not provide it)
        'LogCb' : 15.0,                         # float: Coefficient "b" in the logistic function, default values is 15.0 (you may not provide it)
        'LogCc' : -1.0,                         # float: Coefficient "a" in the logistic function, default values is -1.0 (you may not provide it)
        'LogCd' : 1.0,                          # float: Coefficient "d" in the logistic function, default values is 1.0 (you may not provide it)
        'gamma': 100,                           # float: Characteristic frequency (in cm^-1 for the provided trained FMO models, in (a.u.) for spin-boson model)
        'lamb': 10,                             # float: System-bath coupling strength  (in cm^-1 for the provided trained FMO models, in (a.u.) for spin-boson model)
        'temp': 300,                            # float: temperature in K  (in Kilven for the provided trained FMO models, in (a.u.) for spin-boson model)
        'gammaNorm': 500,                       # float: Normalizer for Characteristic frequency. Default value is 500 in the case of FMO complex and 10 in the case of spin-boson model. The same values are also adopted in the provided trained models  
        'lambNorm': 520,                        # float: Normalizer for System-bath coupling strength. Default value is 520 (FMO complex) and 1 (SB model). The same values are also adopted in the provided trained models 
        'tempNorm': 500,                        # float: Normalizer for temperature. Default value is 510 (FMO complex) and 1 (SB model). The same values are also adopted in the provided trained models.
      
        'QDmodel': 'useQDmodel',                # str: In MLQD, the dafault option is useQDmodel tells the MLQD to propagate dynamics with an existing trained model
        'QDmodelType': 'AIQD',                  # str: The type of model we wanna use, here AIQD. The default option is OSTL
        'systemType': 'FMO',                    # str: (Not optional)  Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'QDmodelIn': 'AIQD_FMO_model.hdf5',     # str: (Not Optional for useQDmodel), provide the name of the trained ML model
        'QDtrajOut': 'Qd_trajectory'            # str: (Optional), File name where the trajectory should be saved
        }
```

* **OSTL model** (Recommended for fast and smooth propagation of dyanmics)
For OSTL, the input is the same as AIQD except in OSTL, we just don't use the logistic functions here. The OSTL predict the whole dynamics in one shot in the following format $$\boldsymbol{\mathcal{Y}}(t_0), \boldsymbol{\mathcal{Y}}(t_1), \dots, \boldsymbol{\mathcal{Y}}(t_{k-1}), \boldsymbol{\mathcal{Y}}(t_k),  \boldsymbol{\mathcal{Y}}(t_{k+1}), \dots,  \boldsymbol{\mathcal{Y}}(t_M)$$ 
where 
$$\boldsymbol{\mathcal{Y}}(t) = \mathcal{R}[\rho_{11}(t)], \mathcal{R}[\rho_{12}(t)], \mathcal{I}[\rho_{12}(t)] \dots, \mathcal{R}[\rho_{1N}(t)], \mathcal{I}[\rho_{1N}(t)], \mathcal{R}[\rho_{22}(t)], \dots, \mathcal{R}[\rho_{2N}(t)], \mathcal{I}[\rho_{2N}(t)], \mathcal{R}[\rho_{33}(t)], \dots, \mathcal{R}[\rho_{3N}(t)],\mathcal{I}[\rho_{3N}(t)],\dots, \dots,   \mathcal{R}[\rho_{NN}(t)]$$ 

where $N$ is the dimension of the reduced density matrix and $\mathcal{R}$ and $\mathcal{I}$ represent the real and imaginary parts of the off-diagonal terms, respectively.  

   I. **Case-I:**
If a user wants to provide parameters for propagation in a file, in the shape of an array or in the form of a list. (In this case, the user needs to normalized the data him/herself).

```
        param={ 
        'n_states': 2,                          # int:  Number of states (SB) or sites (FMO), default 2 (SB) and 7 (FMO).
        'time': 20,                             # float: Propagation time in picoseconds (ps)  for FMO complex and in (a.u.) for spin-boson model
        'time_step': 0.1,                       # float: Time-step for time-propagation (you are not restricted to the time-step used in the training data, however better  stick to that for good accuracy). Default values are 0.1 (KRR SB), 0.05 (AIQD and OSTL for spin-boson model) and 0.005ps for FMO complex
        'QDmodel': 'useQDmodel',                # str: In MLQD, the dafault option is useQDmodel tells the MLQD to propagate dynamics with an existing trained model
        'QDmodelType': 'AIQD',                  # str: Type of model we wanna use, here AIQD. The default option is OSTL
        'XfileIn': 'x_input',                   # str: Input parameters should be in the same format as the model was trained on. Here "x_input" can be a txt file ('XfileIn': 'x_input'). It can be a list or an array and in this case you need to pass the name of the array or list (XfileIn = x_input). 
        'systemType': 'SB',                     # str: (Not optional) Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'QDmodelIn': 'OSTL_SB_model.hdf5',      # str: (Not Optional for useQDmodel), provide the name of the trained ML model
        'QDtrajOut': 'Qd_trajectory'            # str: (Optional), File name where the trajectory should be saved
        }
```

   II. **Case-2**
A user can also just provide simulation parameters (Characteristic frequency, System-bath coupling strengt, Temperature etc.) and MLQD will predict the correspinding dynamics. 
```
        param={ 
        'n_states': 2,                          # int:  Number of states (SB) or sites (FMO). Default is 2 (SB) and 7 (FMO).
        'time': 20,                             # float: Propagation time in picoseconds (ps)  for FMO complex and in (a.u.) for spin-boson model
        'time_step': 0.05,                      # float: Time-step for time-propagation (OSTL does not use it, however will use it in the output file). Default values are 0.1 (KRR SB), 0.05 (AIQD and OSTL for spin-boson model) and 0.005ps for FMO complex
        'energyDiff': 1.0,                      # float: Energy difference between the two states (in the unit of (a.u.)). Only required in SB model
        'Delta': 1.0,                           # float: The tunneling matrix element (in the unit of (a.u.)). Only required in SB model
        'gamma': 10,                           # float: Characteristic frequency (in cm^-1 for the provided trained FMO models, in (a.u.) for spin-boson model)
        'lamb': 0.1,                             # float: System-bath coupling strength  (in cm^-1 for the provided trained FMO models, in (a.u.) for spin-boson model)
        'temp': 1.0,                            # float: temperature in K  (in Kilven for the provided trained FMO models, in (a.u.) for spin-boson model)
        'energyNorm': 1.0,                      # float: Normalizer for energy difference. Default value is 1.0 (adopted in the provided trained models)
        'DeltaNorm': 1.0,                       # float: Normalizer for Delta. Default value is 1.0 (adopted in the provided trained models)
        'gammaNorm': 10.0,                       # float: Normalizer for Characteristic frequency. Default value is 500 in the case of FMO complex and 10 in the case of spin-boson model. The same values are also adopted in the provided trained models  
        'lambNorm': 1.0,                        # float: Normalizer for System-bath coupling strength. Default value is 520 (FMO complex) and 1 (SB model). The same values are also adopted in the provided trained models 
        'tempNorm': 1.0,                        # float: Normalizer for temperature. Default value is 510 (FMO complex) and 1 (SB model). The same values are also adopted in the provided trained models.
      
        'QDmodel': 'useQDmodel',                # str: In MLQD, the dafault option is useQDmodel tells the MLQD to propagate dynamics with an existing trained model
        'QDmodelType': 'OSTL',                  # str: The type of model we wanna use, here AIQD. The default option is OSTL
        'systemType': 'SB',                     # str: (Not optional)  Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'QDmodelIn': 'OSTL_SB_model.hdf5',      # str: (Not Optional for useQDmodel), provide the name of the trained ML model
        'QDtrajOut': 'Qd_trajectory'            # str: (Optional), File name where the trajectory should be saved 
        }
```

## Model training on your own data <a name="training"></a> [[Go to Top](#Top)]
Here we will show how to train data on you data. If you don't have your own data, you can go to our recently released dataset [QDDSET-1: A Quantum Dissipative Dynamics Dataset](https://github.com/Arif-PhyChem/QDDSET "Named link title") and download the data. If you don't want to train own model and want to use our provided ready made trained models, click here [Coming soon] and how to to propagate dynamics with it, go to [Dynamics Propagation](#propagation) 

### Training a model along with the preparation of training data and optimization of hyperparameters <a name="preparation"></a>

* **KRR** 

 MLQD is using MLatom package [http://mlatom.com/] for KRR in the backend.
For KRR model, You need to provide the following parameters
```
        param={ 
        'QDmodel': 'createQDmodel',     # str: create QD model. The dafault option is useQDmodel
        'QDmodelType': 'KRR',           # str: The type of model. Here KRR and the default option is OSTL
        'prepInput' : 'True',           # str: Prepare input files from the data (Default False)
        'XfileIn': 'x_train',           # str: (Optional, txt file) The prepared X file will be saved at the provided file name 
        'YfileIn': 'y_train',           # str: (Optional, txt file) The prepared Y file will be saved at the provided file name
        'dataPath': 'data/sb' ,         # str: Data path
        'dataCol': 1,                   # int: Default is 1, we may have multiple columns in our data files, mention a single column (KRR model works only for single output)
        'dtype': 'real',                # str: Default is real. If the data in complex and if we pass 'real', it will prepare data only for real part and if we pass 'imag' is mentioned, only imaginary data will be considered. 
        'xlength': 81,                  # int:  Default is 81. Length of the short trajectory which will be used as an input
        'hyperParam': 'True',           # str: Default is False, we can pass True (optimize the hyperparameters) or False (don't optimize and run with the default values)
        'krrSigma': 4.0,                # float: If you pass False to hyperParam, then we need to provide a value for hyperparameter Sigma in Gaussian kernel. Otherwise the model will run with the default value. 
        'krrLamb': 0.00000001,          # float: If you pass False to hyperParam, then we need to provide a value for hyper parameter Lambda in KRR. Otherwise the model will run with the default value.
        'systemType': 'SB',             # str: (Not optional) Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'QDmodelOut': 'KRR_SB_model'    # str: (Optional), providing a name to save the model at
        }
```
* **AIQD**

Just to emphasize, the data files should be in the same format as was adopted in out [QDDSET-1: A Quantum Dissipative Dynamics Dataset](https://github.com/Arif-PhyChem/QDDSET "Named link title")
```
        param={ 
        'n_states': 8,                  # int:  Number of states (SB) or sites (FMO), default 2 (SB) and 7 (FMO).
        'time': 50,                     # float: Propagation time in picoseconds (ps) for FMO complex and in (a.u.) for spin-boson model
        'time_step': 0.005,             # float: Time-step for time-propagation. Default values are 0.05 (spin-boson model) and 0.005ps for FMO complex.
        'QDmodel': 'createQDmodel',     # str: createQDmodel, the dafault option is useQDmodel
        'QDmodelType': 'AIQD',          # str: Type of model. The default option is OSTL
        'prepInput' : 'True',           # str: Prepare input files from the data (Default False)
        'XfileIn': 'x_data',            # str: (Optional, npy file) The prepared X file will be saved at the provided file name 
        'YfileIn': 'y_data',            # str: (Optional, npy file) The prepared Y file will be saved at the provided file name 
        'numLogf': 1,                   # int: Number of Logistic function for the normalization of time dimension. Default value is 1.0.    
        'LogCa' : 1.0,                  # float: Coefficient "a" in the logistic function, default values is 1.0 (you may not provide it)
        'LogCb' : 15.0,                 # float: Coefficient "b" in the logistic function, default values is 15.0 (you may not provide it)
        'LogCc' : -1.0,                 # float: Coefficient "a" in the logistic function, default values is -1.0 (you may not provide it)
        'LogCd' : 1.0,                  # float: Coefficient "d" in the logistic function, default values is 1.0 (you may not provide it)
        'gammaNorm': 500,               # float: Normalizer for Characteristic frequency. Default value is 500 in the case of FMO complex and 10 in the case of spin-boson model. The same values are also adopted in the provided trained models  
        'lambNorm': 520,                # float: Normalizer for System-bath coupling strength. Default value is 520 (FMO complex) and 1 (SB model). The same values are also adopted in the provided trained models 
        'tempNorm': 500,                # float: Normalizer for temperature. Default value is 510 (FMO complex) and 1 (SB model). The same values are also adopted in the provided trained models.
        'systemType': 'FMO',            # str: (Not optional) Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'hyperParam': 'True',           # str: Default is False, we can pass True (optimize the hyperparameters) or False (don't optimize and run with the default structure)
        'patience': 10,                 # int: Patience for early stopping in CNN training 
        'epochs': 100,                  # int: Number of epochs for training or optimization
        'max_evals': 100,               # int: Number of maximum evaluations in hyperopt optimization
        'dataPath': 'data/fmo',         # str: Data path
        'QDmodelOut': 'AIQD_FMO_model'  # str: (Optional), providing a name to save the model at
        }
```
* **OSTL**

Just to emphasize, the data files should be in the same format as was adopted in out [QDDSET-1: A Quantum Dissipative Dynamics Dataset](https://github.com/Arif-PhyChem/QDDSET "Named link title")
```
        param={ 
        'n_states': 8,                  # int:  Number of states (SB) or sites (FMO), default 2 (SB) and 7 (FMO).
        'QDmodel': 'createQDmodel',     # str: createQDmodel, the dafault option is useQDmodel
        'QDmodelType': 'OSTL',          # str: Type of model. The default option is OSTL
        'prepInput' : 'True',           # str: Prepare input files from the data (Default False)
        'XfileIn': 'x_data',            # str: (Optional, npy file) The prepared X file will be saved at the provided file name 
        'YfileIn': 'y_data',            # str: (Optional, npy file) The prepared Y file will be saved at the provided file name 
        'gammaNorm': 500,               # float: Normalizer for Characteristic frequency. Default value is 500 in the case of FMO complex and 10 in the case of spin-boson model. The same values are also adopted in the provided trained models  
        'lambNorm': 520,                # float: Normalizer for System-bath coupling strength. Default value is 520 (FMO complex) and 1 (SB model). The same values are also adopted in the provided trained models 
        'tempNorm': 500,                # float: Normalizer for temperature. Default value is 510 (FMO complex) and 1 (SB model). The same values are also adopted in the provided trained models.
        'systemType': 'FMO',            # str: (Not optional) Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'hyperParam': 'True',           # str: Default is False, we can pass True (optimize the hyperparameters) or False (don't optimize and run with the default structure)
        'patience': 10,                 # int: Patience for early stopping in CNN training
        'epochs': 100,                  # int: Number of epochs for training or optimization
        'max_evals': 100,               # int: Number of maximum evaluations in hyperopt optimization
        'dataPath': 'data/fmo',         # str: Data path
        'QDmodelOut': 'OSTL_FMO_model'  # str: (Optional), providing a name to save the model at
        }
```

### Training a model without preparation of training data <a name="nopreparation"></a> [[Go to Top](#Top)]

Let suppose we already have our prepared training data then 
* **KRR**

For KRR model, You need to provide the following parameters
```
        param={ 
        'QDmodel': 'createQDmodel',     # str: create QD model. The dafault option is useQDmodel
        'QDmodelType': 'KRR',           # str: The type of model. Here KRR and the default option is OSTL
        'XfileIn': 'x_train',           # str: (Not Optional, txt file) The X file 
        'YfileIn': 'y_train',           # str: (Not Optional, txt file) The Y file
        'hyperParam': 'True',           # str: Default is False, we can pass True (optimize the hyperparameters) or False (don't optimize and run with the default values)
        'krrSigma': 4.0,                # float: If you pass False to hyperParam, then we need to provide a value for hyperparameter Sigma in Gaussian kernel. Otherwise the model will run with the default value. 
        'krrLamb': 0.00000001,          # float: If you pass False to hyperParam, then we need to provide a value for hyperparameter Lambda in KRR. Otherwise the model will run with the default value.
        'systemType': 'SB',             # str: (Not optional) Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'QDmodelOut': 'KRR_SB_model'    # str: (Optional), providing a name to save the model at
        }
```

* **AIQD**

Just to emphasize, the data files should be in the same format as was adopted in out [QDDSET-1: A Quantum Dissipative Dynamics Dataset](https://github.com/Arif-PhyChem/QDDSET "Named link title")
```
        param={ 
        'n_states': 8,                  # int:  Number of states (SB) or sites (FMO), default 2 (SB) and 7 (FMO).
        'QDmodel': 'createQDmodel',     # str: createQDmodel, the dafault option is useQDmodel
        'QDmodelType': 'AIQD',          # str: Type of model. The default option is OSTL
        'systemType': 'FMO',            # str: (Not optional) Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'XfileIn': 'x_data',            # str: (Not Optional, npy file) The X file 
        'YfileIn': 'y_data',            # str: (Not Optional, npy file) The Y file  
        'hyperParam': 'True',           # str: Default is False, we can pass True (optimize the hyperparameters) or False (don't optimize and run with the default structure)
        'patience': 10,                 # int: Patience for early stopping in CNN training
        'epochs': 100,                  # int: Number of epochs for training or optimization
        'max_evals': 100,               # int: Number of maximum evaluations in hyperopt optimization
        'QDmodelOut': 'AIQD_FMO_model'  # str: (Optional), providing a name to save the model at
        }
```
* **OSTL**

Just to emphasize, the data files should be in the same format as was adopted in out [QDDSET-1: A Quantum Dissipative Dynamics Dataset](https://github.com/Arif-PhyChem/QDDSET "Named link title")
```
        param={ 
        'n_states': 8,                  # int:  Number of states (SB) or sites (FMO), default 2 (SB) and 7 (FMO).
        'QDmodel': 'createQDmodel',     # str: createQDmodel, the dafault option is useQDmodel
        'QDmodelType': 'OSTL',          # str: Type of model. The default option is OSTL
        'systemType': 'FMO',            # str: (Not optional) Need to define, wether your model is spin-boson (SB) or FMO complex (FMO) 
        'XfileIn': 'x_data',            # str: (Not Optional, npy file) The X file 
        'YfileIn': 'y_data',            # str: (Not Optional, npy file) The X file 
        'hyperParam': 'True',           # str: Default is False, we can pass True (optimize the hyperparameters) or False (don't optimize and run with the default structure)
        'patience': 10,                 # int: Patience for early stopping in CNN training
        'epochs': 100,                  # int: Number of epochs for training or optimization
        'max_evals': 100,               # int: Number of maximum evaluations in hyperopt optimization
        'QDmodelOut': 'OSTL_FMO_model'  # str: (Optional), providing a name to save the model at
        }
```

## Licence statement <a name="licence"></a> [[Go to Top](#Top)]

MLQD is a python package developed for Machine Learning-based Quantum Dissipative Dynamics, Version 1.0.0
                                https://github.com/Arif-PhyChem/MLQD
                                    Copyright (c) 2022 Arif Ullah
All rights reserved. This work is licensed under the Attribution-NonCommercial-NoDerivatives 4.0 International http://creativecommons.org/licenses/by-nc-nd/4.0/) license. See LICENSE.CC-BY-NC-ND-4.0

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

The software is provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties ofmerchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.
                                         
 **Cite as:**
1) Ullah A. and Dral P. O., New Journal of Physics, 2021, 23(11), 113019
2) Ullah A. and Dral P. O., Nature Communications, 2022, 13(1), 1930
3) Ullah A. and Dral P. O., Journal of Physical Chemistry Letters, 2022, 13(26), 6037
4) Rodriguez L. E. H.; Ullah A.; Espinosa K. J. R.; Dral P. O. and A. A. Kananenka, Machine Learning: Science and Technology, 2022, 3(4), 045016"

**Contributers List:**
1) Arif Ullah (main) 
2) Pavlo O. Dral"
