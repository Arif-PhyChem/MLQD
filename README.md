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
* Spin-boson (SB) model:
```param={ 'time': 20,       # *propagation time in picoseconds (ps)*
        'time_step': 0.1,    # *time-step for time-propagation (you are restricted to the time-step used in the training data)*
        'QDmodel': 'useQDmodel', # *In MLQD, the dafault option is useQDmodel tells the MLQD to propagate dynamics with an existing trained model*
        'MLmodelType': 'OSTL',  # * In MLQD, the default option OSTL
        # 'XfileIn': 'x_input',   # Donot include *.npy extension, not optional, In case of useQDmodel, if you provide input as file, the delimiter (each column should be seperater) should a space.  
        'systemType': 'SB', # not optional  
        'QDmodelIn': 'trained_models/ostl_sb_model-8483-tloss-3.226e-07-vloss-1.040e-06.hdf5',  # not optional for useQDmodel
        }
```



## Model training on your own data <a name="training"></a>
