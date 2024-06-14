# MLQD- A Python package for Machine Learning-based Quantum Dissipative Dynamics <a name="Top"></a>
In MLQD, we provide three Machine Learning (ML) methods for propagating Qauntum Dissipative Dynamics **[For Licence statement, please [[Click here](#licence)]]**

**The corresponding article is at Computer Physics Communications https://doi.org/10.1016/j.cpc.2023.108940,   arXiv Preprint: http://arxiv.org/abs/2303.01264**

* **Kernel Ridge Regression (KRR)-based recursive (iterative) Quantum Dissipative Dynamics method:** Here is the corresponding article $\boldsymbol{\rightarrow}$ [Speeding up quantum dissipative dynamics of open systems with kernel methods](https://iopscience.iop.org/article/10.1088/1367-2630/ac3261 "Named link title"). Recently, we have performed a comparative study where KKR method outperforms NN models, here is the article $\boldsymbol{\rightarrow}$ [A comparative study of different machine learning methods for dissipative quantum dynamics](https://dx.doi.org/10.1088/2632-2153/ac9a9d "Named link title")
* **AIQD non-recursive  (non-iterative) approach:** Here is the corresponding article $\boldsymbol{\rightarrow}$ [Predicting the future of excitation energy transfer in light-harvesting complex with artificial intelligence-based quantum dynamics](https://doi.org/10.1038/s41467-022-29621-w "Named link title") 
* **The blazingly fast OSTL non-recursive (non-iterative) approach:** Here is the corresponding article $\boldsymbol{\rightarrow}$ [One-Shot Trajectory Learning of Open Quantum Systems Dynamics]( https://doi.org/10.1021/acs.jpclett.2c01242 "Named link title")

*We have also released a database with quantum dissipative dynamics datasets https://doi.org/10.48550/arXiv.2301.12096, you can use it to train your own model*

**MLQD provides**

* Propagation of dynamics with the existing trained models.
* Training convolutional neural networks (CNN) and KRR models on the data.
* Transformation of data into input files X and Y and direct training with out transformation.
* Optimization of the hyperparameters in CNN and KRR models.
* Auto-plotting.

**You can also Run MLQD on the XACS cloud computing platform (https://xacs.xmu.edu.cn/) and the user manual is here (http://mlatom.com/manual/#mlqd)**

### Installation and dependencies

You can install MLQD as a pip package ```pip install mlqd```, however for developers, we provide the source code here (in the ``MLQD`` folder). 


However you want to directly use the up-to-date developers version, you can pull or download the MLQD repo. Before using the source code, we suggest to create an environent, i.e., 

Create a conda environment 

```conda create --name mlqd```

Activate the environment

```conda activate mlqd```

Install the following required dependencies

* tensorflow  ```conda install -c conda-forge tensorflow```

* scikit-learn ```conda install -c anaconda scikit-learn```

* hyperopt ```conda install -c conda-forge hyperopt```

* matplotlib ```conda install -c conda-forge matplotlib```

* MLatom ```pip install MLatom```

In the end, add path to MLQD source code to the system's path

```sys.path.append(path to mlqd source code)```

and then you can import 


```from evolution import quant_dyn```



**Hands-on practice**

For hands-on practice, you can run the Jupyter Notebooks provided in the ``Jupyter Notebooks`` folder. 



## User-Manual

MLQD provides User-Manual and to get to that, we need to import ```quant_dyn``` class from ```evolution.py```

``` from mlqd.evolution import quant_dyn ```

and then call ```quant_dyn``` with out passing any parameters, i.e.,  ``` quant_dyn()``` 


## Licence statement <a name="licence"></a> [[Go to Top](#Top)]

MLQD is a python package developed for Machine Learning-based Quantum Dissipative Dynamics, Version 1.0.0  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;  https://github.com/Arif-PhyChem/MLQD  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;   Copyright (c) 2022 Arif Ullah  
All rights reserved. This package is provided under the Apache Software License 2.0. 

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.  

The software is provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.   
                                         
 **Cite as:**
1) Ullah A. and Dral P. O., Computer Physics Communications, 2023, 294, 108940
2) Ullah A. and Dral P. O., New Journal of Physics, 2021, 23(11), 113019
3) Ullah A. and Dral P. O., Nature Communications, 2022, 13(1), 1930
4) Ullah A. and Dral P. O., Journal of Physical Chemistry Letters, 2022, 13(26), 6037
5) Rodriguez L. E. H.; Ullah A.; Espinosa K. J. R.; Dral P. O. and A. A. Kananenka, Machine Learning: Science and Technology, 2022, 3(4), 045016
6) Ullah, A., Rodriguez, L. E. H., Dral P. O., and Kananenka, A. A., Frontiers in Physics, 2023, 11, 1223973


**Contributers List:**
1) Arif Ullah (main)  
2) Pavlo O. Dral
