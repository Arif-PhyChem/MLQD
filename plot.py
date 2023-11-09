import os
import re
import numpy as np
import matplotlib.pyplot as plt

def plot(n_states: int, 
	    QDmodelType: int,
        QDtrajOut: str, 
        refTraj: str, 
        dataCol: int, 
        dtype: str,
        pltNstates: int,
        xlim: int):
    QDtrajOut = re.split(r'.npy', QDtrajOut)[0]
    X = str(os.getcwd()) + "/" + str(QDtrajOut) + ".npy" 
    pred = np.load(X)
    ref = np.load(refTraj)
    if QDmodelType == 'KRR':
        if dtype == 'real':
            plt.plot(np.real(pred[:,0]), np.real(pred[:,dataCol]), lw=2.0, label=str(dataCol))
            plt.plot(np.real(ref[:,0]), np.real(ref[:,dataCol]), lw=1.0, ls='--', marker='o', markevery=20, label='_nolegend_')
            plt.legend(bbox_to_anchor=(0.5, 1.1), loc="upper center", ncol=n_states, frameon=False)
        elif dtype == 'imag':
            plt.plot(np.real(pred[:,0]), np.imag(pred[:,dataCol]), lw=2.0, label=str(dataCol))
            plt.plot(np.real(ref[:,0]), np.imag(ref[:,dataCol]), lw=1.0, ls='--', marker='o', label='_nolegend_')
            plt.legend(bbox_to_anchor=(0.5, 1.1), loc="upper center", ncol=n_states, frameon=False)
        plt.xlabel("time")
        if xlim is not None:
            plt.xlim(0, xlim)
        plt.savefig('krr_plot.png', bbox_inches='tight')
        print('Dynamics is plotted and saved as "krr_plot.png". Dash Line with markers is the reference trajectory while the solid line is the predicted dynamics')
        plt.close()
    else:
        a=1
        for i in range(1, pltNstates**2 + n_states, n_states+1):
            idx = str(a) + str(a) 
            plt.plot(np.real(pred[:,0]), np.real(pred[:,i]), lw=2.0, label=idx)
            plt.plot(np.real(ref[:,0]), np.real(ref[:,i]), lw=1.0, ls='--', marker='o',markevery=20, label='_nolegend_')
            plt.legend(bbox_to_anchor=(0.5, 1.1), loc="upper center", ncol=n_states, frameon=False)
            a += 1
        plt.xlabel("time")
        if xlim is not None:
            plt.xlim(0, xlim)
        plt.savefig('population.png', bbox_inches='tight')
        print('The diagonal part (population) is plotted and saved as "population.png". Dash Line with markers is the reference trajectory while the solid line is the predicted dynamics')
        plt.close()
        a = 2; b = n_states
        c = 1;  
        for j in range(0, pltNstates):
            d = c + 1 
            for i in range(a, b + 1): 
                idx = str(c) + str(d)
                plt.plot(np.real(pred[:,0]), np.real(pred[:,i]), lw=2.0, label=idx)
                plt.plot(np.real(ref[:,0]), np.real(ref[:,i]), lw=1.0, ls='--', marker='o', markevery=20, label='_nolegend_')
                plt.legend(bbox_to_anchor=(0.5, 1.1), loc="upper center", ncol=n_states, frameon=False)
                d += 1
            a += n_states + 1
            b += n_states
            c += 1
        plt.xlabel("time")
        if xlim is not None:
            plt.xlim(0, xlim)
        plt.savefig('real_part_coherence.png', bbox_inches='tight')
        print('The real part of the offdiagonal elements is plotted and saved as "real_part_coherence.png". Dash Line with markers is the reference trajectory while the solid line is the predicted dynamics ')
        plt.close()
        a = 2; b = n_states
        c = 1;  
        for j in range(0, pltNstates):
            d = c + 1 
            for i in range(a, b + 1): 
                idx = str(c) + str(d)
                plt.plot(np.real(pred[:,0]), np.imag(pred[:,i]), lw=2.0, label=idx)
                plt.plot(np.real(ref[:,0]), np.imag(ref[:,i]), lw=1.0, ls='--', marker='o',markevery=20, label='_nolegend_')
                plt.legend(bbox_to_anchor=(0.5, 1.1), loc="upper center", ncol=n_states, frameon=False)
                d += 1
            a += n_states + 1
            b += n_states
            c += 1
        plt.xlabel("time")
        if xlim is not None:
            plt.xlim(0, xlim)
        plt.savefig('imag_part_coherence.png', bbox_inches='tight')
        print('The imaginary part of the offdiagonal elements is plotted and saved as "imag_part_coherence.png". Dash Line with markers is the reference trajectory while the solid line is the predicted dynamics ')
        plt.close()

