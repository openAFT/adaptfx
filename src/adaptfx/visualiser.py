# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def policy_plot(sfs, states, policy_class, plot=False):
    policy = policy_class
    [n_fractions, _, _] = policy.shape
    n_rows = n_columns = int(np.sqrt(n_fractions))
    while n_rows * n_columns < n_fractions:
        if n_rows >= n_columns:
            n_columns += 1
        else:
            n_rows += 1
    fig, ax = plt.subplots(n_rows, n_columns)
    colmin = np.min(policy)
    colmax = np.max(policy)

    axs = ax.ravel()

    for i in range(0, n_fractions):
        pol = axs[i].imshow(policy[i], interpolation=None, origin='upper',
            vmin=colmin, vmax=colmax, cmap='jet', aspect='auto',
            extent=[sfs[0], sfs[-1], states[0], states[-1]])
        #     policy = ax.imshow(z[fraction][state], cmap='jet', aspect='auto', 
        #               extent=[x0,x1,y1,y0])
        # axs[i].set_ylabel('state')
    fig.supxlabel('sparing factor')
    fig.supylabel('remaining BED')

    fig.tight_layout()
    fig.colorbar(pol, ax=ax)
    
    if plot:
        plt.show()

    return fig