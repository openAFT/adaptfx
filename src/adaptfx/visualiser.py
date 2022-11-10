# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def policy_plot(sfs, states, policy_class, n_rows, n_columns, display=True):
    policy = policy_class
    [n_fractions, _, _] = policy.shape
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
        axs[i].set_xlabel('sf')

    fig.tight_layout()
    fig.colorbar(pol, ax=ax)

    if display:
        plt.show()

    return fig