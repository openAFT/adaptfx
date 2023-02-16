# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as normalise
import matplotlib.cm as cm

def plot_val(sfs, states, data, fractions, colmap='turbo'):
    """
    creates a subplot grid for the policy in each fraction that is given

    Parameters
    ----------
    sfs : array
        1d array with dimension n
    states : array
        1d array with dimension m
    data : array
        2d array with dimension n_policies:n:m
    fractions : array
        1d array with fractions numbers
    colmap : string
        heat colourmap for matplotlib

    Returns
    -------
    fig : matplotlib.pyplot.figure
        matplotlib pyplot figure

    """
    if colmap == 'turbo':
        label = r'Policy $\pi$ in BED$_{10}$ [Gy]'
    elif colmap == 'viridis':
        label = r'Value $v$'
    elif colmap == 'plasma':
        label = 'Expected Remaining Number'
    else:
        label = 'empty'
    [n_grids, _, _] = data.shape
    # search for optimal rectangular size of subplot grid
    n_rows = n_columns = int(np.sqrt(n_grids))
    while n_rows * n_columns < n_grids:
        if n_rows >= n_columns:
            n_columns += 1
        else:
            n_rows += 1
    # initiate plot and parameters
    fig, ax = plt.subplots(n_rows, n_columns)
    x_min, x_max, y_min, y_max = sfs[0], sfs[-1], states[0], states[-1]

    # create shared colorbar
    colmin, colmax = np.min(data), np.max(data)
    normaliser = normalise(colmin, colmax)
    colormap = cm.get_cmap(colmap)
    im = cm.ScalarMappable(cmap=colormap, norm=normaliser)

    # loop through the axes
    try:
        axs = ax.ravel()
    except:
        # in case ax is a 1x1 subplot
        axs = np.array([ax])
    # turn off axes
    for a in axs:
        a.axis(False)
    
    for i, pol in enumerate(data):
        axs[i].axis(True)
        axs[i].imshow(pol, interpolation=None, origin='upper',
            norm=normaliser, cmap=colormap, aspect='auto',
            extent=[x_min, x_max, y_min, y_max])
        axs[i].set_title(rf'$t = {fractions[i]}$', loc='left')
        try: # get rid of inner axes values
            axs[i].label_outer()
        except:
            pass

    T_text = r'{\mathrm{T}}'
    fig.supxlabel(r'$\delta$')
    fig.supylabel(rf'$B^{T_text}$ [Gy]')
    fig.tight_layout()
    fig.colorbar(mappable=im, ax=axs.tolist(), label=label)

    return fig

def show_plot():
    plt.show()