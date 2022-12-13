# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as normalise
import matplotlib.cm as cm

def plot_val(sfs, states, policies, fractions):
    """
    creates a subplot grid for the policy in each fraction that is given

    Parameters
    ----------
    sfs : array
        1d array with dimension n
    states : array
        1d array with dimension m
    policies : array
        2d array with dimension n_policies:n:m

    Returns
    -------
    fig : matplotlib.pyplot.figure
        matplotlib pyplot figure

    """
    [n_policies, _, _] = policies.shape
    # search for optimal rectangular size of subplot grid
    n_rows = n_columns = int(np.sqrt(n_policies))
    while n_rows * n_columns < n_policies:
        if n_rows >= n_columns:
            n_columns += 1
        else:
            n_rows += 1
    # initiate plot and parameters
    fig, ax = plt.subplots(n_rows, n_columns)
    x_min, x_max, y_min, y_max = sfs[0], sfs[-1], states[0], states[-1]

    # create shared colorbar
    colmin, colmax = np.min(policies), np.max(policies)
    normaliser = normalise(colmin, colmax)
    colormap = cm.get_cmap('turbo')
    im = cm.ScalarMappable(cmap=colormap, norm=normaliser)

    # loop through the axes
    try:
        axs = ax.ravel()
    except:
        # in case ax is a 1x1 subplot
        axs = np.array([ax])
        
    for i, pol in enumerate(policies):
        axs[i].imshow(pol, interpolation=None, origin='upper',
            norm=normaliser, cmap=colormap, aspect='auto',
            extent=[x_min, x_max, y_min, y_max])
        axs[i].set_title(fractions[i])
            
    fig.supxlabel('sparing factor')
    fig.supylabel('remaining BED')
    fig.tight_layout()
    fig.colorbar(mappable=im, ax=axs.tolist())

    return fig

def show_plot():
    plt.show()