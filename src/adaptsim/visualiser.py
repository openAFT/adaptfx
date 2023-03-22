# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import Normalize as normalise
import matplotlib.cm as cm

def plot_dose(data, sf_list, n_frac, c_list, oar_array, mean, std, plot_sets=None):
    """
    creates a plot of applied dose and corresponding sparing factor

    Parameters
    ----------
    data : array
        2d array
    sf_list : array
        1d array with dimension n
    n_frac : int

    Returns
    -------
    ax : matplotlib.pyplot.axes

    """
    if plot_sets:
        rcParams.update(plot_sets)

    x = np.arange(1, n_frac+1)
    fig, ax = plt.subplots(1,1)
    # plot the applied policy
    N_text = r'{\mathrm{N}}'
    for i, c_raw in enumerate(c_list):
        oar_bed = np.round(oar_array[i], 1)
        c = np.round(c_raw, 1)
        ax.plot(x, data[i], label=rf'$C={c:.1f}$, $B^{N_text}_{n_frac}={oar_bed}$ Gy',
            alpha=0.5, color='black')
    # plot the sparing factors
    ax2 = ax.twinx()
    ax2.scatter(x, sf_list[1:], label=rf'$\delta_t \sim \mu={mean}, \sigma={std}$',
        marker='^', color='black')
    ax2.invert_yaxis()
    ax2.set_ylabel(r'$\delta$')
    ax.set_ylabel(r'BED$_{10}$ [Gy]')
    ax.set_xlabel(r'Fraction $t$')
    ax.set_xticks(range(min(x), max(x)+1))
    ax.tick_params(axis='x', which='minor', bottom=False)
    lines, labels = ax.get_legend_handles_labels()
    cross, clabels = ax2.get_legend_handles_labels()
    ax2.legend(lines + cross, labels + clabels, loc=0)
    fig.tight_layout()

    return fig

def plot_hist(data, n_frac, plot_sets=None):
    """
    creates a histogram plot of numbers of fractions used

    Parameters
    ----------
    data : array
        1d array
    n_frac : int

    Returns
    -------
    ax : matplotlib.pyplot.axes

    """
    if plot_sets:
        rcParams.update(plot_sets)
    
    x = np.arange(1, n_frac+1)
    fig, ax = plt.subplots(1,1)
    ax.hist(data, bins=x, alpha=0.4, align= 'left',
        histtype= 'stepfilled', color='red')
    ax.set_xticks(range(min(x), max(x)+2))
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.set_ylabel(r'Number of Patients')
    ax.set_xlabel(r'Fraction $t$')
    fig.tight_layout()

    return fig

def plot_val_single(sfs, states, data, fractions, index, label, colmap='turbo', plot_sets=None):
    if plot_sets:
        rcParams.update(plot_sets)
    [n_grids, _, _] = data.shape
    # search for optimal rectangular size of subplot grid
    n_rows = n_columns = int(np.sqrt(n_grids))
    while n_rows * n_columns < n_grids:
        if n_rows >= n_columns:
            n_columns += 1
        else:
            n_rows += 1
    # initiate plot and parameters
    fig, ax = plt.subplots(1, 1)
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

    i = np.where(fractions==index)[0][0]
    axs[0].imshow(data[i], interpolation=None, origin='upper',
        norm=normaliser, cmap=colormap, aspect='auto',
        extent=[x_min, x_max, y_min, y_max])
    T_text = r'{\mathrm{T}}'
    axs[0].set_title(rf'$t = {fractions[i]}$')
    axs[0].set_xlabel(r'$\delta$')
    axs[0].set_ylabel(rf'$B^{T_text}$ [Gy]')

    fig.subplots_adjust(wspace=0.3, hspace=0.3, bottom=0.2, left=0.25, right=0.75)
    fig.colorbar(mappable=im, ax=axs.tolist(), label=label)

    return fig

def plot_val_all(sfs, states, data_full, fractions, label, colmap='turbo', plot_sets=None):
    if plot_sets:
        rcParams.update(plot_sets)
    data = data_full[:-1]
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
    colmin, colmax = np.min(data_full), np.max(data_full)
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
        axs[i].set_title(rf'$t = {fractions[i]}$')
        try: # get rid of inner axes values
            axs[i].label_outer()
        except:
            pass

    T_text = r'{\mathrm{T}}'
    fig.supxlabel(r'$\delta$')
    fig.supylabel(rf'$B^{T_text}$ [Gy]')

    # fig.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.3, bottom=0.13, left=0.15,right=0.92)
    fig.colorbar(mappable=im, ax=axs.tolist(), label=label)

    return fig