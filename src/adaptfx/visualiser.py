# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as normalise
import matplotlib.cm as cm
import adaptfx as afx

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
        label = r'Expected Remaining Number $\varepsilon$'
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
    x_min, x_max, y_min, y_max = np.min(sfs), np.max(sfs), np.min(states), np.max(states)

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
            extent=[x_min, x_max, y_max, y_min])
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

def plot_accumulated_bed(n_list, bed_dict):
    """
    creates plot for simulated adaptive fractionation therapies

    Parameters
    ----------
    n_list : array
        1d array for number of fractions
    accumulated_bed : list of arrays
        list of 1d array for average accumulated bed

    Returns
    -------
    fig : matplotlib.pyplot.figure
        matplotlib pyplot figure
    """
    fig, ax = plt.subplots(1, 1)
    for key in bed_dict:
        ax.plot(n_list, bed_dict[key], label=key)

    fig.supylabel('accumulated BED')
    fig.supxlabel('total number of fractions $n$')
    plt.legend()
    
    return fig

def plot_probability(sf_list, pdf_list, fractions_list):
    """
    creates plot for multiple probability density functions

    Parameters
    ----------
    sf_list : list
        list with sf lists
    pdf_list : list
        list with probability density values list
    fractions_list : list
        list with specified fraction

    Returns
    -------
    fig : matplotlib.pyplot.figure
        matplotlib pyplot figure
    """
    fig, ax = plt.subplots(1, 1)
    for i, fraction in enumerate(fractions_list):
        ax.plot(sf_list[i], pdf_list[i], label=rf'$t={fraction}$')

    fig.supxlabel(r'$\delta$')
    fig.supylabel(r'$PDF(\delta)$')
    fig.tight_layout()
    plt.legend()
    
    return fig


def show_plot():
    plt.show()

def save_plot(basename, *figures):
    if len(figures)==1:
        figures[0].savefig(f'{basename}.pdf', format='pdf')
        plt.clf()
        plt.close()
    else:
        for fig in figures:
            # create name from base, search for untaken name
            fig_name = afx.create_name(basename, 'pdf')
            fig.savefig(fig_name, format='pdf')