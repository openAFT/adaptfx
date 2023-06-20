from matplotlib import cycler

ALL_SIM_DICT = {'n_patients': 0,            # histogram
                'fixed_mean_sample': 0,     # histogram
                'fixed_std_sample': 0,      # histogram
                'c_list': 0,                # fraction
                'plot_index': 1,            # single state
                'data_filepath': './',      # data
                'data_selection': [],       # data
                'data_row_hue': 0,          # data
                'figsize': [6,4],           # settings
                'fontsize': 14,             # settings
                'save': 0,                  # settings
                'usetex': 0                 # settings
                }

KEY_DICT_SIM = {'sim': list(ALL_SIM_DICT)}

plot_line_cycle = cycler('linestyle', ['-', '--', ':', '-.'])

RCPARAMS = {'figure.figsize'        : [6,4],
            'font.size'             : 14,
            'font.family'           : 'serif',
            'legend.handlelength'   : 1.4,
            'legend.fontsize'       : 11,
            'legend.title_fontsize' : 11,
            'text.usetex'           : False,
            'axes.labelpad'         : 10,
            'axes.linewidth'        : 1.1,
            'axes.xmargin'          : 0.05,
            'axes.prop_cycle'       : plot_line_cycle,
            'axes.autolimit_mode'   : 'data',
            'axes.titlelocation'    : 'left',
            'xtick.major.size'      : 7,
            'xtick.minor.size'      : 3.5,
            'xtick.major.width'     : 1.1,
            'xtick.minor.width'     : 1.1,
            'xtick.major.pad'       : 5,
            'xtick.minor.visible'   : True,
            'ytick.major.size'      : 7,
            'ytick.minor.size'      : 3.5,
            'ytick.major.width'     : 1.1,
            'ytick.minor.width'     : 1.1,
            'ytick.major.pad'       : 5,
            'ytick.minor.visible'   : True,
            'lines.markersize'      : 7,
            'lines.markerfacecolor' : 'none',
            'lines.markeredgewidth' : 0.8}