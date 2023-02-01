# -*- coding: utf-8 -*-
# log settings
LOG_BOOL = 0
LOG_BOOL_LIST = [0,1]
LOG_LEVEL = 1
LOG_LEVEL_LIST = [0,1,2]

# dose, sf
DOSE_STEPSIZE = 0.1
STATE_STEPSIZE = 1
SF_LOW = 0
SF_HIGH = 1.7
SF_STEPSIZE = 0.01
SF_PROB_THRESHOLD = 1e-5
INF_PENALTY = 1e4

# keys
ALPHA_BETA_TUMOR = 10
ALPHA_BETA_OAR = 3

FULL_DICT = {'number_of_fractions':None,
        'fraction': 0,
        'sparing_factors': None,
        'prob_update': 0,
        'fixed_mean': None,
        'fixed_std': None,
        'shape': None,
        'scale': None,
        'shape_inv': None,
        'scale_inv': None,
        'tumor_goal': None,
        'oar_limit': None,
        'c': None,
        'abt':ALPHA_BETA_TUMOR,
        'abn':ALPHA_BETA_OAR,
        'accumulated_oar_dose': 0,
        'accumulated_tumor_dose': 0,
        'min_dose': 0,
        'max_dose': -1,
        }

SETTING_DICT = {
        'dose_stepsize': DOSE_STEPSIZE,
        'state_stepsize': STATE_STEPSIZE,
        'sf_low': SF_LOW,
        'sf_high': SF_HIGH,
        'sf_stepsize': SF_STEPSIZE,
        'sf_prob_threshold': SF_PROB_THRESHOLD,
        'inf_penalty': INF_PENALTY,
        'plot_policy': 0,
        'plot_values': 0,
        'plot_remains': 0,
        }

STANDARD_LIST = [
        'number_of_fractions',
        'fraction',
        'sparing_factors',
        'prob_update',
        'fixed_mean',
        'fixed_std',
        'shape',
        'scale',
        'shape_inv',
        'scale_inv',
        'abt',
        'abn',
        'accumulated_oar_dose',
        'accumulated_tumor_dose',
        'min_dose',
        'max_dose',
        ]

OAR_LIST = STANDARD_LIST + ['tumor_goal']

TUMOR_LIST = STANDARD_LIST + ['oar_limit']

FRAC_LIST = STANDARD_LIST + ['tumor_goal', 'c']

TUMOR_OAR_LIST = STANDARD_LIST + ['tumor_goal', 'oar_limit']

KEY_DICT = {
        'oar': OAR_LIST, 'oar_old': OAR_LIST, 
        'tumor': TUMOR_LIST, 'tumor_old': TUMOR_LIST,
        'frac': FRAC_LIST, 'frac_old': FRAC_LIST,
        'tumor_oar': TUMOR_OAR_LIST, 'tumor_oar_old': TUMOR_OAR_LIST
        }