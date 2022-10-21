# -*- coding: utf-8 -*-
BEDT_STEPSIZE = 0.1
SF_LOW = 0
SF_HIGH = 1.7
SF_STEPSIZE = 0.01
SF_PROB_THRESHOLD = 1e-5
INF_PENALTY = 1e4

# keys
ALPHA_BETA_TUMOR = 10
ALPHA_BETA_OAR = 3
MIN_DOSE = 0
MAX_DOSE = -1
FIXED_PROB = 0

FULL_DICT = {'number_of_fractions':None,
        'sparing_factors':None,
        'alpha':None,
        'beta':None,
        'tumor_goal':None,
        'oar_limit':None,
        'c':None,
        'abt':ALPHA_BETA_TUMOR,
        'abn':ALPHA_BETA_OAR,
        'min_dose':MIN_DOSE,
        'max_dose':MAX_DOSE,
        'fixed_prob':FIXED_PROB,
        'fixed_mean':None,
        'fixed_std':None
}

SETTING_DICT = {
        'bedt_stepsize': BEDT_STEPSIZE,
        'sf_low': SF_LOW,
        'sf_high': SF_HIGH,
        'sf_stepsize': SF_STEPSIZE,
        'sf_prob_threshold': SF_PROB_THRESHOLD,
        'inf_penalty': INF_PENALTY
}

OAR_LIST = ['number_of_fractions',
        'sparing_factors',
        'alpha',
        'beta',
        'tumor_goal',
        'abt',
        'abn',
        'min_dose',
        'max_dose',
        'fixed_prob',
        'fixed_mean',
        'fixed_std']

TUMOR_LIST = ['number_of_fractions',
        'sparing_factors',
        'alpha',
        'beta',
        'oar_limit',
        'abt',
        'abn',
        'min_dose',
        'max_dose',
        'fixed_prob',
        'fixed_mean',
        'fixed_std']

FRAC_LIST = ['number_of_fractions',
        'sparing_factors',
        'alpha',
        'beta',
        'tumor_goal',
        'c',
        'abt',
        'abn',
        'min_dose',
        'max_dose',
        'fixed_prob',
        'fixed_mean',
        'fixed_std']

TUMOR_OAR_LIST = ['number_of_fractions',
        'sparing_factors',
        'alpha',
        'beta',
        'tumor_goal',
        'oar_limit',
        'abt',
        'abn',
        'min_dose',
        'max_dose',
        'fixed_prob',
        'fixed_mean',
        'fixed_std']

KEY_DICT = {'oar':OAR_LIST, 'tumor':TUMOR_LIST, 'frac':FRAC_LIST, 'tumor_oar':TUMOR_OAR_LIST}
    