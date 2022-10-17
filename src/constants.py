# -*- coding: utf-8 -*-
DOSE_STEP_SIZE = 0.1
SF_STEPSIZE = 0.01
SF_LOW = 0 + SF_STEPSIZE
SF_HIGH = 1.7 + SF_STEPSIZE
SF_PROB = 1e-5
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
        'C':None,
        'abt':ALPHA_BETA_TUMOR,
        'abn':ALPHA_BETA_OAR,
        'min_dose':MIN_DOSE,
        'max_dose':MAX_DOSE,
        'fixed_prob':FIXED_PROB,
        'fixed_mean':None,
        'fixed_std':None,
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
        'C',
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
    