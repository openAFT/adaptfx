DOSE_STEP_SIZE = 0.1
SF_STEP_SIZE = 0.01
INF_PENALTY = 1e4
ALPHA_BETA_TUMOR = 10
ALPHA_BETA_OAR = 3
MIN_DOSE = 0
MAX_DOSE = 22.3

FULL_DICT = {'number_of_fractions':None,
        'sparing_factors':None,
        'alpha':None,
        'beta':None,
        'goal':None,
        'C':None,
        'bound_OAR':None,
        'bound_tumor':None,
        'BED_OAR':None,
        'BED_tumor':None,
        'OAR_limit':None,
        'abt':ALPHA_BETA_TUMOR,
        'abn':ALPHA_BETA_OAR,
        'min_dose':MIN_DOSE,
        'max_dose':MAX_DOSE,
        'fixed_prob':0,
        'fixed_mean':0,
        'fixed_std':0,
}

OAR_LIST = ['number_of_fractions',
        'sparing_factors',
        'alpha',
        'beta',
        'goal',
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
        'OAR_limit',
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
        'goal',
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
        'bound_OAR',
        'bound_tumor',
        'abt',
        'abn',
        'min_dose',
        'max_dose',
        'fixed_prob',
        'fixed_mean',
        'fixed_std']

KEY_DICT = {'oar':OAR_LIST, 'tumor':TUMOR_LIST, 'frac':FRAC_LIST, 'tumor_oar':TUMOR_OAR_LIST}
    