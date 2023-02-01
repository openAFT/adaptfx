# The `adaptfx` manual

## Working tree

One should perform calculations in a working folder e.g. `adaptfx/work`. There the instruction files can be specified by the user and `adaptfx` will automatically produce log files if the user wishes to.

## Format of the instruction file
The user specifies following elements of the dictionary for the main entries: 
- `"algorithm"`
- `"debug"`
- `"log"`
- `"keys"`
- `"settings"`

Each entry of the dictionary is either a parameter or a dictionary itself. In the next section explained are the main entries.

## Main entries

```
algorithm: frac, oar, tumor, tumor_oar
    type of algorithm
    frac : minimise number of fractions
    oar: minimise oar BED, with tumor constraint
    tumor : maximise tumor BED, with oar constraint
    tumor_oar : minimise oar and maximise tumor BED simultaneously
level: 0, 1, 2
    quiet mode, 0 (for scripting)
    normal mode, 1
    debug mode, 2 (for developing)
    default: 1
log: 0,1
    log output to a file
    default: 0
keys: dict
    algorithm instruction
settings: dict
    algorithm settings
```

## Optimisation parameters
```
keys
----------------
number_of_fractions : integer
    number of fractions that will be delivered.
fraction : int
    if only a single fraction should be calculated.
    default: 0
sparing_factors : list/array
    list/array with all observed sparing factors.
prob_update : int
    set type of updating for probability distribution of sparing factor.
    If set to 0, the sparing factor is not updated and the probabability
    distribution fixed, specified with 'fixed_mean' and 'fixed_std'.
    If set to 1, then the assumed probability distribution is a normal
    distribution which is updated with maximum a posteriori estimation. 
    The prior is a gamma distribution with hyperparameters 'scale' and 'shape'.
    If set to 2, full Bayesian approach is employed, where a posterior
    predictive distribution is estimated. The conjugate prior is an
    inverse-gamma distribution with hyperparameters 'scale_inv' and 'shape_inv'.
    default: 0
fixed_mean: float
    mean of the fixed sparing factor normal distribution.
    mandatory if 'prob_update' set to 0.
fixed_std: float
    standard deviation of the fixed sparing factor normal distribution.
    mandatory if 'prob_update' set to 0.
shape : float
    shape of gamma distribution, for prior.
    mandatory if 'prob_update' set to 1.
scale : float
    scale of gamma distribution, for prior.
    mandatory if 'prob_update' set to 1.
shape_inv : float
    shape of inverse-gamma distribution, for prior.
    mandatory if 'prop_update' set to 2.
scale_inv : float
    scale of inverse-gamme distribution, for prior.
    mandatory if 'prob_update' set to 2.
abt : float
    alpha-beta ratio of tumor.
    default: 10
abn : float
    alpha-beta ratio of OAR.
    default: 3
accumulated_oar_dose : float
    accumulated OAR BED (from previous fractions).
accumulated_tumor_dose : float
    accumulated tumor BED (from previous fractions).
min_dose : float
    minimal physical doses to be delivered in one fraction.
    The doses are aimed at PTV 95.
    default: 0
max_dose : float
    maximal physical doses to be delivered in one fraction.
    The doses are aimed at PTV 95. If -1 the dose is adapted to the
    remaining dose tumor dose to be delivered or the remaining OAR dose 
    allowed to be prescribed.
    default: -1
```

## Specific entries according to algorithm type

```
maximise tumor BED
----------------
oar_limit : float
    upper BED limit of OAR.

minimise oar BED
----------------
tumor_goal : float
    prescribed tumor BED.

minimise number of fractions
----------------
tumor_goal : float
    prescribed tumor BED.
c: float
    fixed constant to penalise each additional fraction.
```

## General settings for calculation

```
settings
----------------
dose_stepsize : float
    stepsize of the actionspace.
state_stepsize : float
    stepsize of the BED states.
sf_low : float
    lower bound of the possible sparing factors.
sf_high : float
    upper bound of the possible sparing factors.
sf_stepsize: float
    stepsize of the sparing factor stepsize.
sf_prob_threshold': float
    probability threshold of the sparing factor occuring.
inf_penalty : float
    infinite penalty for certain undesired states.
plot_policy : int
    starting from which fraction policy should be plotted.
plot_values : int
    starting from which fraction value should be plotted.
plot_remains : int
    starting from which fraction expected remaining number 
    of fractions should be plotted.
```

# Example

Outlined is an example instruction file for fraction minimisation. It simply is a `.json` that is translated into a python dictionary. An example can be found [here](work/oar_example.json)

This `.json` file can be called in with the CLI as:

```
$ aft -f work/oar_example.json
```