# The `adaptfx` manual

## Working tree

One should perform calculations in a working folder e.g. `adaptfx/work`. There the instruction files can be specified by the user and `adaptfx` will automatically produce log files if the user wishes to.

## Format of the instruction file
The user specifies following elements of the dictionary for the main entries: 
- `'algorithm'`
- `'debug'`
- `'log'`
- `'keys'`
- `'settings'`

Each entry of the dictionary is either a parameter or a dictionary itself. In the next section explained are the main entries.

## Main entries

```
algorithm: frac, oar, tumor, tumor_oar
    type of algorithm
    frac : minimise number of fractions
    oar: minimise oar BED, with tumor constraint
    tumor : maximise tumor BED, with oar constraint
    tumor_oar : minimise oar and maximise tumor BED simultaneously
debug: 0, 1
    show more information (for developers)
    default: 0
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
alpha : float
    shape of inverse-gamma distribution.
beta : float
    scale of inverse-gamme distribution.
abt : float
    alpha-beta ratio of tumor.
abn : float
    alpha-beta ratio of OAR.
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
fixed_prob : int
    this variable is to turn on a fixed probability distribution.
    If the variable is not used (0), then the probability will be updated.
    If the variable is turned to (1), the inserted mean and std will be used
    for a fixed sparing factor distribution. Then alpha and beta are unused.
fixed_mean: float
    mean of the fixed sparing factor normal distribution.
fixed_std: float
    standard deviation of the fixed sparing factor normal distribution.
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
c: float
    fixed constant to penalise for each additional fraction that is used.
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
    in which fraction should current and future policies be plotted.
```

# Example

Outlined is an example instruction file for fraction minimisation. It simply is a python dictionary with parameters.

```
# instruction_file
{
'algorithm': 'frac',
'debug': 1,
'log': 0,
'keys':
    {
	'number_of_fractions': 6,
	'sparing_factors': [0.98, 0.97, 0.8, 0.83, 0.8, 0.85, 0.94],
	'alpha': None,
	'beta': None,
    'abt': 10,
    'abn': 3,
	'fixed_prob': 1,
	'fixed_mean': 0.9,
	'fixed_std': 0.04,
	'tumor_goal': 72,
	'c': 0.8,
	},
'settings':
    {
    'dose_stepsize': 0.5,
    'state_stepsize': 0.5,
    'sf_low': 0,
    'sf_high': 1.7,
    'sf_stepsize': 0.01,
    'sf_prob_threshold': 1e-5,
    'plot_policy': 0
    }
}
```
This dictionary stored in a `.txt` file can be called in with the CLI as:

```
$ aft -f <instruction_file>
```