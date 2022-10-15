# The `adaptfx` manual

## Working tree

One should perform calculations in a working folder. There instruction files should be specified by the user and `adaptfx` will automatically produce log files if the user wishes to.

## Format of the instruction file
The instruction file is simply a python dictionary with parameters. This dictionary is called in the CLI as:

```
$ aft -f <instruction_file>
```
As an example in the first keys of the dictionary the user specifies:

```
# instruction_file
{
'algorithm': 'frac',
'single_fraction' : 0,
'debug': 1,
'log': 0,
'parameters': 
	{
	'number_of_fractions': 6,
	'sparing_factors': [0.98, 0.97, 0.8, 0.83, 0.8, 0.85, 0.94],
	'alpha': None,
	'beta': None,
	'fixed_prob': 1,
	'fixed_mean': 0.9,
	'fixed_std': 0.04,
	'tumor_goal': 72,
	'oar_limit': 90,
	'C': 0.8,
	}
}
```

```
key
-------
algorithm: frac, oar, tumor, tumor_oar
    type of algorithm
    default: 
debug: 0, 1
    show more information (for developers)
    default: 0
log: 0,1
    log output to a file
    default: 0
```

```
Parameters
----------
number_of_fractions : integer
    number of fractions that will be delivered.
sparing_factors : list/array
    list/array with all observed sparing factors.
alpha : float
    shape of inverse-gamma distribution.
beta : float
    scale of inverse-gamme distrinbution.
abt : float
    alpha-beta ratio of tumor.
abn : float
    alpha-beta ratio of OAR.
min_dose : float
    minimal physical doses to be delivered in one fraction.
    The doses are aimed at PTV 95.
max_dose : float
    maximal physical doses to be delivered in one fraction.
    The doses are aimed at PTV 95. If -1 the dose is adapted to the remaining dose tumor dose to be delivered or the remaining OAR dose allowed to be presribed.
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

if single_fraction
----------
fraction : integer
    number of actual fraction
accumulated_oar_dose : float
    accumulated OAR BED (from previous fractions).
accumulated_tumor_dose : float
    accumulated tumor BED (from previous fractions).

if tumor
----------
oar_limit : float
    upper BED limit of OAR.

if oar
----------
tumor_goal : float
    prescribed tumor BED.

if frac
----------
c: float
    fixed constant to penalise for each additional fraction that is used.
```