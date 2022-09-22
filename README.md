# The `adaptfx` package

## About

`adaptfx` is a python package distributed on PyPI, which allows to calculate dose fractionation schemes in Adaptive Radiotherapy. Using MR guidance in Adaptive Radiotherapy, treatment plans can be on-line adapted to inter-fractional motion of tumours and organs at risk (OAR). When standard treatments deliver the same dose in each fraction, Adaptive Fractionation exploits the inter-fractional motion by delivering dose adaptively to daily tumour and OAR distance.

For this adaptive approach a Reinforcement Learning algorithm based on dynamic programming was developed. This package was built to provide the toolbox initially developed by Pérez Haas et al. [[1](https://www.estro.org/Congresses/ESTRO-2022/562/inter-fractionmotionandadaptiveradiotherapy/5249/adaptivefractionationatthemr-linacbasedonadynamicp)]. It allows calculation of Adaptive Dose Fractionation discussed in the initial (not-yet) published paper and brings newer applicable features to the user.

## Installation

To install the `adaptfx` package:


#### Method 1: pip

```shell
$ pip install adaptfx
```

#### Method 2: install from source

```shell
$ git clone https://github.com/openAFT/adaptfx.git
$ cd adaptfx
$ pip3 install .
```

the command line tool (CLI) is then available which can be used via

```shell
$ aft [options] <instructions_file>
````

for more infromation on the usage of the CLI read the manual.

The user can also decide to use the scripts from `reinforce` in their python scripts

```python
import reinforce.tumor_maximisation as tumor_max
```
`aft` is also shipped with a GUI. It is however dependent on `Tkinter`. It often comes installed, but if not you can find the relevant installation instructions [here](https://tkdocs.com/tutorial/install.html). E.g. in python and on Ubuntu, you would install it via

```shell
$ sudo apt install python3-tk
```

## Dependecies

Dependent on `click`, `numpy`, `scipy`, `pandas`

## Package Structure

The package is organised in the `src` folder. The relevant scripts that calculate the fractionation schemes are located in `reinforce`. 
```
src
└───common
│   │   constants.py
|   |   maths.py
│   │   radiobiology.py
│   
└───console
|   │   aft.py
│
└───handler
│   │   aft_utils.py
│   │   messages.py
│   
└───reinforce
    │   fraction_minimisation.py
    │   oar_minimisation.py
    |   plan.py
    │   track_tumor_oar.py
    │   tumor_maximisation.py
```

## Description

In the `reinforce` module one can find all relevant code to calculate an OAR tracked adaptive fractionation plan and plan by tracking tumour biological effective dose (tumour BED) and OAR BED (maximizing tumour BED while minimizing OAR BED). 

### The 2D algorithms
```
└───
    │   oar_minimisation.py
    │   tumor_maximisation.py
```
These only track OAR BED or tumour BED and maximizes based on tumour BED or minimises based on OAR BED constraint. These are the faster algorithm, due to the smaller state space, but it could overshoot with the dose delivered to the tumour/OAR. Since only one of the organs can be tracked, one has to decide whether reaching the prescribed tumour dose or staying below the maximum OAR BED is more relevant. Generally the OAR tracking is better suited for patients with anatomies where the OAR and tumour are close to each other and reaching the prescribed dose is not expected. The tumour tracking is better suited when the OAR and tumour are farther apart and the prescribed tumour dose is supposed to be reached while staying below the maximum OAR BED.

```
└───
    │   fraction_minimisation.py
```
This function tracks OAR BED and minimises the number of fractions in cases where there appears an exceptionally low sparing factor.


### The 3D algorithms
```
└───
    │   track_tumor_oar.py
```

The 3D algorithms tracks OAR BED and tumour BED. In this version a prescribed tumour dose must be provided aswell. The algorithm then tries to reach the prescribed tumour dose while minimizing the dose delivered to the OAR. If the prescribed tumour dose can not be reached, it is maximized with respect to the OAR limit. This means, that the OAR limit will be reached, just like in the 2D program. Generally, both algorithms give the same result, if the prescribed tumour dose can not be reached and is maximized.
The algorithms are based on a inverse-gamma prior distribution. To set up this distribution a dataset is needed with prior patient data (sparing factors) from the same population.

There is a function to calculate the hyperparameters of the inverse-gamma distribution. But there is also the option to use a fixed probability distribution for the sparing factors. In this case, the probability distribution must be provided with a mean and a standard deviation and it is not updated as more information is available. To check out how the hyper parameters influence the prior distribution, the `Inverse_gamma_distribution_preview.py` file has been included that allows direct modelling of the distribution.

### Discrete Value Function

There is a subfolder with more basic algorithms, the discrete algorithms. Generally, we can not calculate the Value function for each possible OAR BED and sparing factor. Thus, the values must be calculated for discrete steps. E.g. 0.1Gy BED steps for the OAR BED and 0.01 steps for the sparing factors. The discrete algorithms depict this idea of using these steps to calculate the value for each discrete value of BED and sparing factor. This approach limits the precision of the computed doses, as we must round any given BED to the given steps. So interpolation was used to improve precision, in calculating every possible BED. A higher precision comes with the cost of larger computation time, but the 2D code still runs in a matter of seconds, while the 3D code runs in a matter of minutes.

### GUI

A last addition is made with graphical user interfaces that facilitate the use of the interpolation algorithms. There are two interfaces that can be run. In these interfaces all variables can be given to compute an adaptive frationation plan for a patient. 

>Note!: The interfaces are not optimized and thus it is not recommended to use them to further develop extensions.

### T-distribution
Apart from using a gamma prior for the standard deviation, a full bayesian approach can be done with a conjugate prior for the variance.
In the t-distribution folder the same algorithms as in the paper are applied, but instead of using the gamma prior, the probability distribution is estimated from an updated t-distribution by using a inverse-gamma prior for the variance.
The results are slightly different when alternative priors are applied. Since the t-distribution estimates larger standrad deviations, more sparing factors are relevant and thus the state space is increased which results in a longer computation time.

### Additional Data
The two additional folders (`DVH_figures`, `Patientdata_paper`) contain the DVH data and figures of the 10 patients that were included in the paper.

## Extended Functionality

The algorithms allow to chose some extra parameters to specify extra constraints. The suggested parameters are specified for a 5 fraction SBRT plan where there are not constraints on the maximum or minimum dose.:
- Chose the amount of fractions. Instead of just calculating for the case of a 5 fractions SBRT treatment, the amount of fractions can be chosen freely (e.g. 30 fractions)
- Fix a minimum and maximum dose: Limits the action space by forcing a minimum and maximum dose for each fraction. (e.g. 4-16Gy)
- Calculate optimal fraction size by tracking tumour BED: The 2D GUI has an additional extension, where one can optimize the optimal dose based on the prescribed tumour dose. (E.g. the clinician prescribes a tumour BED of 72 BED. The program will try to minimize the OAR BED while aiming on the 72 BED prescribed dose.)
