# The `adaptfx` package

## About

`adaptfx` is a python package to calculate adaptive fractionation schemes. Using magnetic resonance (MR) guidance in radiotherapy, treatment plans can be adapted daily to a patient's geometry, thereby exploiting inter-fractional motion of tumors and organs at risk (OAR). This can improve OAR sparing or tumor coverage, compared to standard fractionation schemes, which simply apply a predefined dose every time.

For this adaptive approach a reinforcement learning algorithm based on dynamic programming was initially developed by Pérez Haas et al. [[1]](#1). The package is actively maintained and frequently extended as part of our ongoing research on the topic

## Installation

To install the `adaptfx` package, use either of the methods below:

### Method 1: pip

```
$ pip install adaptfx
```

### Method 2: install from source

```
$ git clone https://github.com/openAFT/adaptfx.git
$ cd adaptfx
$ pip3 install .
```

the command line tool (CLI) is then available and can be used via

```
$ aft [options] <instructions_file>
```

for more information on the usage of the CLI, read the manual.

The user can also decide to use the scripts from `reinforce` in their python scripts e.g.

```python
import reinforce.tumor_maximisation as tumor_max
```

`adaptfx` also provides a GUI. However, it depends on `Tkinter`. It often comes installed, but if not you can find the relevant installation instructions [here](https://tkdocs.com/tutorial/install.html). E.g. in python and on Ubuntu, you would install it via

```
$ sudo apt install python3-tk
```

## Package Structure

The package is organized under the `src` folder. The relevant scripts that calculate the fractionation schemes are located in `reinforce`. 

```
src
├── common
│  ├── constants.py
│  ├── maths.py
│  └── radiobiology.py
├── console
│  └── aft.py
├── const_eval.py
├── handler
│  ├── aft_utils.py
│  └── messages.py
└── reinforce
   ├── fraction_minimisation.py
   ├── oar_minimisation.py
   ├── plan.py
   ├── track_tumor_oar.py
   └── tumor_maximisation.py
```

## Description

In the `reinforce` module one can find all relevant code to calculate an OAR tracked adaptive fractionation plan and plan by tracking tumor biological effective dose (tumor BED) and OAR BED (maximizing tumor BED while minimizing OAR BED). 

### The 2D algorithms

The code in `tumor_maximization.py` globally tracks OAR BED to satisfy constraints on the dose to the normal tissue, while attempting to maximize the BED delivered to the tumor.

`oar_minimization.py`, on the other hand, tracks tumor BED to achieve the tumor dose target and in doing so it minimizes the cumulative OAR BED.

Since the state spaces for these two algorithms are essentially two-dimensional, they are the faster algorithm. But they may overshoot w.r.t. the dose delivered to the tumor/OAR, since only one of the structure's BED can be tracked, one has to decide whether reaching the prescribed tumor dose or staying below the maximum OAR BED is more relevant.

Generally the OAR tracking is better suited for patients with anatomies where the OAR and tumor are close to each other and the prescribed dose may not be reached. When the OAR and tumor are farther apart, tracking the tumor BED and minimizing OAR BED can lead to reduced toxicity while achieving the same treatment goals.

`fraction_minimisation.py` defines functions to track OAR BED and minimize the number of fractions in cases where there appears an exceptionally low sparing factor during the course of a treatment.

### The 3D algorithms

The 3D algorithms in `track_tumor_oar.py` track OAR BED and tumor BED simultaneously. In this version a prescribed tumor dose must be provided alongside an OAR BED constraint. The algorithm then tries smartly optimizes for a low OAR BED _and_ high tumor BED at the same time, while never compromising OAR constraints and always preferring to reduce normal tissue dose when achieving the treatment objectives.

The algorithms are based on an inverse-gamma prior distribution. To set up this distribution a dataset is needed with prior patient data (sparing factors) from the same population.

There is a function to calculate the hyperparameters of the inverse-gamma distribution. But there is also the option to use a fixed probability distribution for the sparing factors. In this case, the probability distribution must be provided with a mean and a standard deviation, and it is not updated as more information is available. To check out how the hyperparameters influence the prior distribution, the `Inverse_gamma_distribution_preview.py` file has been included that allows direct modelling of the distribution.

### Discrete Value Function

There is a subfolder with more basic algorithms, the discrete algorithms. Generally, we cannot calculate the value function for each possible OAR BED and sparing factor. Thus, the values must be calculated for discrete steps. E.g. 0.1Gy BED steps for the OAR BED and 0.01 steps for the sparing factors. The discrete algorithms depict this idea of using these steps to calculate the value for each discrete value of BED and sparing factor. This approach limits the precision of the computed doses, as we must round any given BED to the given steps. So interpolation was used to improve precision, in calculating every possible BED. A higher precision comes with the cost of larger computation time, but the 2D code still runs in a matter of seconds, while the 3D code runs in a matter of minutes.

### GUI

A last addition is made with graphical user interfaces that facilitate the use of the interpolation algorithms. There are two interfaces that can be run. In these interfaces all variables can be given to compute an adaptive fractionation plan for a patient. 

> :warning: Note:\
> The interfaces are not optimized, and thus it is not recommended using them to further develop extensions.

### T-distribution

Apart from using a gamma prior for the standard deviation, a full Bayesian approach can be done with a conjugate prior for the variance.
In the t-distribution folder the same algorithms as in the paper are applied, but instead of using the gamma prior, the probability distribution is estimated from an updated t-distribution by using an inverse-gamma prior for the variance.
The results are slightly different when alternative priors are applied. Since the t-distribution estimates larger standard deviations, more sparing factors are relevant and thus the state space is increased which results in a longer computation time.

### Additional Data

The two additional folders (`DVH_figures`, `Patientdata_paper`) contain the DVH data and figures of the 10 patients that were included in the paper.

## Extended Functionality

The algorithms allow to choose some extra parameters to specify extra constraints. The suggested parameters are specified for a 5 fraction SBRT plan where there are not constraints on the maximum or minimum dose:

- Chose the amount of fractions. Instead of just calculating for the case of a 5-fractions SBRT treatment, the amount of fractions can be chosen freely (e.g. 30 fractions)
- Fix a minimum and maximum dose: Limits the action space by forcing a minimum and maximum dose for each fraction. (e.g. 4-16 Gy)
- Calculate optimal fraction size by tracking tumor BED: The 2D GUI has an additional extension, where one can optimize the optimal dose based on the prescribed tumor dose. (E.g., the clinician prescribes a tumor BED of 72 Gy. The program will try to minimize the OAR BED while aiming at the 72 Gy BED prescribed dose.)

## References

<a id="1">[1]</a>
Yoel Samuel Pérez Haas, Roman Ludwig, Riccardo Dal Bello, Lan Wenhong, Stephanie Tanadini-Lang, Jan Unkelbach;
[**Adaptive fractionation at the MR-Linac based on a dynamic programming approach**](https://www.estro.org/Congresses/ESTRO-2022/562/inter-fractionmotionandadaptiveradiotherapy/5249/adaptivefractionationatthemr-linacbasedonadynamicp), _ESTRO 2022_, OC-0944
