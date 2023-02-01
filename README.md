# The `adaptfx` package

## Content

1. [About](#about)  
2. [Installation](#installation)  
3. [Package Structure](#package-structure)  
4. [Describtion](#description)  
    1. [2D Algorithms](#the-2d-algorithms)
    2. [3D Algorithms](#the-3d-algorithms)
    3. [GUI](#gui)
    4. [Probability Updating](#probability-updating)
    5. [Additional Data](#additional-data)
5. [Extended Function](#extended-functionality)
6. [Troubleshooting](#troubleshooting)

---

## About

`adaptfx` is a python package to calculate adaptive fractionation schemes. Using magnetic resonance (MR) guidance in radiotherapy, treatment plans can be adapted daily to a patient's geometry, thereby exploiting inter-fractional motion of tumors and organs at risk (OAR). This can improve OAR sparing or tumor coverage, compared to standard fractionation schemes, which simply apply a predefined dose every time.


For this adaptive approach a reinforcement learning algorithm based on dynamic programming was initially developed by Pérez Haas et al. [[1]](#1). The package is actively maintained and frequently extended as part of our ongoing research on the topic


## Installation

It is recommended to create a virtual environment using the `venv` module:

```
$ python3.10 -m venv adaptfx_env
```

activate the virtual environment
```
$ cd adaptfx_env
$ source bin/activate
```

To install the `adaptfx` package, use either of the methods below.

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

for more information on the usage of the CLI, read the [manual](MANUAL.md).

The user can also decide to use the scripts from `reinforce` in their python scripts e.g.

```python
import adapatfx as afx
plan_output = afx.multiple('oar', keys)
```

`adaptfx` also provides a GUI. However, it depends on `Tkinter`. It often comes installed, but if not you can find the relevant installation instructions [here](https://tkdocs.com/tutorial/install.html). E.g. in python and on Ubuntu, you would install it via

```
$ sudo apt install python3-tk
```

## Package Structure

The package is organized under the `src` folder. All relevant scripts that calculate the fractionation schemes are packed as functions in either `reinforce.py` or `reinforce_old.py`. Where `reinforce.py` holds the newest functions supporting more features and faster calculation. Older functions are also integrated with the CLI, but need to be updated.

```
adaptfx
├── src/adaptfx
│  ├── aft_propmt.py
│  ├── aft_utils.py
│  ├── aft.py
│  ├── constants.py
│  ├── maths.py
│  ├── planning.py
│  ├── radiobiology.py
│  ├── reinforce_old.py
│  ├── reinforce.py
│  └── visualiser.py
└── work
```

## Description

### The 2D algorithms

The function `max_tumor_bed_old` globally tracks OAR BED to satisfy constraints on the dose to the normal tissue, while attempting to maximize the BED delivered to the tumor.

`min_oar_bed` and `min_oar_bed_old`, on the other hand, track tumor BED to achieve the tumor dose target and in doing so it minimizes the cumulative OAR BED.

Since the state spaces for these two algorithms are essentially two-dimensional, they are the faster algorithm. But they may overshoot w.r.t. the dose delivered to the tumor/OAR, since only one of the structure's BED can be tracked, one has to decide whether reaching the prescribed tumor dose or staying below the maximum OAR BED is more relevant.

Generally the OAR tracking is better suited for patients with anatomies where the OAR and tumor are close to each other and the prescribed dose may not be reached. When the OAR and tumor are farther apart, tracking the tumor BED and minimizing OAR BED can lead to reduced toxicity while achieving the same treatment goals.

`frac_min` defines the function to track OAR BED and minimize the number of fractions in cases where there appears an exceptionally low sparing factor during the course of a treatment.

### The 3D algorithms

The 3D algorithms in function `min_oar_max_tumor_old` track OAR BED and tumor BED simultaneously. In this version a prescribed tumor dose must be provided alongside an OAR BED constraint. The algorithm then tries smartly optimizes for a low OAR BED _and_ high tumor BED at the same time, while never compromising OAR constraints and always preferring to reduce normal tissue dose when achieving the treatment objectives.

The algorithms are based on an inverse-gamma prior distribution. To set up this distribution a dataset is needed with prior patient data (sparing factors) from the same population.

There is a function to calculate the hyperparameters of the inverse-gamma distribution. But there is also the option to use a fixed probability distribution for the sparing factors. In this case, the probability distribution must be provided with a mean and a standard deviation, and it is not updated as more information is available. To check out how the hyperparameters influence the prior distribution, the `Inverse_gamma_distribution_preview.py` file has been included that allows direct modelling of the distribution.

### GUI

A last addition is made with graphical user interfaces that facilitate the use of the interpolation algorithms. There are two interfaces that can be run. In these interfaces all variables can be given to compute an adaptive fractionation plan for a patient. 

> :warning: Note:\
> The interfaces are not optimized, and thus it is not recommended using them to further develop extensions.

### Probability Updating

The DP algorithm relies on a description of the environment to compute an optimal policy, in this case the probability distribution of the sparing factor $P(\delta)$, which we assume to be a Gaussian distribution truncated at $0$, with patient-specific parameters for mean and standard deviation. At the start of a treatment, only two sparing factors are available for that patient, from the planning scan and the first fraction. In each fraction, an additional sparing factor is measured, which can be used to calculate updated estimates $\mu_t$ and $\sigma_t$ for mean and standard deviation, respectively.

#### No Updating

In case where the probability is not updated the parameters $\mu_t$ and $\sigma_t$ of the normal distribution can be fixed.

#### Maximum a posteriori estimation

In each fraction $t$, a maximum likelihood estimator of the mean of the sparing factor distribution and an estimator for the standard deviation (following a chi-squared distribution) is used. Both estimators are used to constitute the updated normal distribution in fraction $t$.

However, the standard deviation may be severely under- or overestimated if calculated from only two samples at the very beginning of the treatment. Therefore, we assume a population based prior for the standard deviation and compute the maximum a posterior estimator of $\sigma_t$ via Bayesian inference. As the sparing factors are assumed to follow a normal distribution with unknown variance, a gamma distribution is chosen as prior to estimate the standard deviation $\sigma$.

#### Posterior predicitve distribution

Apart from using a gamma prior for the standard deviation, a full Bayesian approach can be employed with an inverse-gamm distribution as a conjugate prior for the variance. The resulting posterior predictive distribution is a student t-distribution. With this approach instead of using the gamma prior to estimate, the probability distribution is estimated from an updated t-distribution. The results are slightly different compared to the maximum a posteriori estimation.

### Additional Data

The two additional folders (`DVH_figures`, `Patientdata_paper`) contain the DVH data and figures of the 10 patients that were included in the paper.

## Extended Functionality

The algorithms allow to choose some extra parameters to specify extra constraints. The suggested parameters are specified for a 5 fraction SBRT plan where there are not constraints on the maximum or minimum dose:

- Chose the amount of fractions. Instead of just calculating for the case of a 5-fractions SBRT treatment, the amount of fractions can be chosen freely (e.g. 30 fractions)
- Fix a minimum and maximum dose: Limits the action space by forcing a minimum and maximum dose for each fraction. (e.g. 4-16 Gy)
- Calculate optimal fraction size by tracking tumor BED: The 2D GUI has an additional extension, where one can optimize the optimal dose based on the prescribed tumor dose. (E.g., the clinician prescribes a tumor BED of 72 Gy. The program will try to minimize the OAR BED while aiming at the 72 Gy BED prescribed dose.)

## Troubleshooting

### No module named `_ctypes` on install

**Problem:** on Linux distributions it happens that the `pip install .` command fails with the message:

```
Traceback (most recent call last):
   File "<stdin>", line 1, in <module>
   File "/usr/local/lib/python3.10/some/module", line 10, in <module>
   import ctypes
   File "/usr/local/lib/python3.10/ctypes/__init__.py", line 7, in <module>
      from _ctypes import Union, Structure, Array
ImportError: No module named '_ctypes'
```

**Solution:** with the specific package manager of the Linux distribution install `libffi-dev` development tool. E.g. in Fedora Linux and derivatives install this tool

```
sudo dnf install libffi-devel
```

On Ubuntu:
```
sudo apt install libffi-dev
```

### No GUI backend for `matplotlib`

**Problem:** on Linux or MacOS it could be that once `aft` is run the plots are not shown and there is an error message:


```
Collecting tkinter
  Could not find a version that satisfies the requirement tkinter (from versions: )
No matching distribution found for tkinter
```

**Solution:** on Fedora Linux and derivative distributions one could solve this by either installing python tkinter

```
sudo dnf install python3-tkinter
```

on Ubuntu

```
sudo apt-get install python3-tk
```

**Solution:** on MacOS and Linux one could instead use `pip` to install `pyqt`

```
pip install pyqt5
```


## References

<a id="1">[1]</a>
Yoel Samuel Pérez Haas et al.;
**Adaptive fractionation at the MR-linac**, *Physics in Medicine & Biology*, Jan. 2023, doi: https://doi.org/10.1088/1361-6560/acafd4