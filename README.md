# The `adaptfx` package


## Installation

For the GUIs to work one needs Tkinter. It often comes installed, but if not you can find the relevant installation instructions [here](https://tkdocs.com/tutorial/install.html).

E.g., for python and on Ubuntu, you would install it via

```shell
sudo apt install python3-tk
```


## Description

In these scripts one can find all relevant codes to calculate an OAR tracked adaptive fractionation plan and plan by tracking tumor BED and OAR BED (maximizing tumor BED while minimizing OAR BED).

There are two different types of algorithms. The 2D algorithm, which only tracks OAR BED or tumor BED and maximizes based on an OAR BED or tumor BED constraint. This is the faster algorithm, but it could overshoot with the dose delivered to the tumor/OAR. Since only one of the organs can be tracked, one has to decide whether reaching the prescribed tumor dose or staying below the maximum OAR BED is more relevant. Generally the OAR tracking is better suited for patients with anatomies where the OAR and tumor are close to each other and reaching the prescribed dose is not expected. the tumor tracking is better suited when the OAR and tumor are farther apart and the prescribed tumor dose is supposed to be reached while staying below the maximum OAR BED.

The 3D algorithms tracks OAR BED and tumor BED. In this version a prescribed tumor dose must be provided aswell. The algorithm then tries to reach the prescribed tumor dose while minimizing the dose delivered to the OAR. If the prescribed tumor dose can not be reached, it is maximized with respect to the OAR limit. This means, that the OAR limit will be reached, just like in the 2D program. Generally, both algorithms give the same result, if the prescribed tumor dose can not be reached and is maximized.
The algorithms are based on a inverse-gamma prior distribution. To set up this distribution a dataset is needed with prior patient data (sparing factors) from the same population.

There is a function to calculate the hyperparameters of the inverse-gamma distribution. But there is also the option to use a fixed probability distribution for the sparing factors. In this case, the probability distribution must be provided with a mean and a standard deviation and it is not updated as more information is available. To check out how the hyper parameters influence the prior distribution, the Inverse_gamma_distribution_preview.py file has been included that allows direct modelling of the distribution.

There is a subfolder with more basic algorithms, the discrete algorithms. Generally, we can not calculate the Value function for each possible OAR BED and sparing factor. Thus, the values must be calculated for discrete steps. E.g. 0.1Gy BED steps for the OAR BED and 0.01 steps for the sparing factors. The discrete algorithms depict this idea of using these steps to calculate the value for each discrete value of BED and sparing factor. This approach limits the precision of the computed doses, as we must round any given BED to a the given steps. To improve precision, an interpolation was done, to calculate the value for every possible BED. This is used int the interpolation programs (Those not in the discrete folder). A higher precision comes with the cost of larger computation time, but the 2D code still runs in a matter of seconds, while the 3D code runs in a matter of minutes.

A last addition is made with graphical user interfaces that facilitate the use of the interpolation algorithms. There are two interfaces that can be run. In these interfaces all variables can be given to compute an adaptive frationation plan for a patient.

## Extended functions
The algorithms allow to chose some extra parameters to specify extra constraints. The suggested parameters are specified for a 5 fraction SBRT plan where there are not constraints on the maximum or minimum dose.:
- Chose the amount of fractions. Instead of just calculating for the case of a 5 fractions SBRT treatment, the amount of fractions can be chosen freely (e.g. 30 fractions)
- Fix a minimum and maximum dose: Limits the action space by forcing a minimum and maximum dose for each fraction. (e.g. 4-16Gy)
- Calculate optimal fraction size by tracking tumor BED: The 2D GUI has an additional extension, where one can optimize the optimal dose based on the prescribed tumor dose. (E.g. the clinician prescribes a tumor BED of 72 BED. The program will try to minimize the OAR BED while aiming on the 72 BED prescribed dose.)

## Additional Data
The two additional folders (DVH_figures, Patientdata_paper) contain the DVH data and figures of the 10 patients that were included in the paper.

## T-distribution folder
Apart from using a gamma prior for the standard deviation, a full bayesian approach can be done with a conjugate prior for the variance. 
In the t-distribution folder the same algorithms as in the paper are applied, but instead of using the gamma prior, the probability distribution is estimated from an updated t-distribution by using a inverse-gamma prior for the variance.
The results are slightly different when alternative priors are applied. Since the t-distribution estimates larger standrad deviations, more sparing factors are relevant and thus the state space is increased which results in a longer computation time.
