# The `adaptfx` package


## Installation

For the GUIs to work one needs Tkinter. It often comes installed, but if not you can find the relevant installation instructions [here](https://tkdocs.com/tutorial/install.html).

E.g., for python and on Ubuntu, you would install it via

```shell
sudo apt install python3-tk
```


## Description

In these scripts one can find all relevant codes to calculate an OAR tracked adaptive fractionation plan and plan by tracking tumor BED and OAR BED (maximizing tumor BED while minimizing OAR BED).

There are two different types of algorithms. The 2D algorithm, which only tracks OAR BED and maximizes based on an OAR constraint. This is the faster algorithm, but it could overshoot with the dose delivered to the tumor, as no prescribed tumor dose can be given. Therefore, The 2D program aims to maximize the tumor dose and always delivers as much dose as possible while precisely reaching the OAR limit.

The 3D algorithms tracks OAR BED and tumor BED. In this version a prescribed tumor dose must be provided aswell. The algorithm then tries to reach the prescribed tumor dose while minimizing the dose delivered to the OAR. If the prescribed tumor dose can not be reached, it is maximized with respect to the OAR limit. This means, that the OAR limit will be reached, just like in the 2D program. Generally, both algorithms give the same result, if the prescribed tumor dose can not be reached and is maximized.
The algorithms are based on a inverse-gamma prior distribution. To set up this distribution a dataset is needed with prior patient data (sparing factors) from the same population.

There is a function to calculate the hyperparameters of the inverse-gamma distribution. But there is also the option to use a fixed probability distribution for the sparing factors. In this case, the probability distribution must be provided with a mean and a standard deviation and it is not updated as more information is available.

There is a subfolder with more basic algorithms, the discrete algorithms. Generally, we can not calculate the Value function for each possible OAR BED and sparing factor. Thus, the values must be calculated for discrete steps. E.g. 0.1Gy BED steps for the OAR BED and 0.01 steps for the sparing factors. The discrete algorithms depict this idea of using these steps to calculate the value for each discrete value of BED and sparing factor. This approach limits the precision of the computed doses, as we must round any given BED to a the given steps. To improve precision, an interpolation was done, to calculate the value for every possible BED. This is used int the interpolation programs (Those not in the discrete folder). A higher precision comes with the cost of larger computation time, but the 2D code still runs in a matter of seconds, while the 3D code runs in a matter of minutes.

A last addition is made with graphical user interfaces that facilitate the use of the interpolation algorithms. There are two interfaces that can be run. In these interfaces all variables can be given to compute an adaptive frationation plan for a patient.
