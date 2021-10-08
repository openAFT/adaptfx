In these scripts one can find all relevant codes to calculate a OAR tracked adaptive fractionation plan and plan by tracking tumor BED and OAR BED (maximizing tumor BED while minimizing OAR BED). 
The updater plan is well fit to do a posteriori calculation. This means, that all the sparing factors of a patient must be known to calculate the plan. 
The single fraction file can be used when not all sparing factors are known and we want to calculate the best possible plan at any point during the treatment. Note: in the single fraction file you need to feed in the hyperparameters. But there is a function included where one can calculate them by putting in prior patient data.
The algorithms are based on a inverse-gamma prior distribution. To set up this distribution a dataset is needed with prior patient data (sparing factors) form the same population. 

There are two different types of algorithms. Discrete and interpolation algorithms. The interpolation algorithms are more precise and deliver in general better results than the discrete algorithms. But the general idea of a dynamic programming approach to solve the adaptive fractionation is better visible in the discrete codes.
