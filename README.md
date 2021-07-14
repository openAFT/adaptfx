In this scripts one can find all relevant codes to calculate a 2D adaptive fractionation plan. 
The updater plan is well fit to do a posteriori calculation. This means, that all the sparing factors of a patient must be known to calculate the plan. 
The adaptive fractionation file can be used when not all sparing factors are known and we want to calculate the best possible plan at any point during the treatment.
The algorithms are based on a inverse-gamma prior distribution. To set up this distribution a dataset is needed with prior patient data (sparing factors) form the same population. 
