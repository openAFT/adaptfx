# -*- coding: utf-8 -*-
import numpy as np

def bed_calc0(dose, ab, sf=1):
    """
    calculates the BED for a specific dose

    Parameters
    ----------
    dose : float
        physical dose to be delivered.
    ab : float
        alpha-beta ratio of tissue.
    sf : float, optional
        sparing factor. The default is 1 (tumor).

    Returns
    -------
    BED : float
        BED to be delivered based on dose, sparing factor
        and alpha-beta ratio.

    """
    BED = sf * dose * (1 + (sf * dose) / ab)
    return BED


def bed_calc_matrix(actionspace, ab, sf):
    """
    calculates the BED for an array of values

    Parameters
    ----------
    actionspace : list/array
        doses to be delivered.
    ab : float
        alpha-beta ratio of tissue.
    sf : list/array
        list of sparing factors to calculate the correspondent BED.

    Returns
    -------
    BED : List/array
        list of all future BEDs based on the delivered doses
        and sparing factors.

    """
    BED = np.outer(actionspace, sf) * (
        1 + np.outer(actionspace, sf) / ab
    )  # produces a actions space x sparing factor array
    return BED

def convert_to_physical(bed, ab, sf=1):
    """
    Converts given BED to the physical dose

    Parameters
    ----------
    BED : float/array
        tumor or OAR BED for which physical dose has to be calculated.
    ab : float
        alpha-beta ratio.
    sf : float/array
        sparing factor, only specify when OAR BED

    Returns
    -------
    dose : positive values float/array
        physical dose
    """
    bed_array = np.array(bed)
    bed_array[bed_array < 0] = 0
    physical_dose = (-sf + np.sqrt(sf**2 + 4 * sf**2 * bed_array / ab)) / (
        2 * sf**2 / ab)
    return physical_dose
