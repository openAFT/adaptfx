import numpy as np

def BED_calc0(dose, ab, sparing=1):
    """
    calculates the BED for a specific dose

    Parameters
    ----------
    dose : float
        physical dose to be delivered.
    ab : float
        alpha-beta ratio of tissue.
    sparing : float, optional
        sparing factor. The default is 1 (tumor).

    Returns
    -------
    BED : float
        BED to be delivered based on dose, sparing factor
        and alpha-beta ratio.

    """
    BED = sparing * dose * (1 + (sparing * dose) / ab)
    return BED


def BED_calc_matrix(sf, ab, actionspace):
    """
    calculates the BED for an array of values

    Parameters
    ----------
    sf : list/array
        list of sparing factors to calculate the correspondent BED.
    ab : float
        alpha-beta ratio of tissue.
    actionspace : list/array
        doses to be delivered.

    Returns
    -------
    BED : List/array
        list of all future BEDs based on the delivered doses
        and sparing factors.

    """
    BED = np.outer(sf, actionspace) * (
        1 + np.outer(sf, actionspace) / ab
    )  # produces a sparing factors x actions space array
    return BED


def max_action(bed, actionspace, goal, abt=10):
    """
    Computes the maximal dose that can be delivered to the tumor
    in each fraction depending on the actual accumulated dose

    Parameters
    ----------
    bed : float
        accumulated tumor dose so far.
    actionspace : list/array
        array with all discrete dose steps.
    goal : float
        prescribed tumor dose.
    abt : float, optional
        alpha beta ratio of tumor. The default is 10.

    Returns
    -------
    sizer : integer
        gives the size of the resized actionspace to reach
        the prescribed tumor dose.

    """
    max_action = min(max(BED_calc0(actionspace, abt)), goal - bed)
    sizer = np.argmin(np.abs(BED_calc0(actionspace, abt) - max_action))

    return sizer


def argfind(bedt, value):
    """
    This function is used to find the index of certain values

    Parameters
    ----------
    bedt : list/array
        list of tumor BED in which value is searched.
    value : float
        item inside list.

    Returns
    -------
    index : integer
        index of value inside list.

    """
    index = min(range(len(BEDT)), key=lambda i: abs(BEDT[i] - value))
    return index

def converted_dose(bed, ab, sf=1):
    """
    Converts given BED to the physical dose

    Parameters
    ----------
    BED : float/array
        tumor or OAR BED for which physical dose has to be calculated.
    ab : float
        alpha beta ratio.
    sf : float/array
        sparing factor, only specify when OAR BED

    Returns
    -------
    dose : positive values float/array
        physical dose
    """
    dose = (-sf + np.sqrt(sf**2 + 4 * sf**2 * bed / ab)) / (
        2 * sf**2 / ab)
    return dose
