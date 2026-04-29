"""
Implementation of the Beer-Lambert turbidity model used for the simulation. It is a simple function of the depth.

Params:
    - (d)   -> depth
    - (I_0) -> base turbidity level (surface)
    - (k)   -> constant diffuse attenuation coefficient
"""

# TODO: import (I_0, k) from config file (not necessary .xml)
import numpy as np

def turbidity_model(depth, I0=1, k=1): 
    return I0 * (np.exp(-(k * depth)))