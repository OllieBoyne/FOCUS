import numpy as np

def rgb2xyz(rgb: np.ndarray) -> np.ndarray:
    xyz = (rgb - 0.5) * 2
    return xyz

def kappa_to_alpha(kappa: float | np.ndarray) -> np.ndarray:
    alpha = ((2 * kappa) / ((kappa ** 2.0) + 1)) \
            + ((np.exp(- kappa * np.pi) * np.pi) / (1 + np.exp(- kappa * np.pi)))
    alpha = np.degrees(alpha)
    return alpha