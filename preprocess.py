import numpy as np


def euc_normalize(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """normalize vectors in `x` to the range of [0,1]

    Args:
        x (np.ndarray): [description]
        min_value (float): [description]
        max_value (float): [description]

    Returns:
        np.ndarray: [description]
    """
    return (x - min_value) / (max_value - min_value)


def cos_normalize(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """normalization function for cosine space datasets
    Args:
        x (np.ndarray): the vectors
        min_value (float): this value is ignored and takes no effect
        max_value (float): this value is ignored and takes no effect
    Returns:
        np.ndarray: [description]
    """
    norm = np.linalg.norm(x, axis=1)
    norm.resize((len(norm), 1))
    ret = x / norm
    # set elements of all-zero vectors as a large-enough number (here 100)
    # thus they cannot become KNN of any given queries
    ret[np.isnan(ret)] = 100
    return ret
