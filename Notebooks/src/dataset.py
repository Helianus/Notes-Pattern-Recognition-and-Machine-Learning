import numpy as np

from typing import Callable, Tuple

def generate_toy_dataset(f: Callable[[np.ndarray], np.ndarray], 
                        domain: Tuple[float, float],
                        sample_size: int, 
                        std: float
                        ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Generate a toy dataset given a function, a domain and a sample size,
    and then adds Gaussian noise to the samples with zero mean and the given std.
    
    :param f: a function
    :param domain: the domain range
    :param sample_size: the size of the sample
    :param std: standard deviation of the Gaussian noise
    :return: a tuple of the input, target arrays
    """

    # Generate samples from the function
    x = np.linspace(domain[0], domain[1], num=sample_size)
    y = f(x)

    # Add Gaussian noise to the samples
    y_noisy = y + np.random.normal(0, std, size=len(y))

    return x, y_noisy