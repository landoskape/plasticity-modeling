import numpy as np


def create_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))
