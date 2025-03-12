from typing import Any, Dict, Type, TypeVar, Union
import numpy as np
from dataclasses import is_dataclass


def create_rng(seed: int | None = None) -> np.random.Generator:
    """Create a random number generator using the PCG64 algorithm
    (which is usually a good amount faster than the default).

    Parameters
    ----------
    seed : int | None
        The seed for the random number generator.
    """
    return np.random.Generator(np.random.PCG64(seed))


class RNG:
    def __init__(self, seed: int | None = None):
        self.rng = create_rng(seed)

    def set_seed(self, seed: int | None = None):
        self.rng = create_rng(seed)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.rng(*args, **kwargs)


rng = RNG()

T = TypeVar("T")


def resolve_dataclass(input_data: Union[Dict[str, Any], None, T], dataclass_type: Type[T]) -> T:
    """
    Resolve an input to the desired dataclass instance.

    Parameters
    ----------
    input_data : Dict[str, Any] | None | T
        The input data, which can be a dictionary of parameters, None, or
        already an instance of the target dataclass type.
    dataclass_type : Type[T]
        The dataclass type to convert the input to.

    Returns
    -------
    T
        An instance of the specified dataclass type.

    Raises
    ------
    ValueError
        If the input is not a dict, None, or an instance of the target dataclass.
    """
    if not isinstance(dataclass_type, type):
        raise ValueError(f"dataclass_type must be a type, got a {type(dataclass_type)}")

    if input_data is None:
        return dataclass_type()
    elif isinstance(input_data, dict):
        return dataclass_type(**input_data)
    elif is_dataclass(input_data) and isinstance(input_data, dataclass_type):
        return input_data
    else:
        raise ValueError(
            f"Input must be a dict, None, or an instance of {dataclass_type.__name__}, "
            f"got {type(input_data).__name__}"
        )


def cross_correlation(x, y):
    """
    measure the cross correlation between each column in x with every column in y

    sets the cross-correlation to NaN for any element if it has 0 variation
    """
    assert x.ndim == y.ndim == 2, "x and y must be 2-d numpy arrays"
    assert x.shape[0] == y.shape[0], "x and y need to have the same number of dimensions (=rows)!"
    N = x.shape[0]
    xDev = x - np.mean(x, axis=0, keepdims=True)
    yDev = y - np.mean(y, axis=0, keepdims=True)
    xSampleStd = np.sqrt(np.sum(xDev**2, axis=0, keepdims=True) / (N - 1))
    ySampleStd = np.sqrt(np.sum(yDev**2, axis=0, keepdims=True) / (N - 1))
    xIdxValid = xSampleStd > 0
    yIdxValid = ySampleStd > 0
    xSampleStdCorrected = xSampleStd + 1 * (~xIdxValid)
    ySampleStdCorrected = ySampleStd + 1 * (~yIdxValid)
    xDev /= xSampleStdCorrected
    yDev /= ySampleStdCorrected
    std = xDev.T @ yDev / (N - 1)
    std[:, ~yIdxValid[0]] = np.nan
    std[~xIdxValid[0]] = np.nan
    return std


def transpose_list(list_of_lists):
    """helper function for transposing the order of a list of lists"""
    return list(map(list, zip(*list_of_lists)))


def named_transpose(list_of_lists, map_func=list):
    """
    helper function for transposing lists without forcing the output to be a list like transpose_list

    for example, if list_of_lists contains 10 copies of lists that each have 3 iterable elements you
    want to name "A", "B", and "C", then write:
    A, B, C = named_transpose(list_of_lists)
    """
    return map(map_func, zip(*list_of_lists))
