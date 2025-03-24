from typing import Any, Dict, Type, TypeVar, Union, List, Tuple
from pydantic import BaseModel
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


rng = create_rng()

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


def nanshift(s, n, token=np.nan, axis=-1):
    """
    Shifts the input array by n positions, filling with NaN or specified token.

    Parameters:
    -----------
    s : array-like
        Input signal to be shifted
    n : int
        Number of positions to shift (positive: shift right, negative: shift left)
    axis : int
        Axis to shift along (default: -1)
    token : float
        Value to fill the empty spaces (default: np.nan)

    Returns:
    --------
    array-like : Shifted signal
    """
    s = np.array(s)
    s = np.moveaxis(s, axis, 0)

    if n > 0:
        signal = np.concatenate([np.full((n, *s.shape[1:]), token), s[:-n]], axis=0)
    elif n < 0:
        signal = np.concatenate([s[-n:], np.full((-n, *s.shape[1:]), token)], axis=0)
    else:
        signal = s.copy()

    signal = np.moveaxis(signal, 0, axis)
    return signal


def roll_along_axis(arr, shifts, axis):
    """Rolls an array along a specified axis by the given shifts.

    Allows for different shifts for each element in the array. The shifts
    should have the shape (or number of elements) as the array excluding
    the axis to roll along!

    Parameters:
    -----------
    arr : array-like
        The array to roll
    shifts : array-like
        The shifts to apply to the array
    axis : int
        The axis to roll along

    Returns:
    --------
    array-like : Rolled array
    """
    original_shape = arr.shape
    arr = arr.copy()
    arr = np.moveaxis(arr, axis, -1)
    rearr = np.reshape(arr, (-1, original_shape[axis]))
    shifts = np.reshape(shifts, -1)

    if len(shifts) != rearr.shape[0]:
        raise ValueError(
            f"Length of shifts must match the dimensions of the array everywhere except the roll axis ({rearr.shape[0]})"
        )

    for i in range(rearr.shape[0]):
        rearr[i] = np.roll(rearr[i], shifts[i])

    arr = np.reshape(rearr, original_shape)
    return np.moveaxis(arr, -1, axis)


def compare_models(
    a: Union[BaseModel, Dict[str, Any]],
    b: Union[BaseModel, Dict[str, Any]],
    path: str = "",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Recursively compares two Pydantic BaseModel instances or dictionaries.

    Returns:
        - List of keys in A but not in B
        - List of keys in B but not in A
        - List of keys present in both but with different values
    """
    if isinstance(a, BaseModel):
        a = a.model_dump()
    if isinstance(b, BaseModel):
        b = b.model_dump()

    in_a_not_b = []
    in_b_not_a = []
    different_values = []

    keys_a = set(a.keys())
    keys_b = set(b.keys())

    for key in keys_a - keys_b:
        in_a_not_b.append(f"A: {path}{key} = {a[key]!r}")

    for key in keys_b - keys_a:
        in_b_not_a.append(f"B: {path}{key} = {b[key]!r}")

    for key in keys_a & keys_b:
        value_a, value_b = a[key], b[key]
        new_path = f"{path}{key}."

        if isinstance(value_a, dict) and isinstance(value_b, dict):
            sub_a, sub_b, sub_diff = compare_models(value_a, value_b, new_path)
            in_a_not_b.extend(sub_a)
            in_b_not_a.extend(sub_b)
            different_values.extend(sub_diff)
        elif isinstance(value_a, BaseModel) and isinstance(value_b, BaseModel):
            sub_a, sub_b, sub_diff = compare_models(value_a, value_b, new_path)
            in_a_not_b.extend(sub_a)
            in_b_not_a.extend(sub_b)
            different_values.extend(sub_diff)
        elif value_a != value_b:
            different_values.append(f"A: {new_path[:-1]} = {value_a!r}, B: {new_path[:-1]} = {value_b!r}")

    return in_a_not_b, in_b_not_a, different_values
