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
