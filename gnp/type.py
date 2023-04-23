"""
@Creator: Viet Dung Nguyen
@Date: March 28, 2023
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Setup for tools
"""


import numpy as np
from typing import List, Tuple, Union, Callable, Iterator, Dict, Sequence


ArrayLike = Union[List, Tuple, np.ndarray]
ScalarLike = Union[int, float]
Multiplicable = Union[ArrayLike, ScalarLike]
ParamAndGradType = Iterator[Tuple[ArrayLike, ArrayLike]]


