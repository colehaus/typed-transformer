# pylint: skip-file

from collections.abc import Callable, Iterator
from typing import Any, Literal, TypeVar, TypeVarTuple, overload, type_check_only

from numpy import ndarray

from . import nn as nn
from . import numpy as numpy
from . import random as random
from . import tree_util as tree_util

A = TypeVar("A")
Shape = TypeVarTuple("Shape")
Shape2 = TypeVarTuple("Shape2")
Shape3 = TypeVarTuple("Shape3")
Shape4 = TypeVarTuple("Shape4")
Dim1 = TypeVar("Dim1")
Dim2 = TypeVar("Dim2")
Dim3 = TypeVar("Dim3")
Dim4 = TypeVar("Dim4")
Dim5 = TypeVar("Dim5")
DType = TypeVar("DType")
DType2 = TypeVar("DType2")
DType3 = TypeVar("DType3")
DType4 = TypeVar("DType4")

@type_check_only
class AuxDim[*Shape, A]:
    def __iter__(self: AuxDim[Dim1, A]) -> Iterator[A]: ...

@overload
def vmap(
    fun: Callable[
        [
            ndarray[*Shape, DType],
            ndarray[*Shape2, DType2],
            ndarray[*Shape3, DType3],
        ],
        ndarray[*Shape4, DType4],
    ],
) -> Callable[
    [
        ndarray[Dim1, *Shape, DType],
        ndarray[Dim1, *Shape2, DType2],
        ndarray[Dim1, *Shape3, DType3],
    ],
    ndarray[Dim1, *Shape4, DType4],
]: ...
@overload
def vmap(
    fun: Callable[
        [
            ndarray[Dim1, *Shape, DType],
            ndarray[Dim2, *Shape2, DType2],
            ndarray[Dim3, *Shape3, DType3],
        ],
        ndarray[Dim4, *Shape4, DType4],
    ],
    in_axes: Literal[1],
    out_axes: Literal[1],
) -> Callable[
    [
        ndarray[Dim1, Dim5, *Shape, DType],
        ndarray[Dim2, Dim5, *Shape2, DType2],
        ndarray[Dim3, Dim5, *Shape3, DType3],
    ],
    ndarray[Dim4, Dim5, *Shape4, DType4],
]: ...
@overload
def vmap(
    fun: Callable[[ndarray[*Shape, DType]], ndarray[*Shape2, DType2]]
) -> Callable[[ndarray[Dim1, *Shape, DType]], ndarray[Dim1, *Shape2, DType2]]: ...
@overload
def vmap(
    fun: Callable[
        [ndarray[*Shape, DType], ndarray[*Shape2, DType2]],
        A,
    ],
) -> Callable[
    [ndarray[Dim1, *Shape, DType], ndarray[Dim1, *Shape2, DType2]],
    AuxDim[Dim1, A],
]: ...
def jit[CallableT: Callable[..., Any]](func: CallableT) -> CallableT: ...
