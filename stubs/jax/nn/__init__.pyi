from typing import Literal

from numpy import ndarray

def softmax[*Shape, Float: float](x: ndarray[*Shape, Float], axis: Literal[-1] = -1) -> ndarray[*Shape, Float]: ...
def gelu[*Shape, Float: float](x: ndarray[*Shape, Float]) -> ndarray[*Shape, Float]: ...
