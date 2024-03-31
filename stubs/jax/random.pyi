# pylint: skip-file

from typing import Literal, overload

from numpy import ndarray

type Two = Literal[2]
type KeyArray = ndarray[Two, int]

def PRNGKey(seed: int | ndarray[int]) -> ndarray[Two, int]: ...
@overload
def split(key: KeyArray) -> ndarray[Two, Two, int]: ...
@overload
def split[Int: int](key: KeyArray, num: Int) -> ndarray[Int, Two, int]: ...
