# pylint: skip-file

from collections.abc import Callable
from typing import Any, Literal, overload, type_check_only

import nn as nn
from jax.numpy import ndarray

from ._module import *

@type_check_only
class Grads[A]: ...

@overload
def filter_jit[CallableT: Callable[..., Any]](
    *, donate: Literal["all", "warn", "none"]
) -> Callable[[CallableT], CallableT]: ...
@overload
def filter_jit[TCallable: Callable[..., Any]](fun: TCallable) -> TCallable: ...
def filter_value_and_grad[A, B, *Rest, Float: float](
    has_aux: Literal[True],
) -> Callable[
    [Callable[[A, *Rest], tuple[ndarray[Float], B]]],
    Callable[[A, *Rest], tuple[tuple[ndarray[Float], B], Grads[A]]],
]: ...
@type_check_only
class PartOf[A]: ...

def filter[A](
    pytree: A,
    filter_spec: Callable[[Any], bool],
    is_leaf: Callable[[Any], bool] | None = None,
    inverse: bool = False,
) -> PartOf[A]: ...
def is_array(element: Any) -> bool: ...
def apply_updates[M: Module](model: M, updates: Grads[M]) -> M: ...
def field(static: bool = False) -> Any: ...
