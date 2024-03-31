# pylint: skip-file

from typing import NamedTuple, Protocol, type_check_only

import numpy as np
from equinox import Grads
from jax.numpy import ndarray

@type_check_only
class ArraysOf[A]: ...
class OptState[A, Float]: ...

class GradientTransformation(NamedTuple):
    init: TransformInitFn
    update: TransformUpdateFn

class TransformInitFn(Protocol):
    def __call__[A](self, params: ArraysOf[A]) -> OptState[A, np.float32]: ...

class TransformUpdateFn(Protocol):
    def __call__[A, Float: float](
        self, updates: Grads[A], state: OptState[A, Float], params: A | None = None
    ) -> tuple[Grads[A], OptState[A, Float]]: ...

def adam(learning_rate: float) -> GradientTransformation: ...
def clip_by_global_norm(max_norm: float) -> GradientTransformation: ...
def chain(
    *transforms: GradientTransformation,
) -> GradientTransformation: ...
def softmax_cross_entropy_with_integer_labels[*Shape, NumClasses: int, Float: float](
    logits: ndarray[*Shape, NumClasses, Float], labels: ndarray[*Shape, np.Fin[NumClasses]]
) -> ndarray[*Shape, Float]: ...
