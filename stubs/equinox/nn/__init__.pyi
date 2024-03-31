from typing import Literal

from jax.numpy import ndarray
from jax.random import KeyArray
from numpy import Fin

from .._module import Module

# Note that that `Float` parameter used in most of these declarations is not quite right.
# Most of them are actually `float32` after initialization
# (e.g. a `Linear` layer initializes the weights and biases as `float32` arrays).
# But we can change this after the fact by `tree_map`ing.
# Including `Float` as a parameter is a small lie then, but it allows us to ensure
# that all our layers fit together correctly with the same floating point type which
# is a more valuable property for the types to guarantee.
# e.g. It ensures that any `ndarray`s we declare inline (`np.ones`, etc.) use the right type.

class Linear[InDim, OutDim, Float: float](Module):
    bias: ndarray[OutDim, Float]
    weight: ndarray[OutDim, InDim, Float]
    def __init__(
        self,
        in_features: InDim | Literal["scalar"],
        out_features: OutDim | Literal["scalar"],
        use_bias: bool = True,
        *,
        key: KeyArray,
    ) -> None: ...
    def __call__(self, x: ndarray[InDim, Float]) -> ndarray[OutDim, Float]: ...

class Embedding[VocabSize: int, EmbedDim, Float: float](Module):
    num_embeddings: VocabSize
    embedding_size: EmbedDim
    weight: ndarray[VocabSize, EmbedDim, Float]
    def __init__(self, num_embeddings: VocabSize, embedding_size: EmbedDim, *, key: KeyArray) -> None: ...
    def __call__(self, x: ndarray[Fin[VocabSize]]) -> ndarray[EmbedDim, Float]: ...

class LayerNorm[EmbedDim, Float: float](Module):
    def __init__(
        self,
        shape: EmbedDim,
        eps: Float = ...,  # pyright: ignore[reportInvalidTypeVarUse]
        use_weight: bool = True,
        use_bias: bool = True,
    ) -> None: ...
    def __call__(self, x: ndarray[EmbedDim, Float]) -> ndarray[EmbedDim, Float]: ...
