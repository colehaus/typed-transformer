from __future__ import annotations

import functools as ft
from typing import TYPE_CHECKING, Literal, cast, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.numpy import ndarray

from tt.util import InstanceSingleton, product_

if TYPE_CHECKING:
    from numpy import Product

type KeyArray = ndarray[Literal[2], int]


def dot_product_attention_weights[QSeqLen: int, QKDim: int, KSeqLen: int, Float: float](
    query: ndarray[QSeqLen, QKDim, Float],
    key: ndarray[KSeqLen, QKDim, Float],
    mask: ndarray[QSeqLen, KSeqLen, bool] | None = None,
) -> ndarray[QSeqLen, KSeqLen, Float]:
    query = query / jnp.sqrt(query.shape[-1]).astype(query.dtype)
    logits: ndarray[QSeqLen, KSeqLen, Float] = query @ key.T
    if mask is not None:
        logits = jnp.where(mask, logits, jnp.array(jnp.finfo(logits.dtype).min))
    return jax.nn.softmax(logits, axis=-1)


def dot_product_attention[QSeqLen: int, QKDim: int, KVSeqLen: int, VDim: int, Float: float](
    query: ndarray[QSeqLen, QKDim, Float],
    key_: ndarray[KVSeqLen, QKDim, Float],
    value: ndarray[KVSeqLen, VDim, Float],
    mask: ndarray[QSeqLen, KVSeqLen, bool] | None = None,
) -> ndarray[QSeqLen, VDim, Float]:
    weights: ndarray[QSeqLen, KVSeqLen, Float] = dot_product_attention_weights(query, key_, mask)
    return weights @ value


def dot_product_attention_for_query[QKDim: int, KVSeqLen: int, VDim: int, Float: float](
    query: ndarray[QKDim, Float], keys: ndarray[KVSeqLen, QKDim, Float], values: ndarray[KVSeqLen, VDim, Float]
) -> ndarray[VDim, Float]:
    query = query / jnp.sqrt(query.shape[-1]).astype(query.dtype)
    logits: ndarray[KVSeqLen, Float] = jax.vmap(ft.partial(jnp.dot, query))(keys)
    weights: ndarray[KVSeqLen, Float] = jax.nn.softmax(logits)
    return jnp.average(values, axis=0, weights=weights)


def vmap_dot_product_attention[QSeqLen: int, QKDim: int, KVSeqLen: int, VDim: int, Float: float](
    queries: ndarray[QSeqLen, QKDim, Float],
    keys: ndarray[KVSeqLen, QKDim, Float],
    values: ndarray[KVSeqLen, VDim, Float],
) -> ndarray[QSeqLen, VDim, Float]:
    def inner(query: ndarray[QKDim, Float]) -> ndarray[VDim, Float]:
        return dot_product_attention_for_query(query, keys, values)

    return jax.vmap(inner)(queries)


class MHAttention[QDim: int, KDim: int, VDim: int, OutputDim: int, Float: float](eqx.Module):
    # Purely internal types that shouldn't be visible to callers
    type _NumHeads = InstanceSingleton[Literal["NumHeads"]]
    type _QKSize = InstanceSingleton[Literal["QKSize"]]
    type _VOSize = InstanceSingleton[Literal["VOSize"]]

    query_proj: eqx.nn.Linear[QDim, Product[_NumHeads, _QKSize], Float]
    key_proj: eqx.nn.Linear[KDim, Product[_NumHeads, _QKSize], Float]
    value_proj: eqx.nn.Linear[VDim, Product[_NumHeads, _VOSize], Float]
    output_proj: eqx.nn.Linear[Product[_NumHeads, _VOSize], OutputDim, Float]
    num_heads: _NumHeads = eqx.field(static=True)

    @overload
    def __init__(  # noqa: PLR0913
        self,
        *,
        num_heads: int,
        query_size: QDim,
        key_size: KDim,
        value_size: VDim,
        output_size: OutputDim,
        qk_size: int | None = None,
        vo_size: int | None = None,
        key: KeyArray,
    ) -> None:
        ...

    @overload
    def __init__(  # noqa: PLR0913
        self: MHAttention[QDim, QDim, QDim, QDim, Float],
        *,
        num_heads: int,
        query_size: QDim,
        key_size: None = None,
        value_size: None = None,
        output_size: None = None,
        qk_size: int | None = None,
        vo_size: int | None = None,
        key: KeyArray,
    ) -> None:
        ...

    def __init__(  # noqa: PLR0913
        self,
        *,
        num_heads: int,
        query_size: QDim,
        key_size: KDim | None = None,
        value_size: VDim | None = None,
        output_size: OutputDim | None = None,
        qk_size: int | None = None,
        vo_size: int | None = None,
        key: KeyArray,
    ) -> None:
        qkey, kkey, vkey, okey = jax.random.split(key, 4)
        if key_size is None:
            key_size = cast(KDim, query_size)
        if value_size is None:
            value_size = cast(VDim, query_size)
        qk_size = InstanceSingleton[Literal["QKSize"]](
            self, "QKSize", query_size // num_heads if qk_size is None else qk_size
        )
        vo_size = InstanceSingleton[Literal["VOSize"]](
            self, "VOSize", query_size // num_heads if vo_size is None else vo_size
        )
        if output_size is None:
            output_size = cast(OutputDim, query_size)
        self.num_heads = InstanceSingleton(self, "NumHeads", num_heads)
        self.query_proj = eqx.nn.Linear(query_size, product_((self.num_heads, qk_size)), use_bias=False, key=qkey)
        self.key_proj = eqx.nn.Linear(key_size, product_((self.num_heads, qk_size)), use_bias=False, key=kkey)
        self.value_proj = eqx.nn.Linear(value_size, product_((self.num_heads, vo_size)), use_bias=False, key=vkey)
        self.output_proj = eqx.nn.Linear(
            product_((self.num_heads, vo_size)), output_size, use_bias=False, key=okey
        )

    def __call__[QSeqLen: int, KVSeqLen: int](
        self,
        query: ndarray[QSeqLen, QDim, Float],
        key_: ndarray[KVSeqLen, KDim, Float],
        value: ndarray[KVSeqLen, VDim, Float],
        mask: ndarray[QSeqLen, KVSeqLen, bool] | None = None,
    ) -> ndarray[QSeqLen, OutputDim, Float]:
        query_heads: ndarray[QSeqLen, MHAttention._NumHeads, MHAttention._QKSize, Float] = self._project(
            self.query_proj, query
        )
        key_heads: ndarray[KVSeqLen, MHAttention._NumHeads, MHAttention._QKSize, Float] = self._project(
            self.key_proj, key_
        )
        value_heads: ndarray[KVSeqLen, MHAttention._NumHeads, MHAttention._VOSize, Float] = self._project(
            self.value_proj, value
        )

        attn: ndarray[QSeqLen, MHAttention._NumHeads, MHAttention._VOSize, Float] = jax.vmap(
            ft.partial(dot_product_attention, mask=mask), in_axes=1, out_axes=1
        )(query_heads, key_heads, value_heads)
        concatenated_attention: ndarray[
            QSeqLen, Product[MHAttention._NumHeads, MHAttention._VOSize], Float
        ] = jnp.reshape(attn, (query.shape[0], product_(attn.shape[1:])))
        return jax.vmap(self.output_proj)(concatenated_attention)

    def _project[InDim: int, SeqLen: int, OutDim: int](
        self,
        proj: eqx.nn.Linear[InDim, Product[MHAttention._NumHeads, OutDim], Float],
        x: ndarray[SeqLen, InDim, Float],
    ) -> ndarray[SeqLen, MHAttention._NumHeads, OutDim, Float]:
        projection: ndarray[SeqLen, Product[MHAttention._NumHeads, OutDim], Float] = jax.vmap(proj)(x)
        return jnp.reshape(projection, (x.shape[0], self.num_heads, cast(OutDim, -1)))
