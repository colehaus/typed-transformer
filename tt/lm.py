from __future__ import annotations

import functools as ft
import itertools as it
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from statistics import mean
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from jax.numpy import ndarray
from numpy import float32
from tqdm.auto import tqdm

from tt.architecture import LM, CausalMask, NoMask, Output
from tt.config import EmbedSizeL, MaxSeqLenL, VocabSizeL, bos_token_id, max_seq_len, pad_token_id, tiny_arch_config
from tt.data import FullBatch, first_pad_idx, pad_to, seq_batch, to_full
from tt.util import human_bytes

if TYPE_CHECKING:
    from jax import AuxDim
    from numpy import Fin
    from optax import ArraysOf

InSeqLen = TypeVar("InSeqLen", bound=int)
OutSeqLen = TypeVar("OutSeqLen", bound=int)
VocabSize = TypeVar("VocabSize", bound=int)
BatchLen = TypeVar("BatchLen", bound=int)
EmbedDim = TypeVar("EmbedDim", bound=int)
Float = TypeVar("Float", bound=float)
Dim1 = TypeVar("Dim1", bound=int)


@jax.jit
def loss_fn(
    logits: ndarray[BatchLen, OutSeqLen, VocabSizeL, Float],
    label_ids: ndarray[BatchLen, OutSeqLen, Fin[VocabSizeL]],
) -> ndarray[Float]:
    cross_entropy_loss: ndarray[BatchLen, OutSeqLen, Float] = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=label_ids
    )
    mask: ndarray[BatchLen, OutSeqLen, bool] = label_ids != pad_token_id
    average_cross_entropy: ndarray[BatchLen, Float] = jnp.mean(cross_entropy_loss, where=mask, axis=1)
    return jnp.mean(average_cross_entropy)


def _de_aux_lm_output(
    x: AuxDim[Dim1, Output[InSeqLen, OutSeqLen, EmbedDim, VocabSize, Float]]
) -> Output[Dim1, InSeqLen, OutSeqLen, EmbedDim, VocabSize, Float]:
    return cast(Any, x)


@eqx.filter_value_and_grad(has_aux=True)
def loss_and_logits(
    lm: LM[EmbedDim, VocabSizeL, MaxSeqLenL, Float],
    batch: FullBatch[BatchLen, InSeqLen, OutSeqLen, Fin[VocabSizeL]],
) -> tuple[
    ndarray[Float],
    Output[BatchLen, InSeqLen, OutSeqLen, EmbedDim, VocabSizeL, Float],
]:
    outputs: Output[BatchLen, InSeqLen, OutSeqLen, EmbedDim, VocabSizeL, Float] = _de_aux_lm_output(
        jax.vmap(ft.partial(lm.__call__, mask_type=CausalMask()))(batch.encoder_ids, batch.decoder_ids)
    )
    loss: ndarray[Float] = loss_fn(outputs.logit, batch.label_ids)
    return loss, outputs


def tree_size(tree: Any) -> int:
    return sum(x.nbytes for x in jtu.tree_leaves(tree) if eqx.is_array(x))


def arrays_of[A](x: A) -> ArraysOf[A]:
    return cast(Any, eqx.filter(x, eqx.is_array))


def mk_step[Model: eqx.Module, Batch, Float: float, Out, OptFloat: float](
    update: optax.TransformUpdateFn,
    loss_fn: Callable[[Model, Batch], tuple[tuple[ndarray[Float], Out], eqx.Grads[Model]]],
) -> Callable[
    [Model, optax.OptState[Model, OptFloat], Batch],
    tuple[Model, optax.OptState[Model, OptFloat], ndarray[Float], Out],
]:
    @eqx.filter_jit(donate="all")
    def step(
        model: Model,
        opt_state: optax.OptState[Model, OptFloat],
        inputs: Batch,
    ) -> tuple[Model, optax.OptState[Model, OptFloat], ndarray[Float], Out]:
        (loss, out), grads = loss_fn(model, inputs)
        updates, opt_state = update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, out

    return step


def train[Model: eqx.Module, OptFloat: float, Batch, Float: float, Out](
    model: Model,
    opt_state: optax.OptState[Model, OptFloat],
    data: Iterator[Batch],
    step: Callable[
        [
            Model,
            optax.OptState[Model, OptFloat],
            Batch,
        ],
        tuple[
            Model,
            optax.OptState[Model, OptFloat],
            ndarray[Float],
            Out,
        ],
    ],
    *,
    stop_threshold: float,
) -> tuple[Model, optax.OptState[Model, OptFloat]]:
    losses = deque[float](maxlen=500)
    tqdm_bar = tqdm()
    batch = next(data)
    loss = 1000.0
    while len(losses) < 500 or mean(losses) > stop_threshold:  # noqa: PLR2004
        # Slightly weird structuring so that Python work can proceed concurrently with GPU work
        model, opt_state, loss, _ = step(model, opt_state, batch)
        next_batch = next(data)
        losses.append(loss.item())
        tqdm_bar.set_postfix(loss=loss.item())
        tqdm_bar.update()
        batch = next_batch
    return model, opt_state


def lm_main():
    lr = 2e-2
    batch_size = 256
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr))
    model = LM[EmbedSizeL, VocabSizeL, MaxSeqLenL, float32](config=tiny_arch_config, key=jax.random.PRNGKey(5678))
    print("Model size", human_bytes(tree_size(model)))
    opt_state = tx.init(arrays_of(model))
    batch_iter: Iterator[FullBatch[int, MaxSeqLenL, Literal[2], Fin[VocabSizeL]]] = (
        to_full(seq_batch(seed=s, batch_len=batch_size)) for s in it.count()
    )
    step = mk_step(tx.update, loss_and_logits)
    model, opt_state = train(model, opt_state, batch_iter, step, stop_threshold=0.02)
    return model


@eqx.filter_jit
def decode_step(
    model: LM[EmbedDim, VocabSizeL, MaxSeqLenL, Float],
    encoder_ids: ndarray[InSeqLen, Fin[VocabSizeL]],
    decoder_input_ids: ndarray[OutSeqLen, Fin[VocabSizeL]],
) -> ndarray[Fin[VocabSizeL]]:
    return jnp.argmax(
        model(encoder_ids=encoder_ids, decoder_ids=decoder_input_ids, mask_type=NoMask()).logit[
            first_pad_idx(decoder_input_ids) - 1, ...
        ]
    )


def iteratively_decode[EmbedDim: int, Float: float](
    model: LM[EmbedDim, VocabSizeL, MaxSeqLenL, Float], prompt: Sequence[Fin[VocabSizeL]]
) -> ndarray[int, Fin[VocabSizeL]]:
    """Hack sentence completion into the pre-training format. Accumulate output on the decoder."""
    encoder_ids = np.array([*prompt])
    decoder_input_ids = np.array([bos_token_id])
    # Our training data has an `OutSeqLen` of 2
    while decoder_input_ids.shape[0] <= 2:  # noqa: PLR2004
        output_token = decode_step(
            model,
            pad_to(encoder_ids, max_seq_len),
            pad_to(decoder_input_ids, 2),
        )
        decoder_input_ids = jnp.concatenate([decoder_input_ids, jnp.array([output_token])], axis=0)
    return jnp.concatenate((encoder_ids, decoder_input_ids[1:]), axis=0)
