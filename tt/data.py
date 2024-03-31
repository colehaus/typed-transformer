from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np
from jax.numpy import ndarray

from tt.config import MaxSeqLenL, VocabSizeL, bos_token_id, max_seq_len, max_usable_token, pad_token_id
from tt.util import declare_axis, unzip_list

if TYPE_CHECKING:
    from numpy import Fin


# `NamedTuple` rather than our conventional dataclass so that `JAX` automatically handles it as a pytree
class Batch[*Shape, InSeqLen, OutSeqLen, Vocab](NamedTuple):
    input_ids: ndarray[*Shape, InSeqLen, Vocab]
    label_ids: ndarray[*Shape, OutSeqLen, Vocab]


class FullBatch[*Shape, InSeqLen, OutSeqLen, Vocab](NamedTuple):
    encoder_ids: ndarray[*Shape, InSeqLen, Vocab]
    decoder_ids: ndarray[*Shape, OutSeqLen, Vocab]
    label_ids: ndarray[*Shape, OutSeqLen, Vocab]


def to_full[NumSeqs: int, InSeqLen: int, OutSeqLen: int](
    x: Batch[NumSeqs, InSeqLen, OutSeqLen, Fin[VocabSizeL]],
) -> FullBatch[NumSeqs, InSeqLen, OutSeqLen, Fin[VocabSizeL]]:
    return FullBatch(
        encoder_ids=x.input_ids,
        decoder_ids=shift_token_ids(bos_token_id, x.label_ids),
        label_ids=x.label_ids,
    )


def pad_len(n: int, pad_to_num: int):
    return (((n - 1) // pad_to_num) + 1) * pad_to_num


def pad_batch(
    inputs: list[list[Fin[VocabSizeL]]],
    outputs: list[list[Fin[VocabSizeL]]],
    *,
    input_pad_multiple: int,
    output_pad_multiple: int,
) -> Batch[Any, Any, Any, Fin[VocabSizeL]]:
    longest_input = max(len(x) for x in inputs)
    longest_output = max(len(x) for x in outputs)
    input_pad_len = pad_len(longest_input, input_pad_multiple)
    output_pad_len = pad_len(longest_output, output_pad_multiple)

    def batch_array(seqs: list[list[Fin[VocabSizeL]]], target_len: int):
        pad_array = np.full((len(seqs), target_len), pad_token_id)
        for i, seq in enumerate(seqs):
            pad_array[i, : len(seq)] = seq
        return pad_array

    return Batch(
        input_ids=batch_array(inputs, input_pad_len),
        label_ids=batch_array(outputs, output_pad_len),
    )


def first_pad_idx[*Shape, Dim1: int](
    token_ids: ndarray[*Shape, Dim1, Fin[VocabSizeL]]
) -> ndarray[*Shape, Fin[Dim1]]:
    return cast("Fin[Any]", token_ids.shape[-1]) - np.argmax(np.flip(token_ids, axis=-1) != pad_token_id, axis=-1)


def shift_token_ids[Dim1: int, Dim2: int](
    bos_token: Fin[VocabSizeL], token_ids: ndarray[Dim1, Dim2, Fin[VocabSizeL]]
) -> ndarray[Dim1, Dim2, Fin[VocabSizeL]]:
    shifted_token_ids = np.roll(token_ids, shift=1, axis=-1)
    shifted_token_ids[..., 0] = bos_token
    has_padding = np.any(token_ids == pad_token_id, axis=-1)
    return np.where(
        # Identifies each position in the matrix that is the first pad token in its row
        np.expand_dims(np.arange(shifted_token_ids.shape[-1]), 0) == np.expand_dims(first_pad_idx(token_ids), 1),
        # If the row didn't actually have padding, don't replace anything
        np.where(np.expand_dims(has_padding, 1), np.array(pad_token_id), shifted_token_ids),
        shifted_token_ids,
    )


def ascending(seed: int):
    random.seed(seed)
    in_len = random.randrange(3, max_seq_len)
    out_len = 2
    start = random.randrange(max_usable_token - in_len - out_len)
    in_seq = cast(list["Fin[VocabSizeL]"], list(range(start, start + in_len)))
    out_seq = cast(list["Fin[VocabSizeL]"], list(range(start + in_len, start + in_len + out_len)))
    return in_seq, out_seq


def descending(seed: int):
    random.seed(seed)
    in_len = random.randrange(3, max_seq_len)
    out_len = 2
    end = random.randrange(max_usable_token - in_len - out_len)
    in_seq = cast(list["Fin[VocabSizeL]"], list(range(end, end - in_len, -1)))
    out_seq = cast(list["Fin[VocabSizeL]"], list(range(end - in_len, end - in_len - out_len, -1)))
    return in_seq, out_seq


def seq_batch[NumSeqs: int](
    *, seed: int, batch_len: NumSeqs
) -> Batch[NumSeqs, MaxSeqLenL, Literal[2], Fin[VocabSizeL]]:
    def fn(seed: int):
        random.seed(seed)
        seq_fn = random.choice([ascending, descending])
        return seq_fn(seed + 1)

    ins, outs = unzip_list([fn(seed + i) for i in range(batch_len)])
    return pad_batch(ins, outs, input_pad_multiple=max_seq_len, output_pad_multiple=2)


def pad_to[Int: int](ids: ndarray[int, Fin[VocabSizeL]], final_len: Int) -> ndarray[Int, Fin[VocabSizeL]]:
    return declare_axis[Int](
        0, np.pad(ids, (0, final_len - ids.shape[0]), mode="constant", constant_values=pad_token_id)
    )
