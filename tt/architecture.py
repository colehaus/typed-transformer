from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.numpy import ndarray

from tt.attention import MHAttention
from tt.util import InstanceSingleton, declare_dtype

# fmt: off
if TYPE_CHECKING:
    from numpy import Fin
else:
    class Fin[A]: ...
# fmt: on


type KeyArray = ndarray[Literal[2], int]


# fmt: off
@dataclass(frozen=True)
class NoMask: pass  # noqa: E701
@dataclass(frozen=True)
class CausalMask: pass  # noqa: E701
type MaskType = NoMask | CausalMask
# fmt: on


def mk_mask[SeqLen: int](
    padding_mask: ndarray[SeqLen, bool], *, mask_type: MaskType
) -> ndarray[SeqLen, SeqLen, bool]:
    full_padding_mask: ndarray[SeqLen, SeqLen, bool] = jnp.expand_dims(padding_mask, axis=-1) * jnp.expand_dims(
        padding_mask, axis=-2
    )
    match mask_type:
        case NoMask():
            return full_padding_mask
        case CausalMask():
            causal_mask = jnp.tril(jnp.ones((padding_mask.shape[0], padding_mask.shape[0]), bool), k=0)
            return full_padding_mask * causal_mask


@dataclass(frozen=True)
class TransformerLayerConfig[QDim, KVDim, Float]:
    q_dim: QDim
    kv_dim: KVDim
    hidden_dim: int
    num_heads: int


@dataclass(frozen=True)
class ArchConfig[EmbedDim, VocabSize: int, MaxSeqLen: int, Float]:
    layer_config: TransformerLayerConfig[EmbedDim, EmbedDim, Float]
    num_layers: int
    vocab_size: VocabSize
    max_seq_len: MaxSeqLen
    pad_token_id: Fin[VocabSize]


class Embedder[VocabSize: int, MaxSeqLen: int, EmbedDim: int, Float: float](eqx.Module):
    token_embedder: eqx.nn.Embedding[VocabSize, EmbedDim, Float]
    position_embedder: eqx.nn.Embedding[MaxSeqLen, EmbedDim, Float]
    norm: eqx.nn.LayerNorm[EmbedDim, Float]

    def __init__(self, *, vocab_size: VocabSize, max_seq_len: MaxSeqLen, embed_size: EmbedDim, key: KeyArray):
        token_key, position_key = jax.random.split(key, 2)

        self.token_embedder = eqx.nn.Embedding(num_embeddings=vocab_size, embedding_size=embed_size, key=token_key)
        self.position_embedder = eqx.nn.Embedding(
            num_embeddings=max_seq_len, embedding_size=embed_size, key=position_key
        )
        self.norm = eqx.nn.LayerNorm(shape=embed_size)

    def __call__[SeqLen: int](
        self, token_ids: ndarray[SeqLen, Fin[VocabSize]]
    ) -> ndarray[SeqLen, EmbedDim, Float]:
        tokens: ndarray[SeqLen, EmbedDim, Float] = jax.vmap(self.token_embedder.__call__)(token_ids)
        assert token_ids.shape[0] <= self.position_embedder.num_embeddings
        positions: ndarray[SeqLen, EmbedDim, Float] = jax.vmap(self.position_embedder.__call__)(
            # jnp.arange(SeqLen) produces `ndarray[SeqLen, Fin[SeqLen]]`
            # Our `assert` guarantees we can safely cast this to `Fin[MaxSeqLen]`
            declare_dtype[Fin[MaxSeqLen]](jnp.arange(token_ids.shape[-1]))
        )
        return jax.vmap(self.norm.__call__)(tokens + positions)


class FeedForward[EmbedDim: int, Float: float](eqx.Module):
    type _HiddenDim = InstanceSingleton[Literal["HiddenSize"]]

    hidden: eqx.nn.Linear[EmbedDim, _HiddenDim, Float]
    output: eqx.nn.Linear[_HiddenDim, EmbedDim, Float]
    norm: eqx.nn.LayerNorm[EmbedDim, Float]

    def __init__(self, *, embed_dim: EmbedDim, hidden_dim: int, key: KeyArray):
        mlp_key, output_key = jax.random.split(key)
        hidden_dim_ = InstanceSingleton[Literal["HiddenSize"]](self, "HiddenSize", hidden_dim)
        self.hidden = eqx.nn.Linear(in_features=embed_dim, out_features=hidden_dim_, key=mlp_key)
        self.output = eqx.nn.Linear(in_features=hidden_dim_, out_features=embed_dim, key=output_key)
        self.norm = eqx.nn.LayerNorm(shape=embed_dim)

    def __call__(self, input_: ndarray[EmbedDim, Float]) -> ndarray[EmbedDim, Float]:
        output: ndarray[EmbedDim, Float] = self.output.__call__(
            jax.nn.gelu(self.hidden.__call__(input_)),
        )
        return self.norm.__call__(output + input_)


class CrossAttention[QDim: int, KVDim: int, Float: float](eqx.Module):
    attention: MHAttention[QDim, KVDim, KVDim, QDim, Float]
    norm: eqx.nn.LayerNorm[QDim, Float]

    def __init__(self, *, q_dim: QDim, kv_dim: KVDim, num_heads: int, key: KeyArray):
        self.attention = MHAttention(
            num_heads=num_heads, query_size=q_dim, output_size=q_dim, key_size=kv_dim, value_size=kv_dim, key=key
        )
        self.norm = eqx.nn.LayerNorm(shape=q_dim)

    def __call__[QSeqLen: int, KVSeqLen: int](
        self,
        query: ndarray[QSeqLen, QDim, Float],
        key_value: ndarray[KVSeqLen, KVDim, Float],
        query_padding_mask: ndarray[QSeqLen, bool],
        key_value_padding_mask: ndarray[KVSeqLen, bool],
    ) -> ndarray[QSeqLen, QDim, Float]:
        attn_out: ndarray[QSeqLen, QDim, Float] = self.attention.__call__(
            query=query,
            key_=key_value,
            value=key_value,
            mask=jnp.expand_dims(query_padding_mask, axis=-1) * jnp.expand_dims(key_value_padding_mask, axis=-2),
        )
        return jax.vmap(self.norm.__call__)(attn_out + query)


class SelfAttention[EmbedDim: int, Float: float](eqx.Module):
    attention: MHAttention[EmbedDim, EmbedDim, EmbedDim, EmbedDim, Float]
    norm: eqx.nn.LayerNorm[EmbedDim, Float]
    mask_type: MaskType = eqx.field(static=True)

    def __init__(self, *, embed_dim: EmbedDim, num_heads: int, mask_type: MaskType, key: KeyArray):
        self.attention = MHAttention(num_heads=num_heads, query_size=embed_dim, key=key)
        self.norm = eqx.nn.LayerNorm(shape=embed_dim)
        self.mask_type = mask_type

    def __call__[SeqLen: int](
        self, input_: ndarray[SeqLen, EmbedDim, Float], padding_mask: ndarray[SeqLen, bool]
    ) -> ndarray[SeqLen, EmbedDim, Float]:
        mask: ndarray[SeqLen, SeqLen, bool] = mk_mask(padding_mask, mask_type=self.mask_type)
        attn_out: ndarray[SeqLen, EmbedDim, Float] = self.attention.__call__(
            query=input_, key_=input_, value=input_, mask=mask
        )
        return jax.vmap(self.norm.__call__)(attn_out + input_)


class DecoderLayer[QDim: int, KVDim: int, Float: float](eqx.Module):
    self_attention: SelfAttention[QDim, Float]
    cross_attention: CrossAttention[QDim, KVDim, Float]
    feed_forward: FeedForward[QDim, Float]

    def __init__(self, config: TransformerLayerConfig[QDim, KVDim, Float], *, mask_type: MaskType, key: KeyArray):
        self_attention_key, cross_attention_key, ff_key = jax.random.split(key, num=3)
        self.self_attention = SelfAttention(
            embed_dim=config.q_dim, num_heads=config.num_heads, key=self_attention_key, mask_type=mask_type
        )
        self.cross_attention = CrossAttention(
            q_dim=config.q_dim, kv_dim=config.kv_dim, num_heads=config.num_heads, key=cross_attention_key
        )
        self.feed_forward = FeedForward(embed_dim=config.q_dim, hidden_dim=config.hidden_dim, key=ff_key)

    def __call__[QSeqLen: int, KVSeqLen: int](
        self,
        query: ndarray[QSeqLen, QDim, Float],
        key_value: ndarray[KVSeqLen, KVDim, Float],
        query_padding_mask: ndarray[QSeqLen, bool],
        key_value_padding_mask: ndarray[KVSeqLen, bool],
    ) -> ndarray[QSeqLen, QDim, Float]:
        self_attn_out: ndarray[QSeqLen, QDim, Float] = self.self_attention.__call__(query, query_padding_mask)
        cross_attn_out: ndarray[QSeqLen, QDim, Float] = self.cross_attention.__call__(
            query=self_attn_out,
            key_value=key_value,
            query_padding_mask=query_padding_mask,
            key_value_padding_mask=key_value_padding_mask,
        )
        return jax.vmap(self.feed_forward.__call__)(cross_attn_out)


class EncoderLayer[EmbedDim: int, Float: float](eqx.Module):
    self_attention: SelfAttention[EmbedDim, Float]
    feed_forward: FeedForward[EmbedDim, Float]

    def __init__(self, config: TransformerLayerConfig[EmbedDim, EmbedDim, Float], *, key: KeyArray):
        attention_key, ff_key = jax.random.split(key)
        self.self_attention = SelfAttention(
            embed_dim=config.q_dim, num_heads=config.num_heads, mask_type=NoMask(), key=attention_key
        )
        self.feed_forward = FeedForward(embed_dim=config.q_dim, hidden_dim=config.hidden_dim, key=ff_key)

    def __call__[SeqLen: int](
        self, input_: ndarray[SeqLen, EmbedDim, Float], mask: ndarray[SeqLen, bool]
    ) -> ndarray[SeqLen, EmbedDim, Float]:
        attn_out: ndarray[SeqLen, EmbedDim, Float] = self.self_attention.__call__(input_, mask)
        return jax.vmap(self.feed_forward.__call__)(attn_out)


class Decoder[QDim: int, KVDim: int, Float: float](eqx.Module):
    layers: tuple[DecoderLayer[QDim, KVDim, Float], ...]

    def __init__(
        self,
        layer_config: TransformerLayerConfig[QDim, KVDim, Float],
        *,
        num_layers: int,
        mask_type: MaskType,
        key: KeyArray,
    ):
        self.layers = tuple(
            DecoderLayer(layer_config, mask_type=mask_type, key=k) for k in jax.random.split(key, num=num_layers)
        )

    def __call__[OutSeqLen: int, InSeqLen: int](
        self,
        query: ndarray[OutSeqLen, QDim, Float],
        key_value: ndarray[InSeqLen, KVDim, Float],
        query_padding_mask: ndarray[OutSeqLen, bool],
        key_value_padding_mask: ndarray[InSeqLen, bool],
    ) -> ndarray[OutSeqLen, QDim, Float]:
        for layer in self.layers:
            query = layer.__call__(query, key_value, query_padding_mask, key_value_padding_mask)
        return query


class EmbeddingDecoder[VocabSize: int, MaxSeqLen: int, EmbedDim: int, Float: float, MaskT: MaskType](eqx.Module):
    embedder: Embedder[VocabSize, MaxSeqLen, EmbedDim, Float]
    decoder: Decoder[EmbedDim, EmbedDim, Float]
    pad_token_id: Fin[VocabSize] = eqx.field(static=True)

    def __init__(  # noqa: PLR0913
        self,
        layer_config: TransformerLayerConfig[EmbedDim, EmbedDim, Float],
        *,
        pad_token_id: Fin[VocabSize],
        vocab_size: VocabSize,
        max_seq_len: MaxSeqLen,
        num_layers: int,
        mask_type: MaskT,
        key: KeyArray,
    ):
        self.pad_token_id = pad_token_id
        emb_key, dec_key = jax.random.split(key)
        self.embedder = Embedder(
            vocab_size=vocab_size, max_seq_len=max_seq_len, embed_size=layer_config.q_dim, key=emb_key
        )
        self.decoder = Decoder(layer_config, num_layers=num_layers, mask_type=mask_type, key=dec_key)

    def __call__[OutSeqLen: int, InSeqLen: int](
        self,
        query: ndarray[OutSeqLen, Fin[VocabSize]],
        key_value: ndarray[InSeqLen, EmbedDim, Float],
        key_value_padding_mask: ndarray[InSeqLen, bool],
    ) -> ndarray[OutSeqLen, EmbedDim, Float]:
        embeds: ndarray[OutSeqLen, EmbedDim, Float] = self.embedder.__call__(query)
        return self.decoder.__call__(embeds, key_value, (query != self.pad_token_id), key_value_padding_mask)


class Encoder[EmbedDim: int, Float: float](eqx.Module):
    layers: tuple[EncoderLayer[EmbedDim, Float], ...]

    def __init__(
        self, layer_config: TransformerLayerConfig[EmbedDim, EmbedDim, Float], *, num_layers: int, key: KeyArray
    ):
        self.layers = tuple(EncoderLayer(layer_config, key=k) for k in jax.random.split(key, num=num_layers))

    def __call__[SeqLen: int](
        self, embeds: ndarray[SeqLen, EmbedDim, Float], padding_mask: ndarray[SeqLen, bool]
    ) -> ndarray[SeqLen, EmbedDim, Float]:
        for layer in self.layers:
            embeds = layer.__call__(embeds, padding_mask)
        return embeds


class EmbeddingEncoder[VocabSize: int, MaxSeqLen: int, EmbedDim: int, Float: float](eqx.Module):
    embedder: Embedder[VocabSize, MaxSeqLen, EmbedDim, Float]
    encoder: Encoder[EmbedDim, Float]
    pad_token_id: Fin[VocabSize] = eqx.field(static=True)

    def __init__(  # noqa: PLR0913
        self,
        layer_config: TransformerLayerConfig[EmbedDim, EmbedDim, Float],
        *,
        pad_token_id: Fin[VocabSize],
        vocab_size: VocabSize,
        max_seq_len: MaxSeqLen,
        num_layers: int,
        key: KeyArray,
    ):
        self.pad_token_id = pad_token_id
        emb_key, enc_key = jax.random.split(key)
        self.embedder = Embedder(
            vocab_size=vocab_size, max_seq_len=max_seq_len, embed_size=layer_config.q_dim, key=emb_key
        )
        self.encoder = Encoder(layer_config, num_layers=num_layers, key=enc_key)

    def __call__[SeqLen: int](
        self, token_ids: ndarray[SeqLen, Fin[VocabSize]]
    ) -> ndarray[SeqLen, EmbedDim, Float]:
        embeds: ndarray[SeqLen, EmbedDim, Float] = self.embedder.__call__(token_ids)
        return self.encoder.__call__(embeds, token_ids != self.pad_token_id)


# `NamedTuple` rather than our conventional dataclass so that `JAX` automatically handles it as a pytree
class Output[*Shape, InSeqLen, OutSeqLen, EmbedDim, VocabSize, Float](NamedTuple):
    encoder: ndarray[*Shape, InSeqLen, EmbedDim, Float]
    decoder: ndarray[*Shape, OutSeqLen, EmbedDim, Float]
    logit: ndarray[*Shape, OutSeqLen, VocabSize, Float]


class LM[EmbedDim: int, VocabSize: int, MaxSeqLen: int, Float: float](eqx.Module):
    encoder: EmbeddingEncoder[VocabSize, MaxSeqLen, EmbedDim, Float]
    decoder: EmbeddingDecoder[VocabSize, MaxSeqLen, EmbedDim, Float, CausalMask]
    logit: eqx.nn.Linear[EmbedDim, VocabSize, Float]
    pad_token_id: Fin[VocabSize] = eqx.field(static=True)

    def __init__(self, config: ArchConfig[EmbedDim, VocabSize, MaxSeqLen, Float], *, key: KeyArray):
        self.pad_token_id = config.pad_token_id
        encoder_key, decoder_key, logit_key = jax.random.split(key, num=3)

        self.encoder = EmbeddingEncoder(
            layer_config=config.layer_config,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            pad_token_id=config.pad_token_id,
            key=encoder_key,
        )
        self.decoder = EmbeddingDecoder(
            layer_config=config.layer_config,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            pad_token_id=config.pad_token_id,
            mask_type=CausalMask(),
            key=decoder_key,
        )
        self.logit = eqx.nn.Linear(
            in_features=config.layer_config.q_dim, out_features=config.vocab_size, key=logit_key
        )

    def __call__[InSeqLen: int, OutSeqLen: int](
        self,
        encoder_ids: ndarray[InSeqLen, Fin[VocabSize]],
        decoder_ids: ndarray[OutSeqLen, Fin[VocabSize]],
        *,
        mask_type: MaskType,
    ) -> Output[InSeqLen, OutSeqLen, EmbedDim, VocabSize, Float]:
        enc_out: ndarray[InSeqLen, EmbedDim, Float] = self.encoder.__call__(encoder_ids)
        decoder_out: ndarray[OutSeqLen, EmbedDim, Float] = self.decoder.__call__(
            decoder_ids, enc_out, (encoder_ids != self.pad_token_id)
        )
        logits: ndarray[OutSeqLen, VocabSize, Float] = jax.vmap(self.logit.__call__)(decoder_out)
        return Output(enc_out, decoder_out, logits)
