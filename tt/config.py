from __future__ import annotations

from typing import Literal

from numpy import float32

from tt.architecture import ArchConfig, TransformerLayerConfig
from tt.util import fin

type VocabSizeL = Literal[128]
vocab_size: VocabSizeL = 128
pad_token_id = fin[VocabSizeL](127, vocab_size)
bos_token_id = fin[VocabSizeL](126, vocab_size)
max_usable_token = fin[VocabSizeL](125, vocab_size)

type MaxSeqLenL = Literal[16]
max_seq_len: MaxSeqLenL = 16

type EmbedSizeL = Literal[128]
embed_size_l: EmbedSizeL = 128


tiny_arch_config = ArchConfig[EmbedSizeL, VocabSizeL, MaxSeqLenL, float32](
    vocab_size=vocab_size,
    num_layers=4,
    layer_config=TransformerLayerConfig(
        q_dim=embed_size_l,
        kv_dim=embed_size_l,
        hidden_dim=embed_size_l * 4,
        num_heads=2,
    ),
    max_seq_len=max_seq_len,
    pad_token_id=pad_token_id,
)
