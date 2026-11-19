# t5

## Overview

The T5 encoder and decoder stacks (Raffel et al., "Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer", JMLR 2020) that Chronos, Chronos-Bolt and MOMENT build on: root-mean-square layer
normalization, unscaled attention with a bucketed relative position bias, and ReLU or gated-GELU feed-forward
networks. The stacks take embeddings instead of token ids and compute exactly what the Hugging Face `transformers`
implementation computes, so published T5 weights load through `T5OfficialNames`.

## Components

| Component | Description |
| --------- | ----------- |
| [configuration.py](./configuration.py) | `T5Configuration`: sizes of a stack, named after the published config fields. |
| [feed_forward_kind.py](./feed_forward_kind.py) | `T5FeedForwardKind`: ReLU or gated-GELU feed-forward networks. |
| [layer_norm.py](./layer_norm.py) | `T5LayerNorm`: root-mean-square layer normalization without bias. |
| [relative_position_bias.py](./relative_position_bias.py) | `T5RelativePositionBias`: learned bias per head and bucket of relative positions. |
| [attention_mask.py](./attention_mask.py) | `T5AttentionMask`: padding and causal masks as additive biases. |
| [attention.py](./attention.py) | `T5Attention`: multi-head attention without scaling or biases. |
| [feed_forward.py](./feed_forward.py) | `T5FeedForward`: position-wise feed-forward network. |
| [encoder_block.py](./encoder_block.py) | `T5EncoderBlock`: self-attention and feed-forward network. |
| [decoder_block.py](./decoder_block.py) | `T5DecoderBlock`: causal self-attention, cross-attention and feed-forward network. |
| [encoder.py](./encoder.py) | `T5Encoder`: the encoder stack. |
| [decoder.py](./decoder.py) | `T5Decoder`: the decoder stack, over whole sequences or step by step with caches. |
| [decoder_layer_cache.py](./decoder_layer_cache.py) | `T5DecoderLayerCache`: keys and values of one decoder block between decoding steps. |
| [official_names.py](./official_names.py) | `T5OfficialNames`: renamings of the published `transformers` parameter names. |
