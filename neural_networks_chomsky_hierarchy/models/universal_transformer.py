# https://github.com/google-research/scenic/tree/main/scenic/projects/baselines/universal_transformer
import dataclasses
from typing import Callable, Optional

import chex
import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp

from neural_networks_chomsky_hierarchy.models import (
    positional_encodings as pos_encs_lib,
)
from neural_networks_chomsky_hierarchy.models.act import AdaptiveComputationTime
from neural_networks_chomsky_hierarchy.models.transformer import (
    MultiHeadDotProductAttention,
    layer_norm,
    shift_right,
)


@chex.dataclass
class AdaptiveComputationTimeConfig:
    act_max_steps: int = 12
    act_epsilon: float = 0.01
    act_type: str = "basic"  # 'random', 'accumulated', 'basic'
    act_level: str = "per_example"  # 'per_example', 'per_token'
    act_halting_bias_init: float = -1.0
    act_loss_weight: float = 0.01


@chex.dataclass
class UniversalTransformerConfig:
    """Hyperparameters used in the Transformer architectures."""

    # The size of the model output (i.e., the output vocabulary size).
    output_size: int
    # The dimension of the first embedding.
    embedding_dim: int = 64
    # The number of multi-head attention layers.
    num_layers: int = 5
    # The number of heads per layer.
    num_heads: int = 8
    # The number of hidden neurons per head. If None, it is set to be equal to
    # `embedding_dim // num_heads`.
    num_hiddens_per_head: Optional[int] = None
    # The probability that each element is discarded by the dropout modules.
    dropout_prob: float = 0.1
    # The parameter initialization scale for the embeddings.
    emb_init_scale: float = 0.02
    # Whether to use the embeddings rather than raw inputs.
    use_embeddings: bool = True
    # Whether to share embeddings between the Encoder and the Decoder.
    share_embeddings: bool = False
    # The size of the sliding attention window. See MultiHeadDotProductAttention.
    attention_window: Optional[int] = None
    # The positional encoding used with default sin/cos (Vaswani et al., 2017).
    positional_encodings: pos_encs_lib.PositionalEncodings = dataclasses.field(
        default_factory=lambda: pos_encs_lib.PositionalEncodings.SIN_COS
    )
    # The maximum size of the context (used by the posiitonal encodings).
    max_time: int = 10_000
    # The parameters for the positional encodings, default sin/cos.
    positional_encodings_params: pos_encs_lib.PositionalEncodingsParams = (
        dataclasses.field(default_factory=pos_encs_lib.SinCosParams)
    )
    # How much larger the hidden layer of the feedforward network should be
    # compared to the `embedding_dim`.
    widening_factor: int = 4
    # Add mask to make causal predictions.
    causal_masking: bool = False

    def __post_init__(self) -> None:
        """Sets `num_hiddens_per_head` if it is `None`."""
        if self.num_hiddens_per_head is None:
            self.num_hiddens_per_head = self.embedding_dim // self.num_heads

    deterministic: Optional[bool] = None
    stochastic_depth: float = 0.1  # 0.0
    parameter_sharing: bool = False
    ac_config: Optional[AdaptiveComputationTimeConfig] = AdaptiveComputationTimeConfig()


class UTStochasticDepth(hk.Module):
    """Performs layer-dropout (also known as stochastic depth).

    Described in
    Huang & Sun et al, "Deep Networks with Stochastic Depth", 2016
    https://arxiv.org/abs/1603.09382

    Attributes:
      rate: the layer dropout probability (_not_ the keep rate!).
      deterministic: If false (e.g. in training) the inputs are scaled by `1 / (1
        - rate)` and the layer dropout is applied, whereas if true (e.g. in
        evaluation), no stochastic depth is applied and the inputs are returned as
        is.
    """

    def __init__(self, rate: float = 0.0):
        super().__init__()
        self.rate = rate

    def __call__(
        self, x: jnp.ndarray, deterministic: Optional[bool] = None
    ) -> jnp.ndarray:
        """Applies a stochastic depth mask to the inputs.

        Args:
          x: Input tensor.
          deterministic: If false (e.g. in training) the inputs are scaled by `1 /
            (1 - rate)` and the layer dropout is applied, whereas if true (e.g. in
            evaluation), no stochastic depth is applied and the inputs are returned
            as is.

        Returns:
          The masked inputs reweighted to preserve mean.
        """
        if self.rate <= 0.0 or deterministic:
            return x

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rng = hk.next_rng_key()
        mask = jax.random.bernoulli(rng, self.rate, shape)
        return x * (1.0 - mask)


class EncoderBlock(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self._config = config

    def __call__(self, inputs, causal_mask, pos_enc_params):
        h = layer_norm(inputs)

        h = MultiHeadDotProductAttention(
            num_heads=self._config.num_heads,
            num_hiddens_per_head=self._config.num_hiddens_per_head,
            positional_encodings=self._config.positional_encodings,
            positional_encodings_params=pos_enc_params,
            attention_window=self._config.attention_window,
        )(
            inputs_q=h,
            inputs_kv=h,
            mask=causal_mask,
            causal=self._config.causal_masking,
        )
        h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)

        h = UTStochasticDepth(rate=self._config.stochastic_depth)(
            h, self._config.deterministic
        )
        h = h + inputs

        # Position-wise feedforward network.
        t = layer_norm(h)
        t = hk.Linear(self._config.embedding_dim * self._config.widening_factor)(t)
        t = jnn.relu(t)
        t = hk.Linear(self._config.embedding_dim)(t)
        t = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, t)
        t = UTStochasticDepth(rate=self._config.stochastic_depth)(
            t, self._config.deterministic
        )
        return t + h


class TransformerEncoder(hk.Module):
    """Transformer Encoder (Vaswani et al., 2017)."""

    def __init__(
        self,
        config: UniversalTransformerConfig,
        shared_embeddings_fn: Optional[Callable[[chex.Array], chex.Array]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initializes the transformer encoder.

        Args:
          config: The hyperparameters used in Transformer architectures.
          shared_embeddings_fn: Embedding function that is shared with the decoder.
          name: The name of the module.
        """
        super().__init__(name=name)
        self._config = config
        self._shared_embeddings_fn = shared_embeddings_fn

    def __call__(self, x: jnp.ndarray) -> chex.Array:
        """Returns the transformer encoder output, shape [B, T, E]."""
        if self._config.use_embeddings:
            if self._shared_embeddings_fn is not None:
                embeddings = self._shared_embeddings_fn(x)
            else:
                # Since `x` is one-hot encoded, using hk.Linear is equivalent to
                # hk.Embed with hk.EmbedLookupStyle.ONE_HOT.
                embs_init = hk.initializers.TruncatedNormal(
                    stddev=self._config.emb_init_scale
                )
                embeddings = hk.Linear(
                    self._config.embedding_dim, with_bias=False, w_init=embs_init
                )(x)

            embeddings *= jnp.sqrt(self._config.embedding_dim)

        else:
            embeddings = x

        batch_size, sequence_length, embedding_size = embeddings.shape

        pos_enc_params = self._config.positional_encodings_params
        if (
            self._config.positional_encodings
            == pos_encs_lib.PositionalEncodings.SIN_COS
        ):
            pos_encodings = pos_encs_lib.sinusoid_position_encoding(
                sequence_length=sequence_length,
                hidden_size=embedding_size,
                memory_length=0,
                max_timescale=pos_enc_params.max_time,
                min_timescale=2,
                clamp_length=0,
                causal=True,
            )
            h = embeddings + pos_encodings
            h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
        else:
            h = embeddings

        # The causal mask is shared across heads.
        if self._config.causal_masking:
            causal_mask = jnp.tril(
                jnp.ones((batch_size, 1, sequence_length, sequence_length))
            )
        else:
            causal_mask = None

        if self._config.ac_config is None:
            if not self._config.parameter_sharing:
                for _ in range(self._config.num_layers):
                    h = EncoderBlock(self._config)(h, causal_mask, pos_enc_params)
            else:
                encoder_block = EncoderBlock(self._config)
                for _ in range(self._config.num_layers):
                    h = encoder_block(h, causal_mask, pos_enc_params)
        else:
            encoder_block = EncoderBlock(self._config)
            h, _ = AdaptiveComputationTime(
                self._config.ac_config,
                encoder_block,
                self._config.parameter_sharing,
                name="act",
            )(h)
        return h


class DecoderBlock(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self._config = config

    def __call__(self, h, encoded, causal_mask):
        # Self-attention
        self_attention = MultiHeadDotProductAttention(
            num_heads=self._config.num_heads,
            num_hiddens_per_head=self._config.num_hiddens_per_head,
            positional_encodings=self._config.positional_encodings,
            positional_encodings_params=self._config.positional_encodings_params,
            attention_window=self._config.attention_window,
        )(inputs_q=h, inputs_kv=h, mask=causal_mask, causal=True)
        self_attention = hk.dropout(
            hk.next_rng_key(), self._config.dropout_prob, self_attention
        )

        self_attention = UTStochasticDepth(rate=self._config.stochastic_depth)(
            self_attention, self._config.deterministic
        )
        h = h + self_attention

        # Cross-attention
        self_attention = layer_norm(h)
        cross_attention = MultiHeadDotProductAttention(
            num_heads=self._config.num_heads,
            num_hiddens_per_head=self._config.num_hiddens_per_head,
            positional_encodings=self._config.positional_encodings,
            positional_encodings_params=self._config.positional_encodings_params,
            attention_window=self._config.attention_window,
        )(inputs_q=self_attention, inputs_kv=encoded, causal=True)
        cross_attention = hk.dropout(
            hk.next_rng_key(), self._config.dropout_prob, cross_attention
        )
        cross_attention = UTStochasticDepth(rate=self._config.stochastic_depth)(
            cross_attention, self._config.deterministic
        )
        h = h + cross_attention

        # Position-wise feedforward network
        cross_attention = layer_norm(h)
        t = hk.Linear(self._config.embedding_dim * self._config.widening_factor)(
            cross_attention
        )
        t = jnn.relu(t)
        t = hk.Linear(self._config.embedding_dim)(t)
        t = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, t)

        t = UTStochasticDepth(rate=self._config.stochastic_depth)(
            t, self._config.deterministic
        )
        return t + h


class TransformerDecoder(hk.Module):
    """Transformer Decoder (Vaswani et al., 2017)."""

    def __init__(
        self,
        config: UniversalTransformerConfig,
        shared_embeddings_fn: Optional[Callable[[chex.Array], chex.Array]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initializes the Transformer decoder.

        Args:
          config: The hyperparameters used in Transformer architectures.
          shared_embeddings_fn: Embedding function that is shared with the encoder.
          name: The name of the module.
        """
        super().__init__(name=name)
        self._config = config
        self._shared_embeddings_fn = shared_embeddings_fn

    def __call__(self, encoded: chex.Array, targets: chex.Array) -> chex.Array:
        """Returns the transformer decoder output, shape [B, T_O, E].

        Args:
          encoded: The output of the encoder, shape [B, T_I, E].
          targets: The one-hot encoded target values, shape [B, T_O, 2].
        """
        targets = shift_right(targets, self._config.output_size)

        if self._config.use_embeddings:
            if self._shared_embeddings_fn is not None:
                output_embeddings = self._shared_embeddings_fn(targets)
            else:
                # Since `x` is one-hot encoded, using hk.Linear is equivalent to
                # hk.Embed with hk.EmbedLookupStyle.ONE_HOT.
                embs_init = hk.initializers.TruncatedNormal(
                    stddev=self._config.emb_init_scale
                )
                output_embeddings = hk.Linear(
                    self._config.embedding_dim, with_bias=False, w_init=embs_init
                )(targets)

            output_embeddings *= jnp.sqrt(self._config.embedding_dim)

        else:
            output_embeddings = targets

        batch_size, output_sequence_length, embedding_size = output_embeddings.shape

        if (
            self._config.positional_encodings
            == pos_encs_lib.PositionalEncodings.SIN_COS
        ):
            pos_encodings = pos_encs_lib.sinusoid_position_encoding(
                sequence_length=output_sequence_length,
                hidden_size=embedding_size,
                memory_length=0,
                max_timescale=self._config.positional_encodings_params.max_time,
                min_timescale=2,
                clamp_length=0,
                causal=True,
            )
            h = output_embeddings + pos_encodings
            h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
        else:
            h = output_embeddings

        # The causal mask is shared across heads.
        causal_mask = jnp.tril(
            jnp.ones((batch_size, 1, output_sequence_length, output_sequence_length))
        )

        if self._config.ac_config is None:
            if not self._config.parameter_sharing:
                for _ in range(self._config.num_layers):
                    h = DecoderBlock(self._config)(h, encoded, causal_mask)
            else:
                decoder_block = DecoderBlock(self._config)
                for _ in range(self._config.num_layers):
                    h = decoder_block(h, encoded, causal_mask)
            return h
        else:
            decoder_block = EncoderBlock(self._config)
            h, _ = AdaptiveComputationTime(
                self._config.ac_config,
                decoder_block,
                self._config.parameter_sharing,
                name="act",
            )(h)
        return h


class Transformer(hk.Module):
    """Transformer (Vaswani et al., 2017)."""

    def __init__(self, config: UniversalTransformerConfig, name: Optional[str] = None):
        """Initializes the Transformer.

        Args:
          config: The hyperparameters used in Transformer architectures.
          name: The name of the module.
        """
        super().__init__(name=name)
        shared_embeddings_fn = None

        if config.share_embeddings:
            shared_embeddings_fn = hk.Linear(
                config.embedding_dim,
                with_bias=False,
                w_init=hk.initializers.TruncatedNormal(stddev=config.emb_init_scale),
                name="shared_embeddings",
            )

        self._encoder = TransformerEncoder(config, shared_embeddings_fn)
        self._decoder = TransformerDecoder(config, shared_embeddings_fn)

    def __call__(self, inputs: chex.Array, targets: chex.Array) -> chex.Array:
        return self._decoder(self._encoder(inputs), targets)


def make_transformer_encoder(
    output_size: int,
    embedding_dim: int = 64,
    num_layers: int = 5,
    num_heads: int = 8,
    num_hiddens_per_head: Optional[int] = None,
    dropout_prob: float = 0.1,
    emb_init_scale: float = 0.02,
    use_embeddings: bool = True,
    share_embeddings: bool = False,
    attention_window: Optional[int] = None,
    positional_encodings: Optional[pos_encs_lib.PositionalEncodings] = None,
    positional_encodings_params: Optional[
        pos_encs_lib.PositionalEncodingsParams
    ] = None,
    widening_factor: int = 4,
    return_all_outputs: bool = False,
    causal_masking: bool = False,
    stochastic_depth: float = 0.1,
) -> Callable[[chex.Array], chex.Array]:
    """Returns a transformer encoder model."""
    if positional_encodings is None:
        positional_encodings = pos_encs_lib.PositionalEncodings.SIN_COS
        positional_encodings_params = pos_encs_lib.SinCosParams()
    elif positional_encodings_params is None:
        raise ValueError("No parameters for positional encodings are passed.")
    config = UniversalTransformerConfig(
        output_size=output_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_hiddens_per_head=num_hiddens_per_head,
        dropout_prob=dropout_prob,
        emb_init_scale=emb_init_scale,
        use_embeddings=use_embeddings,
        share_embeddings=share_embeddings,
        attention_window=attention_window,
        positional_encodings=positional_encodings,
        positional_encodings_params=positional_encodings_params,
        widening_factor=widening_factor,
        causal_masking=causal_masking,
        stochastic_depth=stochastic_depth,
    )

    def transformer_encoder(inputs: chex.Array) -> chex.Array:
        output = TransformerEncoder(config)(inputs)
        if not return_all_outputs:
            output = output[:, -1, :]
        return hk.Linear(output_size)(output)

    return transformer_encoder


def make_transformer(
    output_size: int,
    embedding_dim: int = 64,
    num_layers: int = 5,
    num_heads: int = 8,
    num_hiddens_per_head: Optional[int] = None,
    dropout_prob: float = 0.1,
    emb_init_scale: float = 0.02,
    use_embeddings: bool = True,
    share_embeddings: bool = False,
    attention_window: Optional[int] = None,
    positional_encodings: Optional[pos_encs_lib.PositionalEncodings] = None,
    positional_encodings_params: Optional[
        pos_encs_lib.PositionalEncodingsParams
    ] = None,
    widening_factor: int = 4,
    stochastic_depth: float = 0.1,
    return_all_outputs: bool = False,
) -> Callable[[chex.Array, chex.Array], chex.Array]:
    """Returns a transformer model."""
    if positional_encodings is None:
        positional_encodings = pos_encs_lib.PositionalEncodings.SIN_COS
        positional_encodings_params = pos_encs_lib.SinCosParams()
    elif positional_encodings_params is None:
        raise ValueError("No parameters for positional encodings are passed.")
    config = UniversalTransformerConfig(
        output_size=output_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_hiddens_per_head=num_hiddens_per_head,
        dropout_prob=dropout_prob,
        emb_init_scale=emb_init_scale,
        use_embeddings=use_embeddings,
        share_embeddings=share_embeddings,
        attention_window=attention_window,
        positional_encodings=positional_encodings,
        positional_encodings_params=positional_encodings_params,
        widening_factor=widening_factor,
        stochastic_depth=stochastic_depth,
    )

    def transformer(inputs: chex.Array, targets: chex.Array) -> chex.Array:
        output = Transformer(config)(inputs, targets)
        if not return_all_outputs:
            output = output[:, -1, :]
        return hk.Linear(output_size)(output)

    return transformer
