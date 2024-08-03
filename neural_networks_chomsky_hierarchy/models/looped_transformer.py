# %%
# import sys
from typing import Callable, Optional

import chex
import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp

# sys.path.append("/work/gg45/g45004/Looped-Transformer/")
from neural_networks_chomsky_hierarchy.models import (
    positional_encodings as pos_encs_lib,
)
from neural_networks_chomsky_hierarchy.models.transformer import (
    MultiHeadDotProductAttention,
    TransformerConfig,
    layer_norm,
    shift_right,
)


class EncoderBlock(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self._config = config

    def __call__(self, h, causal_mask, pos_enc_params):
        attention = MultiHeadDotProductAttention(
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
        attention = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, attention)
        attention = layer_norm(h + attention)

        # Position-wise feedforward network.
        h = hk.Linear(self._config.embedding_dim * self._config.widening_factor)(
            attention
        )
        h = jnn.relu(h)
        h = hk.Linear(self._config.embedding_dim)(h)

        h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
        h = layer_norm(h + attention)
        return h


class TransformerEncoder(hk.Module):
    """Transformer Encoder (Vaswani et al., 2017)."""

    def __init__(
        self,
        config: TransformerConfig,
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

        encoder_block = EncoderBlock(self._config)
        for _ in range(self._config.num_layers):
            h = encoder_block(h, causal_mask, pos_enc_params)
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
        self_attention = layer_norm(h + self_attention)

        # Cross-attention
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
        cross_attention = layer_norm(self_attention + cross_attention)

        # Position-wise feedforward network
        h = hk.Linear(self._config.embedding_dim * self._config.widening_factor)(
            cross_attention
        )
        h = jnn.relu(h)
        h = hk.Linear(self._config.embedding_dim)(h)

        h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
        h = layer_norm(h + cross_attention)
        return h


class TransformerDecoder(hk.Module):
    """Transformer Decoder (Vaswani et al., 2017)."""

    def __init__(
        self,
        config: TransformerConfig,
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

        decoder_block = DecoderBlock(self._config)
        for _ in range(self._config.num_layers):
            h = decoder_block(h, encoded, causal_mask)
        return h


class Transformer(hk.Module):
    """Transformer (Vaswani et al., 2017)."""

    def __init__(self, config: TransformerConfig, name: Optional[str] = None):
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
) -> Callable[[chex.Array], chex.Array]:
    """Returns a transformer encoder model."""
    if positional_encodings is None:
        positional_encodings = pos_encs_lib.PositionalEncodings.SIN_COS
        positional_encodings_params = pos_encs_lib.SinCosParams()
    elif positional_encodings_params is None:
        raise ValueError("No parameters for positional encodings are passed.")
    config = TransformerConfig(
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
    return_all_outputs: bool = False,
) -> Callable[[chex.Array, chex.Array], chex.Array]:
    """Returns a transformer model."""
    if positional_encodings is None:
        positional_encodings = pos_encs_lib.PositionalEncodings.SIN_COS
        positional_encodings_params = pos_encs_lib.SinCosParams()
    elif positional_encodings_params is None:
        raise ValueError("No parameters for positional encodings are passed.")
    config = TransformerConfig(
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
    )

    def transformer(inputs: chex.Array, targets: chex.Array) -> chex.Array:
        output = Transformer(config)(inputs, targets)
        if not return_all_outputs:
            output = output[:, -1, :]
        return hk.Linear(output_size)(output)

    return transformer


# %%
if __name__ == "__main__":
    import jax

    inputs = jnp.ones((1, 10, 10))
    outputs = jnp.ones((1, 10, 10))
    model = make_transformer(10, num_layers=100)
    # print params
    transformer = hk.transform(model)
    params = transformer.init(jax.random.PRNGKey(42), inputs, outputs)
    print(params)

    out = transformer.apply(params, jax.random.PRNGKey(42), inputs, outputs)

# %%
