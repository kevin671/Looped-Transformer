# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adaptive Computation Time layers."""

from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp


class Identity(hk.Module):
    """Identity layer (used for shunting)."""

    def __call__(self, *args):
        # Inputs and outputs must maintain the same tree structure.
        return args[0] if len(args) == 1 else args


class ActStep(hk.Module):
    """Takes an ACT step."""

    def __init__(self, ac_config: Any, layer: hk.Module):
        super().__init__()
        self.ac_config = ac_config
        self.layer = layer

    def __call__(self, inputs: Any) -> Any:
        """An act step (which either adaptively applies the layer or skips).

        Args:
          inputs: A tuple of:
            - state: An array of shape `[batch_size, length, channel]`.
            - halting_probability: An array containing the halting probs.
            - remainders: An array containing the act remainders.
            - n_updates: An array containing the act n_updates.
            - previous_state: An array that has the previous state.
            - layer_call_args: Arguments to be passed to the self.layer.

        Returns:
          A tuple of (output_state, new halting_probabilities,
            updated remainders, updated n_updates, new_state).
        """
        threshold = 1.0 - self.ac_config.act_epsilon
        act_type = self.ac_config.act_type
        halting_bias_init = self.ac_config.act_halting_bias_init
        act_level = self.ac_config.act_level

        (
            state,
            halting_probability,
            remainders,
            n_updates,
            previous_state,
            *layer_call_args,
        ) = inputs
        if act_type == "random":
            rng = hk.next_rng_key()
            p = jax.random.uniform(rng, shape=halting_probability.shape)
        else:
            halting_layer = hk.Linear(
                output_size=1,
                with_bias=True,
                w_init=hk.initializers.Constant(0.0),
                b_init=hk.initializers.Constant(halting_bias_init),
                name="step_halting_prob",
            )
            p = hk.sigmoid(halting_layer(state))

            if act_level == "per_example":
                p = jnp.mean(p, axis=1)
            p = jnp.squeeze(p, axis=-1)

        still_running = jnp.less(halting_probability, 1.0).astype(jnp.float32)
        new_halted = (
            jnp.greater(halting_probability + p * still_running, threshold).astype(
                jnp.float32
            )
            * still_running
        )
        still_running = (
            jnp.less_equal(halting_probability + p * still_running, threshold).astype(
                jnp.float32
            )
            * still_running
        )
        halting_probability += p * still_running
        remainders += new_halted * (1 - halting_probability)
        halting_probability += new_halted * remainders
        n_updates += still_running + new_halted

        update_weights = jnp.expand_dims(
            p * still_running + new_halted * remainders, -1
        )
        if act_level == "per_example":
            update_weights = jnp.expand_dims(update_weights, -1)

        output_state = self.layer(state, *layer_call_args)

        if act_type in ["basic", "random"]:
            new_state = (output_state * update_weights) + (
                previous_state * (1 - update_weights)
            )
        elif act_type == "accumulated":
            new_state = (output_state * update_weights) + previous_state
        else:
            raise ValueError(f"Unknown act_type {act_type}!")

        return (
            output_state,
            halting_probability,
            remainders,
            n_updates,
            new_state,
            *layer_call_args,
        )


class ACTFunction(hk.Module):
    def __init__(self, ac_config, layer, stop_fn):
        super().__init__(name="ACTFunction")
        self.ac_config = ac_config
        self.layer = layer
        self.stop_fn = stop_fn
        self.act_step = ActStep(
            ac_config=self.ac_config, layer=self.layer, name="act_step"
        )

    def take_a_step(self, x):
        return self.act_step(x)

    def skip_a_step(self, x):
        return x

    def __call__(self, x, params):
        if params is None:
            out = self.take_a_step(x)
        else:
            decision = self.stop_fn(x)
            out = jax.lax.cond(
                decision,
                lambda _: self.skip_a_step(x),
                lambda _: self.take_a_step(x),
                None,
            )
        return out, None


class AdaptiveComputationTime(hk.Module):
    def __init__(self, ac_config, layer, share_parameters):
        super().__init__(name="AdaptiveComputationTime")
        self.ac_config = ac_config
        self.layer = layer
        self.share_parameters = share_parameters

    def __call__(self, x, *layer_call_args):
        threshold = 1.0 - self.ac_config.act_epsilon
        max_steps = self.ac_config.act_max_steps

        state = x
        original_state_shape = state.shape

        if self.ac_config.act_level == "per_example":
            state_slice = slice(0, 1)
        elif self.ac_config.act_level == "per_token":
            state_slice = slice(0, 2)
        else:
            raise ValueError(f"Unknown act_level {self.ac_config.act_level}")

        update_shape = state.shape[state_slice]
        halting_probability = jnp.zeros(update_shape)
        remainders = jnp.zeros(update_shape)
        n_updates = jnp.zeros(update_shape)
        previous_state = jnp.zeros_like(state)

        def stop_fn(inputs):
            _, halting_probability, _, _, _, *_ = inputs
            return jnp.all(jnp.greater_equal(halting_probability, threshold))

        intermedia_output = (
            state,
            halting_probability,
            remainders,
            n_updates,
            previous_state,
            *layer_call_args,
        )

        if self.share_parameters:

            def scan_fn(carry, _):
                (
                    state,
                    halting_probability,
                    remainders,
                    n_updates,
                    previous_state,
                    *layer_call_args,
                ) = carry
                act_fn = ACTFunction(self.ac_config, self.layer, stop_fn)
                output, _ = act_fn(state, None)
                return output, None

            output, _ = hk.scan(scan_fn, intermedia_output, None, length=max_steps)

        else:

            def scan_fn(carry, _):
                (
                    state,
                    halting_probability,
                    remainders,
                    n_updates,
                    previous_state,
                    *layer_call_args,
                ) = carry
                act_fn = ACTFunction(self.ac_config, self.layer, stop_fn)
                output, _ = act_fn(state, None)
                return output, None

            output, _ = hk.scan(scan_fn, intermedia_output, None, length=max_steps)

        (
            output_state,
            halting_probability,
            remainders,
            ponder_times,
            new_state,
            *layer_call_args,
        ) = output

        assert output_state.shape == new_state.shape == original_state_shape
        for x in [halting_probability, remainders, n_updates]:
            assert x.shape == original_state_shape[state_slice]
        return new_state, (ponder_times, remainders)
