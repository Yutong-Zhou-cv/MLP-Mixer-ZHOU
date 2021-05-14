# Copyright 2021 Google LLC.
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

from typing import Any

import einops
import flax.linen as nn
import jax.numpy as jnp


class MlpBlock(nn.Module):
  mlp_dim: int

  @nn.compact
  def __call__(self, x):
    y = nn.Dense(self.mlp_dim)(x)
    y = nn.gelu(y)
    return nn.Dense(x.shape[-1])(y)


class MixerBlock_c_cat_t(nn.Module):
  """Mixer block layer. (channel_mixing concat token_mixing)"""
  tokens_mlp_dim: int
  channels_mlp_dim: int

  @nn.compact
  def __call__(self, x):
    y_channel = nn.LayerNorm()(x)
    y_channel = MlpBlock(self.channels_mlp_dim, name='channel_mixing')(y_channel)
    channel_mixing = x + y_channel
    y_token = nn.LayerNorm()(x)
    y_token = jnp.swapaxes(y_token, 1, 2)
    y_token = MlpBlock(self.tokens_mlp_dim, name='token_mixing')(y_token)
    y_token = jnp.swapaxes(y_token, 1, 2)
    token_mixing = x + y_token
    channel_cat_token = jnp.concatenate(channel_mixing, token_mixing)
    return channel_cat_token


class MlpMixer_c_cat_t(nn.Module):
  """Mixer architecture. (channel_mixing concat token_mixing)"""
  patches: Any
  num_classes: int
  num_blocks: int
  hidden_dim: int
  tokens_mlp_dim: int
  channels_mlp_dim: int

  @nn.compact
  def __call__(self, inputs, *, train):
    del train
    x = nn.Conv(self.hidden_dim, self.patches.size,
                strides=self.patches.size, name='stem')(inputs)
    x = einops.rearrange(x, 'n h w c -> n (h w) c')
    for _ in range(self.num_blocks):
      x = MixerBlock_c_cat_t(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
    x = nn.LayerNorm(name='pre_head_layer_norm')(x)
    x = jnp.mean(x, axis=1)
    return nn.Dense(self.num_classes, kernel_init=nn.initializers.zeros,
                    name='head')(x)