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

import models_mixer_ct_cat_tc
import models_mixer_tc_cat_ct

class MlpMixer_mixed(nn.Module):
  """Mixer architecture. (Mixed cttc-tcct)"""
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
        if _ % 2 != 0:
            x = models_mixer_ct_cat_tc.MixerBlock_ct_cat_tc(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
        if _ % 2 == 0: 
            x = models_mixer_tc_cat_ct.MixerBlock_tc_cat_ct(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
    x = nn.LayerNorm(name='pre_head_layer_norm')(x)
    x = jnp.mean(x, axis=1)
    return nn.Dense(self.num_classes, kernel_init=nn.initializers.zeros,
                    name='head')(x)