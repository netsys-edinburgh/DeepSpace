# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the Score SDE library
# which was released under the Apache License.
#
# Source:
# https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_Apache). The modifications
# to this file are subject to the same Apache License.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: skip-file
''' Codes adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''

import functools

import numpy as np
import torch
import torch.nn as nn
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from einops import rearrange

from . import dense_layer, layers, layerspp, utils


ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one

WaveletResnetBlockBigGAN = layerspp.WaveletResnetBlockBigGANpp_Adagn

Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
default_initializer = layers.default_init
dense = dense_layer.dense


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


@utils.register_model(name='wavelet_ncsnpp')
class WaveletNCSNpp(nn.Module):
    """NCSN++ wavelet model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        self.cond_dim = cond_dim = int(config.num_channels/2)
        self.cond_emb_dim = cond_emb_dim = config.cond_emb_dim

        self.patch_size = config.patch_size
        assert config.image_size % self.patch_size == 0

        self.nf = nf = config.num_channels_dae
        ch_mult = config.ch_mult
        self.num_res_blocks = num_res_blocks = config.num_res_blocks

        # resolution of applying attention
        # mechanisms that enable the network to focus on specific parts of the input data, 
        # allowing it to weight different regions differently during processing
        self.attn_resolutions = attn_resolutions = config.attn_resolutions
        dropout = config.dropout
        resamp_with_conv = config.resamp_with_conv # up and down sample with convolution
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            (config.image_size // self.patch_size) // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.conditional  # noise-conditional
        fir = config.fir
        fir_kernel = config.fir_kernel
        self.skip_rescale = skip_rescale = config.skip_rescale
        self.resblock_type = resblock_type = config.resblock_type.lower()
        self.progressive = progressive = config.progressive.lower()
        self.progressive_input = progressive_input = config.progressive_input.lower()
        self.embedding_type = embedding_type = config.embedding_type.lower()
        init_scale = 0.
        assert progressive in ['none', 'output_skip', 'residual'] #none
        assert progressive_input in ['none', 'input_skip', 'residual'] #residual
        assert embedding_type in ['fourier', 'positional'] # positional
        combine_method = config.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        self.no_use_fbn = getattr(self.config, "no_use_fbn", False)
        self.no_use_freq = getattr(self.config, "no_use_freq", False)
        self.no_use_residual = getattr(self.config, "no_use_residual", False)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == 'positional': #yes
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional: #yes
            # incorporate time information into the embedding 
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        # self-attention mechanism
        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)
        Upsample = functools.partial(layerspp.Upsample,
                                     with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        Downsample = functools.partial(layerspp.Downsample,
                                       with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)


        if progressive_input == 'residual': #yes
            pyramid_downsample = functools.partial(layerspp.WaveletDownsample)

        if resblock_type == 'biggan': #yes  
            ResnetBlock = functools.partial(WaveletResnetBlockBigGAN,
                                                act=act,
                                                dropout=dropout,
                                                init_scale=init_scale,
                                                skip_rescale=skip_rescale,
                                                temb_dim=nf * 4,
                                                condemb_dim=cond_emb_dim)
        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block

        channels = config.num_channels * self.patch_size**2
        if progressive_input != 'none': #yes
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        hs_c2 = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                
                # resnet blocks append
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                # attention blocks append
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                hs_c2.append(in_ch)
                if resblock_type == 'ddpm': #no
                    modules.append(Downsample(in_ch=in_ch))
                else: #yes
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip': #no
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual': #yes
                    # downsample block append
                    modules.append(pyramid_downsample(
                        in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        # reverse order: common in upsampling blocks to start from the lower resolution 
        # and progressively move to higher resolutions
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            
            if i_level != 0:
                modules.append(ResnetBlock(in_ch=in_ch, up=True, hi_in_ch=hs_c2.pop()))

        assert not hs_c

        if progressive != 'output_skip': # yes
            channels = getattr(config, "num_out_channels", channels)
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, int(channels / 2), init_scale=init_scale)) # last conv exiting from the gen

        self.all_modules = nn.ModuleList(modules)

        mapping_layers = [PixelNorm(),
                          dense(cond_dim*config.image_size*config.image_size, cond_emb_dim),
                          self.act, ]

        # transforming the low-res conditioning input (config.nz) into an embedding (cond_emb_dim)
        for _ in range(config.n_mlp):
            mapping_layers.append(dense(cond_emb_dim, cond_emb_dim))
            mapping_layers.append(self.act)
        self.cond_transform = nn.Sequential(*mapping_layers)

        # wavelet pooling
        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")




    def forward(self, x, time_cond, x_cond):
        
        # concat between pure noise and low-res conditioning inputs (x + c_cond)
        x = torch.cat([x, x_cond], dim=1) 
        
        
        # patchify
        x = rearrange(x, "n c (h p1) (w p2) -> n (p1 p2 c) h w",
                      p1=self.patch_size, p2=self.patch_size)
        
        # timestep/conditional embedding
        x_cond = torch.flatten(x_cond, start_dim=1)
        condemb = self.cond_transform(x_cond)
        modules = self.all_modules
        m_idx = 0

        if self.embedding_type == 'positional': #yes
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            temb = layers.get_timestep_embedding(timesteps, self.nf)
        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional: #yes
            # time embedding
            temb = modules[m_idx](temb) #linear
            m_idx += 1
            temb = modules[m_idx](self.act(temb)) #linear
            m_idx += 1
        else:
            temb = None

        if not self.config.centered: # no
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none': #yes
            input_pyramid = x

        hs = [modules[m_idx](x)] # conv3x3 to stretch the channels (in_ch -> nf (64 or 128))
        skipHs = [] # intermediate outputs of the residual blocks used for skip connections
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, condemb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:

                h, skipH = modules[m_idx](h, temb, condemb)
                skipHs.append(skipH)
                m_idx += 1

                if self.progressive_input == 'residual': # yes
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale: #yes
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]


        h, hlh, hhl, hhh = self.dwt(h)
        h = modules[m_idx](h / 2., temb, condemb) #resblock
        h = self.iwt(h * 2., hlh, hhl, hhh)
        m_idx += 1

        # attn block
        h = modules[m_idx](h)
        m_idx += 1

        # forward on original feature space
        h = modules[m_idx](h, temb, condemb)
        h, hlh, hhl, hhh = self.dwt(h)
        h = modules[m_idx](h / 2., temb, condemb)  # forward on wavelet space - resblock
        h = self.iwt(h * 2., hlh, hhl, hhh)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, condemb)

                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if i_level != 0:
                h = modules[m_idx](h, temb, condemb, skipH=skipHs.pop())
                m_idx += 1

        assert not hs


        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)
        # unpatchify
        h = rearrange(h, "n (c p1 p2) h w -> n c (h p1) (w p2)",
                      p1=self.patch_size, p2=self.patch_size)

        if not self.not_use_tanh:
            return torch.tanh(h)
        else:
            return h
