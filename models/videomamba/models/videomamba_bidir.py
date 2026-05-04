from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from models.videomamba.models.videomamba import (
    Block,
    PatchEmbed,
    inflate_weight,
    segm_init_weights,
    _init_weights,
)


class Bimamba(Mamba):
    """Bidirectional Mamba block (bimamba_type='v2'): adds backward-direction
    counterparts of A_log, D, conv1d, x_proj, dt_proj. Shares in_proj and
    out_proj with the forward direction. Used by VideoMamba for SSv2.

    Compatible with the released OpenGVLab VideoMamba checkpoints whose state
    dicts contain `A_b_log`, `D_b`, `conv1d_b.{weight,bias}`,
    `x_proj_b.weight`, `dt_proj_b.{weight,bias}` per layer.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="v2",
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=False,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
        self.bimamba_type = bimamba_type

        factory_kwargs = {"device": device, "dtype": dtype}

        A_b = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_b_log = nn.Parameter(torch.log(A_b))
        self.A_b_log._no_weight_decay = True

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory_kwargs
        )

        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_b.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_b.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj_b.bias.copy_(inv_dt)
        self.dt_proj_b.bias._no_reinit = True

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_b._no_weight_decay = True

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, _ = hidden_states.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)

        A = -torch.exp(self.A_log.float())
        x_f = F.silu(self.conv1d(x)[..., :seqlen])
        x_dbl = self.x_proj(rearrange(x_f, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj.weight @ dt.t(), "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) d -> b d l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) d -> b d l", l=seqlen).contiguous()
        y_f = selective_scan_fn(
            x_f, dt, A, B, C, self.D.float(),
            z=z, delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
        )

        A_b = -torch.exp(self.A_b_log.float())
        x_flip = torch.flip(x, dims=[2])
        z_flip = torch.flip(z, dims=[2])
        x_b = F.silu(self.conv1d_b(x_flip)[..., :seqlen])
        x_dbl_b = self.x_proj_b(rearrange(x_b, "b d l -> (b l) d"))
        dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_b = rearrange(self.dt_proj_b.weight @ dt_b.t(), "d (b l) -> b d l", l=seqlen)
        B_b = rearrange(B_b, "(b l) d -> b d l", l=seqlen).contiguous()
        C_b = rearrange(C_b, "(b l) d -> b d l", l=seqlen).contiguous()
        y_b = selective_scan_fn(
            x_b, dt_b, A_b, B_b, C_b, self.D_b.float(),
            z=z_flip, delta_bias=self.dt_proj_b.bias.float(), delta_softplus=True,
        )
        y_b = torch.flip(y_b, dims=[2])

        y = y_f + y_b
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


def create_block_bidir(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba_type="v2",
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(
        Bimamba,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        **ssm_cfg,
        **factory_kwargs,
    )
    norm_cls = partial(nn.LayerNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class VisionMambaBidir(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        depth=24,
        embed_dim=192,
        channels=3,
        num_classes=174,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        initializer_cfg=None,
        fused_add_norm=False,
        rms_norm=False,
        residual_in_fp32=True,
        bimamba_type="v2",
        kernel_size=1,
        num_frames=8,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head = (
            nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layers = nn.ModuleList(
            [
                create_block_bidir(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm_f = nn.LayerNorm(embed_dim, eps=norm_epsilon, **factory_kwargs)

        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.temporal_pos_embedding, std=0.02)
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x)  # (B*T, C', H', W') -> see PatchEmbed
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # spatial pos embed
        x = x + self.pos_embed
        # temporal pos embed (broadcast across spatial tokens)
        cls_tokens = x[:, :1]
        x = x[:, 1:]
        x = rearrange(x, "b (t n) m -> (b n) t m", t=T)
        x = x + self.temporal_pos_embedding
        x = rearrange(x, "(b n) t m -> b (t n) m", b=B)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)

        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))
        return hidden_states[:, 0]

    def forward(self, x, inference_params=None):
        x = self.forward_features(x, inference_params)
        return self.head(x)


@register_model
def videomamba_small_bidir(pretrained=False, **kwargs):
    model = VisionMambaBidir(
        patch_size=16,
        embed_dim=384,
        depth=24,
        rms_norm=False,
        residual_in_fp32=True,
        fused_add_norm=False,
        bimamba_type="v2",
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def videomamba_tiny_bidir(pretrained=False, **kwargs):
    model = VisionMambaBidir(
        patch_size=16,
        embed_dim=192,
        depth=24,
        rms_norm=False,
        residual_in_fp32=True,
        fused_add_norm=False,
        bimamba_type="v2",
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def videomamba_middle_bidir(pretrained=False, **kwargs):
    model = VisionMambaBidir(
        patch_size=16,
        embed_dim=576,
        depth=32,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        bimamba_type="v2",
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
