# src\comfyui_remote\custom_nodes\ddcoloring\core.py
"""
ddcoloring.core
Standalone DDColor implementation (model + inference pipelines) for ComfyUI.
Includes optional reference-image palette steering and optical-flow temporal smoothing.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from enum import Enum, auto
from timm.layers import DropPath, trunc_normal_


# =============================================================================
# ENCODER (ConvNeXt)
# =============================================================================

ENCODER_CONFIGS = {
    "convnext-t": {"depths": (3, 3, 9, 3), "dims": (96, 192, 384, 768)},
    "convnext-l": {"depths": (3, 3, 27, 3), "dims": (192, 384, 768, 1536)},
}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), drop_path_rate=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        for i in range(4):
            layer = LayerNorm(dims[i], eps=1e-6, data_format="channels_first")
            self.add_module(f"norm{i}", layer)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            norm_layer = getattr(self, f"norm{i}")
            norm_layer(x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        return self.forward_features(x)


class Hook:
    feature = None
    def __init__(self, module):
        self.hook = module.register_forward_hook(self._hook_fn)
    def _hook_fn(self, module, input, output):
        self.feature = output if isinstance(output, torch.Tensor) else output.get("out", output)
    def remove(self):
        self.hook.remove()


# =============================================================================
# DECODER
# =============================================================================

class NormType(Enum):
    Batch = auto()
    BatchZero = auto()
    Weight = auto()
    Spectral = auto()


def batchnorm_2d(nf, norm_type=NormType.Batch):
    bn = nn.BatchNorm2d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0.0 if norm_type == NormType.BatchZero else 1.0)
    return bn


def init_default(m, func=nn.init.kaiming_normal_):
    if func and hasattr(m, "weight"):
        func(m.weight)
    if hasattr(m, "bias") and hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)
    return m


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


def custom_conv_layer(ni, nf, ks=3, stride=1, padding=None, bias=None, norm_type=NormType.Batch, use_activ=True, transpose=False, init=nn.init.kaiming_normal_, self_attention=False, extra_bn=False):
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if norm_type == NormType.Weight:
        conv = nn.utils.weight_norm(conv)
    elif norm_type == NormType.Spectral:
        conv = nn.utils.spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(nn.ReLU(True))
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)


class CustomPixelShuffle_ICNR(nn.Module):
    def __init__(self, ni, nf=None, scale=2, blur=True, norm_type=NormType.Spectral, extra_bn=False):
        super().__init__()
        nf = nf or ni
        self.conv = custom_conv_layer(ni, nf * (scale ** 2), ks=1, use_activ=False, norm_type=norm_type, extra_bn=extra_bn)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        self.do_blur = blur
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.do_blur else x


class UnetBlockWide(nn.Module):
    def __init__(self, up_in_c, x_in_c, n_out, hook, blur=False, self_attention=False, norm_type=NormType.Spectral):
        super().__init__()
        self.hook = hook
        self.shuf = CustomPixelShuffle_ICNR(up_in_c, n_out, blur=blur, norm_type=norm_type, extra_bn=True)
        self.bn = batchnorm_2d(x_in_c)
        self.conv = custom_conv_layer(n_out + x_in_c, n_out, norm_type=norm_type, self_attention=self_attention, extra_bn=True)
        self.relu = nn.ReLU()

    def forward(self, up_in):
        s = self.hook.feature
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv(cat_x)


# =============================================================================
# ATTENTION
# =============================================================================

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        q = k = tgt + query_pos if query_pos is not None else tgt
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.multihead_attn(
            query=(tgt + query_pos) if query_pos is not None else tgt,
            key=(memory + pos) if pos is not None else memory,
            value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=False
        )[0]
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * 3.141592653589793

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


# =============================================================================
# COLOR DECODER
# =============================================================================

class MultiScaleColorDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, num_queries=100, nheads=8, dim_feedforward=2048, dec_layers=9, pre_norm=False, color_embed_dim=256, enforce_input_project=True, num_scales=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = dec_layers
        self.num_feature_levels = num_scales
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(num_scales, hidden_dim)
        self.input_proj = nn.ModuleList([nn.Conv2d(in_ch, hidden_dim, kernel_size=1) if in_ch != hidden_dim or enforce_input_project else nn.Sequential() for in_ch in in_channels])
        for proj in self.input_proj:
            if isinstance(proj, nn.Conv2d):
                nn.init.kaiming_uniform_(proj.weight, a=1)
                if proj.bias is not None:
                    nn.init.constant_(proj.bias, 0)

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(dec_layers):
            self.transformer_self_attention_layers.append(SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm))
            self.transformer_cross_attention_layers.append(CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm))
            self.transformer_ffn_layers.append(FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm))

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

    def forward(self, x, img_features):
        src, pos = [], []
        for i, feature in enumerate(x):
            pos.append(self.pe_layer(feature).flatten(2).permute(2, 0, 1))
            proj_feat = self.input_proj[i](feature).flatten(2)
            level_emb = self.level_embed.weight[i][None, :, None]
            src.append((proj_feat + level_emb).permute(2, 0, 1))

        bs = src[0].shape[1]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            output = self.transformer_cross_attention_layers[i](output, src[level_index], pos=pos[level_index], query_pos=query_embed)
            output = self.transformer_self_attention_layers[i](output, query_pos=query_embed)
            output = self.transformer_ffn_layers[i](output)

        decoder_output = self.decoder_norm(output).transpose(0, 1)
        color_embed = self.color_embed(decoder_output)
        return torch.einsum("bqc,bchw->bqhw", color_embed, img_features)


# =============================================================================
# MAIN MODEL
# =============================================================================

class _ImageEncoder(nn.Module):
    def __init__(self, encoder_name, hook_names):
        super().__init__()
        config = ENCODER_CONFIGS[encoder_name]
        self.arch = ConvNeXt(depths=config["depths"], dims=config["dims"])
        self.hooks = [Hook(self.arch._modules[name]) for name in hook_names]

    def forward(self, x):
        return self.arch(x)


class _UNetDecoder(nn.Module):
    def __init__(self, hooks, encoder_dims, nf=512, blur=True, last_norm="Spectral", num_queries=100, num_scales=3, dec_layers=9):
        super().__init__()
        self.hooks = hooks
        self.last_norm = getattr(NormType, last_norm)
        in_c = encoder_dims[-1]
        out_c = nf
        skip_dims = encoder_dims[-2::-1]
        self.layers = nn.ModuleList()
        for layer_index, (hook, feature_c) in enumerate(zip(hooks[-2::-1], skip_dims)):
            if layer_index == len(skip_dims) - 1:
                out_c = out_c // 2
            self.layers.append(UnetBlockWide(up_in_c=in_c, x_in_c=feature_c, n_out=out_c, hook=hook, blur=blur, self_attention=False, norm_type=NormType.Spectral))
            in_c = out_c

        embed_dim = nf // 2
        self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, blur=blur, norm_type=self.last_norm, scale=4)
        self.color_decoder = MultiScaleColorDecoder(in_channels=[nf, nf, nf // 2], num_queries=num_queries, num_scales=num_scales, dec_layers=dec_layers)

    def forward(self):
        encode_feat = self.hooks[-1].feature
        out0 = self.layers[0](encode_feat)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out1)
        out3 = self.last_shuf(out2)
        return self.color_decoder([out0, out1, out2], out3)


class DDColor(nn.Module):
    def __init__(self, encoder_name="convnext-l", input_size=(256, 256), nf=512, num_output_channels=2, last_norm="Spectral", num_queries=100, num_scales=3, dec_layers=9):
        super().__init__()
        if encoder_name not in ENCODER_CONFIGS:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        encoder_dims = ENCODER_CONFIGS[encoder_name]["dims"]
        self.encoder = _ImageEncoder(encoder_name, hook_names=["norm0", "norm1", "norm2", "norm3"])
        self.decoder = _UNetDecoder(hooks=self.encoder.hooks, encoder_dims=encoder_dims, nf=nf, last_norm=last_norm, num_queries=num_queries, num_scales=num_scales, dec_layers=dec_layers)
        self.refine_net = nn.Sequential(custom_conv_layer(num_queries + 3, num_output_channels, ks=1, use_activ=False, norm_type=NormType.Spectral))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, img):
        return (img - self.mean) / self.std

    def forward(self, x):
        x_norm = self.normalize(x) if x.shape[1] == 3 else x
        self.encoder(x_norm)
        out_feat = self.decoder()
        coarse_input = torch.cat([out_feat, x_norm], dim=1)
        return self.refine_net(coarse_input)


# =============================================================================
# INFERENCE HELPERS + PIPELINES
# =============================================================================

# Model cache (keyed by absolute path + device)
_MODEL_CACHE: Dict[str, "DDColor"] = {}
_MODEL_META_CACHE: Dict[str, Dict[str, Any]] = {}


def _safe_torch_load(path: str) -> Any:
    """torch.load wrapper that stays compatible across torch versions."""
    try:
        # PyTorch 2.0+ supports weights_only
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from many common checkpoint formats."""
    if isinstance(checkpoint, dict):
        for k in ("model_state_dict", "state_dict", "model", "params"):
            v = checkpoint.get(k)
            if isinstance(v, dict):
                return _strip_module_prefix(v)
        # Might already be a raw state_dict
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return _strip_module_prefix(checkpoint)  # type: ignore[arg-type]
    if isinstance(checkpoint, dict):
        return _strip_module_prefix(checkpoint)  # type: ignore[arg-type]
    raise ValueError("Unsupported checkpoint format (expected dict).")


def _extract_model_config(checkpoint: Any) -> Dict[str, Any]:
    """Extract model_config dict if present (saved by the trainer)."""
    if isinstance(checkpoint, dict):
        cfg = checkpoint.get("model_config")
        if isinstance(cfg, dict):
            return dict(cfg)
    return {}


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove 'module.' prefix added by (D)DP."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_ddcolor_model(
    model_path: str,
    model_size: str = "large",
    device: str | torch.device | None = None,
) -> Tuple["DDColor", Dict[str, Any]]:
    """Load a DDColor model with caching and checkpoint auto-config.

    Args:
        model_path: Path to .pth/.pt weights.
        model_size: "large" or "tiny" (used only if checkpoint has no model_config).
        device: "cuda"/"cpu" or torch.device. If None, auto-select.

    Returns:
        (model, meta) where meta contains the resolved architecture parameters.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    abs_path = os.path.abspath(os.path.expanduser(model_path))
    cache_key = f"{abs_path}|{device.type}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], _MODEL_META_CACHE.get(cache_key, {})

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model file not found: {abs_path}")

    ckpt = _safe_torch_load(abs_path)
    model_cfg = _extract_model_config(ckpt)

    # Resolve architecture (checkpoint config wins)
    encoder_name = str(model_cfg.get("encoder_name") or ("convnext-t" if model_size == "tiny" else "convnext-l"))
    num_queries = int(model_cfg.get("num_queries", 100))
    num_scales = int(model_cfg.get("num_scales", 3))
    dec_layers = int(model_cfg.get("dec_layers", 9))
    nf = int(model_cfg.get("nf", 512))

    model = DDColor(
        encoder_name=encoder_name,
        input_size=(256, 256),  # not used for computation
        nf=nf,
        num_output_channels=2,
        last_norm="Spectral",
        num_queries=num_queries,
        num_scales=num_scales,
        dec_layers=dec_layers,
    )

    state_dict = _extract_state_dict(ckpt)

    # Strict when model_config exists; fallback to non-strict for legacy checkpoints.
    strict = bool(model_cfg)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    meta = {
        "model_path": abs_path,
        "device": device.type,
        "encoder_name": encoder_name,
        "model_size": "tiny" if encoder_name.endswith("-t") else "large",
        "num_queries": num_queries,
        "num_scales": num_scales,
        "dec_layers": dec_layers,
        "nf": nf,
    }

    _MODEL_CACHE[cache_key] = model
    _MODEL_META_CACHE[cache_key] = meta
    return model, meta


def prepare_input_bgr(
    img_bgr_u8: np.ndarray,
    input_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Prepare a BGR uint8 image for the DDColor model.

    Returns:
        (tensor_input, orig_l) where:
          - tensor_input: (1, 3, input_size, input_size) float32 RGB grayscale in [0,1]
          - orig_l: (H, W, 1) float32 L channel in [0,100] at original resolution
    """
    if img_bgr_u8.dtype != np.uint8:
        raise ValueError("prepare_input_bgr expects uint8 BGR input")

    img_float = img_bgr_u8.astype(np.float32) / 255.0

    # Original L (float LAB mode -> L in [0,100])
    orig_l = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)[:, :, :1].astype(np.float32)

    # Resize to model input size
    img_resized = cv2.resize(img_float, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1].astype(np.float32)

    # Grayscale RGB in [0,1] (set ab=0)
    zeros = np.zeros_like(img_l, dtype=np.float32)
    img_gray_lab = np.concatenate([img_l, zeros, zeros], axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

    tensor_input = torch.from_numpy(img_gray_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    return tensor_input, orig_l


def lab_to_bgr_uint8(orig_l: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Convert (L, ab) in float LAB to BGR uint8."""
    orig_l = np.clip(orig_l, 0.0, 100.0).astype(np.float32)
    ab = np.clip(ab, -128.0, 127.0).astype(np.float32)
    lab = np.concatenate([orig_l, ab], axis=-1)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return (bgr * 255.0).round().clip(0, 255).astype(np.uint8)


def bgr_to_ab(img_bgr_u8: np.ndarray) -> np.ndarray:
    """Extract ab channels (float LAB mode) from a BGR uint8 image."""
    img_f = (img_bgr_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    lab = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab).astype(np.float32)
    return lab[:, :, 1:3]


def match_ab_distribution(
    ab: np.ndarray,
    reference_ab: np.ndarray,
    *,
    strength: float = 1.0,
    method: str = "meanstd",
    max_samples: int = 200_000,
    eps: float = 1e-6,
) -> np.ndarray:
    """Match global ab statistics of `ab` to `reference_ab` (no retraining)."""
    if strength <= 0:
        return ab.astype(np.float32)

    strength = float(np.clip(strength, 0.0, 1.0))
    method = str(method).lower().strip()

    ab = ab.astype(np.float32)
    reference_ab = reference_ab.astype(np.float32)

    flat_ab = ab.reshape(-1, 2)
    flat_ref = reference_ab.reshape(-1, 2)

    def _subsample(x: np.ndarray) -> np.ndarray:
        if x.shape[0] <= max_samples:
            return x
        step = max(1, x.shape[0] // max_samples)
        return x[::step][:max_samples]

    a_s = _subsample(flat_ab)
    r_s = _subsample(flat_ref)

    if method in {"meanstd", "mean_std", "ms"}:
        mu_a = a_s.mean(axis=0)
        mu_r = r_s.mean(axis=0)

        std_a = np.maximum(a_s.std(axis=0), eps)
        std_r = np.maximum(r_s.std(axis=0), eps)

        scale = std_r / std_a
        matched = (ab - mu_a.reshape(1, 1, 2)) * scale.reshape(1, 1, 2) + mu_r.reshape(1, 1, 2)

    elif method in {"cov", "covariance"}:

        def _cov(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            mu = x.mean(axis=0)
            xc = x - mu
            cov = (xc.T @ xc) / max(1, (xc.shape[0] - 1))
            cov = cov + np.eye(2, dtype=np.float32) * eps
            return mu, cov.astype(np.float32)

        def _sqrtm(cov: np.ndarray) -> np.ndarray:
            w, v = np.linalg.eigh(cov)
            w = np.maximum(w, eps)
            return (v @ np.diag(np.sqrt(w)) @ v.T).astype(np.float32)

        def _inv_sqrtm(cov: np.ndarray) -> np.ndarray:
            w, v = np.linalg.eigh(cov)
            w = np.maximum(w, eps)
            return (v @ np.diag(1.0 / np.sqrt(w)) @ v.T).astype(np.float32)

        mu_a, cov_a = _cov(a_s)
        mu_r, cov_r = _cov(r_s)

        transform = _sqrtm(cov_r) @ _inv_sqrtm(cov_a)
        matched_flat = (flat_ab - mu_a) @ transform.T + mu_r
        matched = matched_flat.reshape(ab.shape)

    else:
        raise ValueError("reference method must be 'meanstd' or 'cov'")

    out = (1.0 - strength) * ab + strength * matched
    return np.clip(out, -128.0, 127.0).astype(np.float32)


class ColorizationPipeline:
    """Standalone inference pipeline for DDColor (BGR numpy interface)."""

    def __init__(
        self,
        model_path: str,
        *,
        model_size: str = "large",
        input_size: int = 512,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_size = int(input_size)

        self.model, self.meta = load_ddcolor_model(
            model_path=model_path,
            model_size=model_size,
            device=self.device,
        )

    def _predict_ab(self, img_bgr_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img_bgr_u8.shape[:2]
        tensor_input, orig_l = prepare_input_bgr(img_bgr_u8, self.input_size, self.device)

        with torch.inference_mode():
            out_ab = self.model(tensor_input)

        out_ab = F.interpolate(out_ab, size=(h, w), mode="bilinear", align_corners=False)[0]
        ab = out_ab.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        ab = np.clip(ab, -128.0, 127.0)
        return orig_l, ab

    def colorize_bgr(
        self,
        img_bgr_u8: np.ndarray,
        *,
        reference_bgr_u8: Optional[np.ndarray] = None,
        reference_strength: float = 1.0,
        reference_method: str = "meanstd",
    ) -> np.ndarray:
        orig_l, ab = self._predict_ab(img_bgr_u8)

        if reference_bgr_u8 is not None and reference_strength > 0:
            ref_ab = bgr_to_ab(reference_bgr_u8)
            ab = match_ab_distribution(
                ab,
                ref_ab,
                strength=reference_strength,
                method=reference_method,
            )

        return lab_to_bgr_uint8(orig_l, ab)


class VideoColorizationPipeline:
    """Temporal wrapper around ColorizationPipeline (optical-flow warp + blend)."""

    def __init__(
        self,
        pipeline: ColorizationPipeline,
        *,
        temporal_strength: float = 0.8,
        use_edge_weighting: bool = True,
        flow_params: Optional[Dict[str, Any]] = None,
        reference_bgr_u8: Optional[np.ndarray] = None,
        reference_strength: float = 1.0,
        reference_method: str = "meanstd",
        auto_reference: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.temporal_strength = float(np.clip(temporal_strength, 0.0, 1.0))
        self.use_edge_weighting = bool(use_edge_weighting)

        self.flow_params: Dict[str, Any] = {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0,
        }
        if flow_params:
            self.flow_params.update(flow_params)

        self.reference_strength = float(np.clip(reference_strength, 0.0, 1.0))
        self.reference_method = str(reference_method).lower().strip()
        self.auto_reference = bool(auto_reference)

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_ab: Optional[np.ndarray] = None
        self._prev_shape: Optional[Tuple[int, int]] = None

        self._reference_ab: Optional[np.ndarray] = None
        if reference_bgr_u8 is not None and self.reference_strength > 0:
            self._reference_ab = bgr_to_ab(reference_bgr_u8)

    def reset(self, *, reset_reference: bool = False) -> None:
        self._prev_gray = None
        self._prev_ab = None
        self._prev_shape = None
        if reset_reference:
            self._reference_ab = None

    def _compute_flow(self, prev_gray_u8: np.ndarray, curr_gray_u8: np.ndarray) -> np.ndarray:
        p = self.flow_params
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray_u8,
            curr_gray_u8,
            None,
            pyr_scale=float(p.get("pyr_scale", 0.5)),
            levels=int(p.get("levels", 3)),
            winsize=int(p.get("winsize", 15)),
            iterations=int(p.get("iterations", 3)),
            poly_n=int(p.get("poly_n", 5)),
            poly_sigma=float(p.get("poly_sigma", 1.2)),
            flags=int(p.get("flags", 0)),
        )
        return flow.astype(np.float32)

    @staticmethod
    def _warp_ab(prev_ab: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = flow.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = (grid_x - flow[:, :, 0]).astype(np.float32)
        map_y = (grid_y - flow[:, :, 1]).astype(np.float32)
        warped = cv2.remap(
            prev_ab.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        return warped.astype(np.float32)

    @staticmethod
    def _edge_strength(gray_u8: np.ndarray) -> np.ndarray:
        gray_f = gray_u8.astype(np.float32) / 255.0
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        p90 = float(np.percentile(mag, 90))
        if p90 < 1e-6:
            return np.zeros_like(gray_f, dtype=np.float32)

        edge = np.clip(mag / p90, 0.0, 1.0)
        edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=1.0)
        return edge.astype(np.float32)

    def set_reference_image(self, reference_bgr_u8: Optional[np.ndarray]) -> None:
        if reference_bgr_u8 is None:
            self._reference_ab = None
            return
        self._reference_ab = bgr_to_ab(reference_bgr_u8)

    def colorize_frame(
        self,
        frame_bgr_u8: np.ndarray,
        *,
        reference_bgr_u8: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        orig_l, ab = self.pipeline._predict_ab(frame_bgr_u8)

        # Use L as flow grayscale input (0..255)
        curr_gray = (orig_l[:, :, 0] * (255.0 / 100.0)).round().clip(0, 255).astype(np.uint8)
        curr_shape = curr_gray.shape

        # Temporal blending
        if (
            self.temporal_strength > 0
            and self._prev_gray is not None
            and self._prev_ab is not None
            and self._prev_shape == curr_shape
        ):
            flow = self._compute_flow(self._prev_gray, curr_gray)
            warped_prev_ab = self._warp_ab(self._prev_ab, flow)

            if self.use_edge_weighting:
                edge = self._edge_strength(curr_gray)
                w_prev = self.temporal_strength * (1.0 - edge)
            else:
                w_prev = np.full(curr_shape, self.temporal_strength, dtype=np.float32)

            w_prev = np.clip(w_prev, 0.0, 1.0).astype(np.float32)
            ab = (1.0 - w_prev[:, :, None]) * ab + w_prev[:, :, None] * warped_prev_ab
            ab = np.clip(ab, -128.0, 127.0).astype(np.float32)

        # Reference palette handling
        if reference_bgr_u8 is not None and self.reference_strength > 0:
            self._reference_ab = bgr_to_ab(reference_bgr_u8)

        if self._reference_ab is None and self.auto_reference and self.reference_strength > 0:
            # Capture first frame palette
            self._reference_ab = ab.copy()
            apply_ref = False
        else:
            apply_ref = True

        if self._reference_ab is not None and self.reference_strength > 0 and apply_ref:
            ab = match_ab_distribution(
                ab,
                self._reference_ab,
                strength=self.reference_strength,
                method=self.reference_method,
            )

        out_bgr = lab_to_bgr_uint8(orig_l, ab)

        # Update temporal state with final ab
        self._prev_gray = curr_gray
        self._prev_ab = ab
        self._prev_shape = curr_shape

        return out_bgr


def comfy_tensor_to_bgr_u8(img: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI IMAGE tensor [H,W,C] RGB float -> BGR uint8."""
    img_np = img.detach().cpu().numpy()
    img_np = np.clip(img_np, 0.0, 1.0)
    rgb_u8 = (img_np * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)


def bgr_u8_to_comfy_tensor(img_bgr_u8: np.ndarray) -> torch.Tensor:
    """Convert BGR uint8 -> ComfyUI IMAGE tensor [H,W,C] RGB float."""
    rgb_u8 = cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2RGB)
    rgb_f = rgb_u8.astype(np.float32) / 255.0
    return torch.from_numpy(rgb_f)
