# SPDX-License-Identifier: LicenseRef-LMU-CompVis-NC-Research-1.0
# SPDX-FileCopyrightText: Copyright 2026 Ulrich Prestel, Stefan Baumann et al., CompVis @ LMU Munich

import math
from typing import Any, Annotated, ClassVar, Self, Sequence
from functools import reduce as functools_reduce, partial
from dataclasses import dataclass, fields

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask, _mask_mod_signature
from einops import rearrange, repeat, reduce as einops_reduce
from jaxtyping import Float, Int, Bool


D_HEAD = 64
FF_EXPAND = 3
KERNEL_SIZE = (1, 7, 7)
PLUECKER_DIM = 6
RGB_DIM = 3

# ---------------------------------------------------------------------------------------------------------------------
# Camera Stuff
# - We wrap our various camera-related tensors in dataclasses with a mixin to enable jaxtyping-style shape annotations,
#   slicing, and other useful stuff.
# - Extrinsics are parametrized in se(3) using axis-angle and translation parameters
# - Intrinsics are a simplified pinhole model with a single focal length parameter
# ---------------------------------------------------------------------------------------------------------------------

FOCAL_LENGTH_EPS = 1e-6


class TensorStruct:
    def __class_getitem__(cls, item: str) -> Any:
        return Annotated[cls, item]

    def __getitem__(self, idx) -> Self:
        if not isinstance(idx, tuple):
            idx = (idx,)
        if any(i is Ellipsis for i in idx):
            raise NotImplementedError()
        return type(self)(
            **{
                f.name: getattr(self, f.name)[idx + (slice(None),) * (getattr(self, f.name).ndim - len(idx))]
                for f in fields(self)
            }
        )

    @classmethod
    def cat(cls, structs: Sequence[Self], dim: int = 0) -> Self:
        return type(structs[0])(
            **{f.name: torch.cat([getattr(s, f.name) for s in structs], dim=dim) for f in fields(structs[0])}
        )


def _hat(w: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3 3"]:
    wx, wy, wz = w.unbind(dim=-1)
    O = torch.zeros_like(wx)
    return torch.stack(
        [
            torch.stack([O, -wz, wy], dim=-1),
            torch.stack([wz, O, -wx], dim=-1),
            torch.stack([-wy, wx, O], dim=-1),
        ],
        dim=-2,
    )


def _safe_div(num: Float[torch.Tensor, "..."], den: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    den = torch.where(den.abs() > 0, den, torch.ones_like(den))
    return torch.nan_to_num(num / den, nan=0.0, posinf=0.0, neginf=0.0)


def _exp_so3(omega: Float[torch.Tensor, "... 3"], eps: float = 1e-5) -> Float[torch.Tensor, "... 3 3"]:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    W = _hat(omega)
    theta2 = theta * theta
    s = torch.where(theta > eps, _safe_div(torch.sin(theta), theta), 1 - theta2 / 6 + theta2**2 / 120)
    c = torch.where(theta > eps, _safe_div(1 - torch.cos(theta), theta2), 0.5 - theta2 / 24 + theta2**2 / 720)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(omega.shape[:-1] + (3, 3))
    return I + s[..., None] * W + c[..., None] * (W @ W)


def _left_jacobian_so3(omega: Float[torch.Tensor, "... 3"], eps: float = 1e-5) -> Float[torch.Tensor, "... 3 3"]:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    W = _hat(omega)
    theta2 = theta * theta
    theta3 = theta2 * theta
    a = torch.where(theta > eps, _safe_div(1 - torch.cos(theta), theta2), 0.5 - theta2 / 24 + theta2**2 / 720)
    b = torch.where(theta > eps, _safe_div(theta - torch.sin(theta), theta3), 1 / 6 - theta2 / 120 + theta2**2 / 5040)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(omega.shape[:-1] + (3, 3))
    return I + a[..., None] * W + b[..., None] * (W @ W)


def _log_so3(R: Float[torch.Tensor, "... 3 3"], eps: float = 1e-5) -> Float[torch.Tensor, "... 3"]:
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    angle = torch.acos(((trace - 1) / 2).clamp(-1 + eps, 1 - eps))
    omega = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    factor = torch.where(angle.abs() < eps, 0.5 + angle**2 / 12, angle / (2 * torch.sin(angle)))
    return omega * factor[..., None]


@dataclass(frozen=True)
class Ray(TensorStruct):
    o: Float[torch.Tensor, "... 3"]
    d: Float[torch.Tensor, "... 3"]


@dataclass(frozen=True)
class Camera(TensorStruct):
    R: Float[torch.Tensor, "... 3 3"]
    t: Float[torch.Tensor, "... 3"]
    f: Float[torch.Tensor, "..."]

    NUM_EXTRINSICS_PARAMS: ClassVar[int] = 6
    NUM_INTRINSICS_PARAMS: ClassVar[int] = 1

    @staticmethod
    def from_parameters(
        ext_params: Float[torch.Tensor, "... 6"], int_params: Float[torch.Tensor, "... 1"], eps: float = 1e-5
    ) -> "Camera":
        omega, v = ext_params[..., :3], ext_params[..., 3:]
        R = _exp_so3(omega, eps=eps)
        t = (_left_jacobian_so3(omega, eps=eps) @ v[..., None]).squeeze(-1)
        f = int_params[..., 0].exp() + FOCAL_LENGTH_EPS
        return Camera(R=R, t=t, f=f)

    @staticmethod
    def interpolate(c1: "Camera", c2: "Camera", alpha: Float[torch.Tensor, "..."]) -> "Camera":
        # Interpolate R along SO(3) geodesic, lerp the rest
        R = c1.R @ _exp_so3(alpha[..., None].to(c1.R) * _log_so3(c1.R.transpose(-2, -1) @ c2.R))
        return Camera(R=R, t=c1.t.lerp(c2.t, alpha[..., None].to(c1.t)), f=c1.f.lerp(c2.f, alpha.to(c1.f)))

    @property
    def shape(self) -> torch.Size:
        return self.R.shape[:-2]

    def get_rays(self, h: int, w: int) -> "Ray":
        device, dtype, cam_shape = self.f.device, self.f.dtype, self.f.shape
        u, v = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing="xy")
        u = u.to(dtype)[*(None for _ in cam_shape)].expand(*cam_shape, -1, -1)
        v = v.to(dtype)[*(None for _ in cam_shape)].expand(*cam_shape, -1, -1)
        scale = 1 / min(h - 1, w - 1)
        u, v = (2 * u - (w - 1)) * scale, (2 * v - (h - 1)) * scale
        x = u / self.f[..., None, None]
        y = v / self.f[..., None, None]
        d_c = F.normalize(torch.stack([x, y, torch.ones_like(x)], dim=-1), dim=-1)
        d_w = rearrange(self.R @ rearrange(d_c, "... h w c -> ... c (h w)"), "... c (h w) -> ... h w c", h=h, w=w)
        return Ray(o=self.t[..., None, None, :].expand_as(d_w), d=F.normalize(d_w, dim=-1))


def rays_to_pluecker(rays: Ray) -> Float[torch.Tensor, "... 6"]:
    return torch.cat([rays.d, torch.cross(rays.o, rays.d, dim=-1)], dim=-1)


# ---------------------------------------------------------------------------------------------------------------------
# Various Modules & Utilities
# ---------------------------------------------------------------------------------------------------------------------


def _rms_norm(
    x: Float[torch.Tensor, "... d"], scale: Float[torch.Tensor, "d"], eps: float
) -> Float[torch.Tensor, "... d"]:
    dtype = torch.promote_types(torch.promote_types(x.dtype, scale.dtype), torch.float32)
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


class AdaRMSNorm(nn.Module):
    def __init__(self, features: int, cond_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(cond_features, features, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(
        self, x: Float[torch.Tensor, "... d"], cond: Float[torch.Tensor, "... d_cond"]
    ) -> Float[torch.Tensor, "... d"]:
        return _rms_norm(x, self.linear(cond) + 1, 1e-6)


class RMSNorm(nn.Module):
    def __init__(self, shape: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(shape))

    def forward(self, x: Float[torch.Tensor, "... d"]) -> Float[torch.Tensor, "... d"]:
        return _rms_norm(x, self.scale, 1e-6)


class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features * 2, bias=False)
        self.out_features = out_features

    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_out"]:
        x, gate = (x @ self.weight.mT).chunk(2, dim=-1)
        return x * F.silu(gate)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_cond: int) -> None:
        super().__init__()
        d_ff = d_model * FF_EXPAND
        self.norm = AdaRMSNorm(d_model, d_cond)
        self.up_proj = LinearSwiGLU(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        nn.init.zeros_(self.down_proj.weight)

    def forward(
        self, x: Float[torch.Tensor, "b ... d"], cond_norm: Float[torch.Tensor, "b ... d_cond"]
    ) -> Float[torch.Tensor, "b ... d"]:
        return x + self.down_proj(self.up_proj(self.norm(x, cond_norm)))


class MLPHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, zero_init_out: bool = False) -> None:
        super().__init__()
        self.norm = RMSNorm(in_features)
        self.out_proj = nn.Linear(in_features, out_features)
        if zero_init_out:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Float[torch.Tensor, "b ... c"]) -> Float[torch.Tensor, "b ... d"]:
        return self.out_proj(self.norm(x))


# ---------------------------------------------------------------------------------------------------------------------
# Attention & RoPE Utilities
# - Mostly mirrors HDiT (Crowson et al., ICML 2024), albeit 3D
# - Coordinate convention:
#   - Spatial: (y, x) in [-1, 1]^2
#   - Temporal: t in [0, 1, ..., T-1]
# ---------------------------------------------------------------------------------------------------------------------


def centers(start: float, stop: float, num: int, dtype=None, device=None) -> Float[torch.Tensor, "n"]:
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def bounding_box(h: int, w: int) -> tuple[float, float, float, float]:
    ar = w / h
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar > 1:
        y_min, y_max = -1 / ar, 1 / ar
    elif ar < 1:
        x_min, x_max = -ar, ar
    return y_min, y_max, x_min, x_max


def make_grid_3d(
    t_pos: Float[torch.Tensor, "t"], h_pos: Float[torch.Tensor, "h"], w_pos: Float[torch.Tensor, "w"]
) -> Float[torch.Tensor, "thw 3"]:
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1).unsqueeze(0)
    grid = torch.cat(
        [torch.ones_like(grid[:, :, :, :1]) * t_pos.view(-1, 1, 1, 1), grid.repeat(t_pos.size(0), 1, 1, 1)], dim=-1
    )
    return grid.reshape(-1, grid.shape[-1])


def make_axial_pos_3d(t: int, h: int, w: int, dtype=None, device=None) -> Float[torch.Tensor, "thw 3"]:
    y_min, y_max, x_min, x_max = bounding_box(h, w)
    t_pos = torch.arange(t, dtype=dtype, device=device)
    h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
    w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid_3d(t_pos, h_pos, w_pos)


def scale_for_cosine_sim(
    q: Float[torch.Tensor, "... d"], k: Float[torch.Tensor, "... d"], scale: Float[torch.Tensor, "..."], eps: float
) -> tuple[Float[torch.Tensor, "... d"], Float[torch.Tensor, "... d"]]:
    dtype = torch.promote_types(torch.promote_types(torch.promote_types(q.dtype, k.dtype), scale.dtype), torch.float32)
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


def apply_rotary_emb(
    x: Float[torch.Tensor, "... d"], theta: Float[torch.Tensor, "... d_rot"]
) -> Float[torch.Tensor, "... d"]:
    out_dtype = x.dtype
    dtype = torch.promote_types(torch.promote_types(x.dtype, theta.dtype), torch.float32)
    d = theta.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1.to(out_dtype), y2.to(out_dtype), x3), dim=-1)


class AxialRoPE3D(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        n_freqs = n_heads * dim // 8
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        spatial_freqs = torch.linspace(log_min, log_max, n_freqs + 1)[:-1].exp()
        spatial_freqs = torch.stack([spatial_freqs] * 2)
        temporal_freqs = 1.0 / (100.0 ** (torch.arange(0, n_freqs).float() / n_freqs))
        self.spatial_freqs = nn.Parameter(spatial_freqs.view(2, dim // 8, n_heads).mT.contiguous(), requires_grad=False)
        self.temporal_freqs = nn.Parameter(temporal_freqs.view(dim // 8, n_heads).T.contiguous(), requires_grad=False)

    def forward(self, pos: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... n_heads d_rot"]:
        theta_t = pos[..., None, 0:1] * self.temporal_freqs
        theta_h = pos[..., None, 1:2] * self.spatial_freqs[0]
        theta_w = pos[..., None, 2:3] * self.spatial_freqs[1]
        return torch.cat((theta_t, theta_h, theta_w), dim=-1)


# ---------------------------------------------------------------------------------------------------------------------
# Transformer Layers/Blocks
# The transformer is, for the most part, a modern standard LLama2-style transformer,
# with some further modifications primarily inspired by HDiT (Crowson et al., ICML 2024)
# Main differences to a "standard" transformer:
# - No biases for linear layers
# - LayerNorm -> RMSNorm
# - GELU -> SwiGLU
# - Additive PE -> RoPE, using the HDiT approach to implement 2D RoPE
# - d_k-scaled attention -> cossim-attention with learnable scales
# - Adaptive norms to introduce conditioning
# Notable somewhat specific implementation details for RayDer:
# - We use a hierarchical architecture, mostly following HDiT
# - We insert camera tokens in the middle block; also, middle block transformer layers are basically VGGT-style
# - We use custom attention masks for self-attention in the middle block, which are implemented via flex attention
# - Neighborhood attention in the down/up levels is implemented via flex attention, too, to keep dependencies minimal
# ---------------------------------------------------------------------------------------------------------------------


class NeighborhoodAttention(nn.Module):
    def __init__(self, d_model: int, d_cond: int) -> None:
        super().__init__()
        n_heads = d_model // D_HEAD
        self.n_heads = n_heads
        self.norm = AdaRMSNorm(d_model, d_cond)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.scale = nn.Parameter(torch.full([n_heads], 10.0))
        self.pos_emb = AxialRoPE3D(D_HEAD, n_heads)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        self.block_masks: dict[tuple, BlockMask] = {}

    @staticmethod
    def _make_mask(ct: int, ch: int, cw: int, kt: int, kh: int, kw: int) -> _mask_mod_signature:
        hw = ch * cw
        kh2, kw2, kt2 = kh // 2, kw // 2, kt // 2

        def split(
            idx: Int[torch.Tensor, ""],
        ) -> tuple[Int[torch.Tensor, ""], Int[torch.Tensor, ""], Int[torch.Tensor, ""]]:
            t = torch.div(idx, hw, rounding_mode="floor")
            rem = idx.remainder(hw)
            return t, torch.div(rem, cw, rounding_mode="floor"), rem.remainder(cw)

        def mask_mod(b, h, q_idx, kv_idx) -> Bool[torch.Tensor, ""]:
            qt, qx, qy = split(q_idx)
            kt_, kx, ky = split(kv_idx)
            return (
                ((qt.clamp(kt2, ct - 1 - kt2) - kt_).abs() <= kt2)
                & ((qx.clamp(kw2, cw - 1 - kw2) - kx).abs() <= kw2)
                & ((qy.clamp(kh2, ch - 1 - kh2) - ky).abs() <= kh2)
            )

        return mask_mod

    @torch.compiler.disable(recursive=False)
    def _lazy_mask(self, dims: list[int], device: torch.device) -> None:
        key = tuple(dims)
        if key not in self.block_masks:
            L = functools_reduce(lambda a, b: a * b, key)
            self.block_masks[key] = create_block_mask(
                self._make_mask(*key, *KERNEL_SIZE), B=1, H=1, Q_LEN=L, KV_LEN=L, device=device, _compile=True
            )

    def forward(
        self,
        x: Float[torch.Tensor, "b ... d"],
        pos: Float[torch.Tensor, "b ... 3"],
        cond_norm: Float[torch.Tensor, "b ... d_cond"],
    ) -> Float[torch.Tensor, "b ... d"]:
        B, *DIMS, _ = x.shape
        skip = x
        x = rearrange(self.norm(x, cond_norm), "b ... c -> b (...) c")
        q, k, v = rearrange(self.qkv_proj(x), "n l (t h e) -> t n h l e", t=3, e=D_HEAD)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
        theta = self.pos_emb(rearrange(pos, "b ... c -> b (...) c")).movedim(-2, -3)
        q = apply_rotary_emb(q, theta)
        k = apply_rotary_emb(k, theta)
        self._lazy_mask(DIMS, x.device)
        out = flex_attention(q, k, v, scale=1.0, block_mask=self.block_masks[tuple(DIMS)])
        return self.out_proj(rearrange(out, "n h l e -> n l (h e)")).view_as(skip) + skip


class NeighborhoodTransformerLayer(nn.Module):
    def __init__(self, d_model: int, d_cond: int) -> None:
        super().__init__()
        self.self_attn = NeighborhoodAttention(d_model, d_cond)
        self.ff = FeedForward(d_model, d_cond)

    def forward(
        self,
        x: Float[torch.Tensor, "b ... d"],
        pos: Float[torch.Tensor, "b ... 3"],
        cond_norm: Float[torch.Tensor, "b ... d_cond"],
    ) -> Float[torch.Tensor, "b ... d"]:
        x = self.self_attn(x, pos, cond_norm)
        x = self.ff(x, cond_norm)
        return x


class RegisterAttention(nn.Module):
    def __init__(self, d_model: int, d_cond: int, use_rope: bool = True) -> None:
        super().__init__()
        n_heads = d_model // D_HEAD
        self.n_heads = n_heads
        self.norm = AdaRMSNorm(d_model, d_cond)
        self.register_norm = AdaRMSNorm(d_model, d_cond)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.scale = nn.Parameter(torch.full([n_heads], 10.0))
        self.pos_emb = AxialRoPE3D(D_HEAD, n_heads) if use_rope else None
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        x: Float[torch.Tensor, "b ... d"],
        registers: Float[torch.Tensor, "b n_r d"],
        cond_norm: Float[torch.Tensor, "b ... d_cond"],
        registers_cond_norm: Float[torch.Tensor, "b n_r d_cond"],
        pos: Float[torch.Tensor, "b ... c"] | None = None,
        registers_pos: Float[torch.Tensor, "b n_r c"] | None = None,
        block_mask: BlockMask | None = None,
    ) -> tuple[Float[torch.Tensor, "b ... d"], Float[torch.Tensor, "b n_r d"]]:
        skip, skip_r = x, registers
        x = self.norm(x, cond_norm)
        registers = self.register_norm(registers, registers_cond_norm)
        B, *DIMS, C = x.shape
        N_R = registers.shape[1]
        # qkv: [3, b, n_heads, l, d_head]
        qkv = rearrange(
            self.qkv_proj(rearrange(x, "b ... c -> b (...) c")),
            "n l (t nh e) -> t n nh l e",
            t=3,
            e=D_HEAD,
        )
        qkv_r = rearrange(self.qkv_proj(registers), "n l (t nh e) -> t n nh l e", t=3, e=D_HEAD)
        q, k, v = qkv
        q_r, k_r, v_r = qkv_r
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
        q_r, k_r = scale_for_cosine_sim(q_r, k_r, self.scale[:, None, None], 1e-6)
        if self.pos_emb is not None:
            theta = self.pos_emb(rearrange(pos, "b ... c -> b (...) c")).movedim(-2, -3)
            q = apply_rotary_emb(q, theta)
            k = apply_rotary_emb(k, theta)
            theta_r = self.pos_emb(registers_pos).movedim(-2, -3)
            q_r = apply_rotary_emb(q_r, theta_r)
            k_r = apply_rotary_emb(k_r, theta_r)
        q, k, v = torch.cat([q, q_r], dim=-2), torch.cat([k, k_r], dim=-2), torch.cat([v, v_r], dim=-2)
        if block_mask is not None:
            out = flex_attention(q, k, v, scale=1.0, block_mask=block_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        out = self.out_proj(rearrange(out, "n nh l e -> n l (nh e)"))
        return out[:, :-N_R].view(B, *DIMS, C) + skip, out[:, -N_R:] + skip_r


class RegisterFeedForward(nn.Module):
    def __init__(self, d_model: int, d_cond: int) -> None:
        super().__init__()
        d_ff = d_model * FF_EXPAND
        self.norm = AdaRMSNorm(d_model, d_cond)
        self.register_norm = AdaRMSNorm(d_model, d_cond)
        self.up_proj = LinearSwiGLU(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        nn.init.zeros_(self.down_proj.weight)

    def forward(
        self,
        x: Float[torch.Tensor, "b ... d"],
        registers: Float[torch.Tensor, "b n_r d"],
        cond_norm: Float[torch.Tensor, "b ... d_cond"],
        registers_cond_norm: Float[torch.Tensor, "b n_r d_cond"],
    ) -> tuple[Float[torch.Tensor, "b ... d"], Float[torch.Tensor, "b n_r d"]]:
        skip, skip_r = x, registers
        x = self.norm(x, cond_norm)
        registers = self.register_norm(registers, registers_cond_norm)
        B, *DIMS, C = x.shape
        N_R = registers.shape[1]
        x = self.down_proj(self.up_proj(torch.cat([rearrange(x, "b ... c -> b (...) c"), registers], dim=1)))
        return x[:, :-N_R].view(B, *DIMS, C) + skip, x[:, -N_R:] + skip_r


class GlobalLocalTransformerLayer(nn.Module):
    def __init__(self, d_model: int, d_cond: int) -> None:
        super().__init__()
        self.global_attn = RegisterAttention(d_model, d_cond, use_rope=False)
        self.local_attn = RegisterAttention(d_model, d_cond, use_rope=True)
        self.ff = RegisterFeedForward(d_model, d_cond)

    def forward(
        self,
        x: Float[torch.Tensor, "b t h w d"],
        registers: Float[torch.Tensor, "b t d"],
        pos: Float[torch.Tensor, "b t h w 3"],
        registers_pos: Float[torch.Tensor, "b t 3"],
        cond_norm: Float[torch.Tensor, "b t h w d_cond"],
        registers_cond_norm: Float[torch.Tensor, "b t d_cond"],
        global_block_mask: BlockMask | None = None,
    ) -> tuple[Float[torch.Tensor, "b t h w d"], Float[torch.Tensor, "b t d"]]:
        B, T, H, W, _ = x.shape
        x, registers = self.global_attn(
            x=x,
            registers=registers,
            block_mask=global_block_mask,
            cond_norm=cond_norm,
            registers_cond_norm=registers_cond_norm,
        )
        x_local, regs_local = self.local_attn(
            x=rearrange(x, "b t h w d -> (b t) h w d"),
            registers=rearrange(registers, "b t d -> (b t) 1 d"),
            pos=rearrange(pos, "b t h w c -> (b t) h w c"),
            registers_pos=rearrange(registers_pos, "b t c -> (b t) 1 c"),
            cond_norm=rearrange(cond_norm, "b t h w c -> (b t) h w c"),
            registers_cond_norm=rearrange(registers_cond_norm, "b t d -> (b t) 1 d"),
        )
        x = rearrange(x_local, "(b t) h w d -> b t h w d", b=B)
        registers = rearrange(regs_local, "(b t) 1 d -> b t d", b=B)
        x, registers = self.ff(x=x, registers=registers, cond_norm=cond_norm, registers_cond_norm=registers_cond_norm)
        return x, registers


# ---------------------------------------------------------------------------------------------------------------------
# Token Merges/Splits: just modules that do the interactions between layers of the hierarchical model & in/out patching
# ---------------------------------------------------------------------------------------------------------------------


class TokenMerge3D(nn.Module):
    def __init__(self, in_features: int, out_features: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.t, self.h, self.w = patch_size
        self.proj = nn.Linear(in_features * self.t * self.h * self.w, out_features, bias=False)

    def forward(
        self, x: Float[torch.Tensor, "b ... d"], pos: Float[torch.Tensor, "b ... 3"]
    ) -> tuple[Float[torch.Tensor, "b ... d_out"], Float[torch.Tensor, "b ... 3"]]:
        x = self.proj(
            rearrange(
                x,
                "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw e)",
                nt=self.t,
                nh=self.h,
                nw=self.w,
            )
        )
        pos = rearrange(pos, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw) e", nt=self.t, nh=self.h, nw=self.w)
        return x, torch.mean(pos, dim=-2)


class TokenSplit3D(nn.Module):
    def __init__(self, in_features: int, out_features: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.t, self.h, self.w = patch_size
        self.proj = nn.Linear(in_features, out_features * self.t * self.h * self.w, bias=False)
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(
        self, x: Float[torch.Tensor, "b ... d"], skip: Float[torch.Tensor, "b ... d_out"]
    ) -> Float[torch.Tensor, "b ... d_out"]:
        x = rearrange(
            self.proj(x),
            "... t h w (nt nh nw e) -> ... (t nt) (h nh) (w nw) e",
            nt=self.t,
            nh=self.h,
            nw=self.w,
        )
        return torch.lerp(skip, x, self.fac.to(x.dtype))


class TokenSplitLast3D(nn.Module):
    def __init__(self, in_features: int, out_features: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.t, self.h, self.w = patch_size
        self.proj = nn.Linear(in_features, out_features * self.t * self.h * self.w, bias=False)
        self.norm = RMSNorm(in_features)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: Float[torch.Tensor, "b ... d"]) -> Float[torch.Tensor, "b ... d_out"]:
        return rearrange(
            self.proj(self.norm(x)),
            "... t h w (nt nh nw e) -> ... (t nt) (h nh) (w nw) e",
            nt=self.t,
            nh=self.h,
            nw=self.w,
        )


# ---------------------------------------------------------------------------------------------------------------------
# Main Transformer Backbone: basically just HDiT-style transformer, with the changes relevant to the layers we use
# ---------------------------------------------------------------------------------------------------------------------


class Backbone(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        d_cond: int,
        down_up_configs: list[tuple[int, int, tuple[int, int, int]]],
        main_patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        configs = down_up_configs

        self.merges = nn.ModuleList()
        self.down_levels = nn.ModuleList()
        prev_in = RGB_DIM + PLUECKER_DIM
        for w, d, ps in configs:
            self.merges.append(TokenMerge3D(prev_in, w, ps))
            self.down_levels.append(nn.ModuleList([NeighborhoodTransformerLayer(w, d_cond) for _ in range(d)]))
            prev_in = w

        prev_out = RGB_DIM
        self.splits = nn.ModuleList()
        self.up_levels = nn.ModuleList()
        for i, (w, d, ps) in enumerate(configs):
            self.up_levels.append(nn.ModuleList([NeighborhoodTransformerLayer(w, d_cond) for _ in range(d)]))
            self.splits.append(TokenSplitLast3D(w, prev_out, ps) if i == 0 else TokenSplit3D(w, prev_out, ps))
            prev_out = w

        self.mid_merge = TokenMerge3D(prev_in, width, main_patch_size)
        self.mid_level = nn.ModuleList([GlobalLocalTransformerLayer(width, d_cond) for _ in range(depth)])
        self.mid_split = TokenSplit3D(width, prev_out, main_patch_size)

    def forward(
        self,
        x: Float[torch.Tensor, "b t h w c"],
        pos: Float[torch.Tensor, "b t h w 3"],
        registers: Float[torch.Tensor, "b t d"],
        registers_pos: Float[torch.Tensor, "b t 3"],
        cond_norm: Float[torch.Tensor, "b t h w d_cond"],
        registers_cond_norm: Float[torch.Tensor, "b t d_cond"],
        global_block_mask: BlockMask | None = None,
    ) -> tuple[Float[torch.Tensor, "b t h w c_out"], Float[torch.Tensor, "b t d"]]:
        skips, poses = [], []
        for merge, level in zip(self.merges, self.down_levels):
            skips.append(x)
            x, pos = merge(x, pos)
            poses.append(pos)
            for layer in level:
                x = layer(x, pos=pos, cond_norm=cond_norm)

        skip_mid = x
        x, pos = self.mid_merge(x, pos)
        for layer in self.mid_level:
            x, registers = layer(
                x,
                registers,
                pos=pos,
                registers_pos=registers_pos,
                global_block_mask=global_block_mask,
                cond_norm=cond_norm,
                registers_cond_norm=registers_cond_norm,
            )
        x = self.mid_split(x, skip=skip_mid)

        for split, level, skip_down, pos_down in reversed(list(zip(self.splits, self.up_levels, skips, poses))):
            for layer in level:
                x = layer(x, pos=pos_down, cond_norm=cond_norm)
            x = split(x, skip=skip_down) if isinstance(split, TokenSplit3D) else split(x)

        return x, registers


# ---------------------------------------------------------------------------------------------------------------------
# Main Model
# - Tokens can have different roles (view, camera, register), and be used in different contexts (camera estimation, NVS)
#   so we condition the block input norms on that, similar to how we condition on time in diffusion models,
#   inspired by Nair et al. (ICCV 2025).
# - We also use "fancy" masking, but it's mostly quite simple conceptually:
#   - During training, we use AR masking over input views, followed by target views attending to the first few input views
#   - During inference, AR masking over input views is retained (otherwise, there would be a train/test gap),
#     but all target views can attend to all input views
#   - Token layout is: [input views, target views, camera tokens (one for each view, in the same order)].
#     This makes the logic seem a bit more complex at first vs. grouping the camera tokens with the view tokens,
#     but it gives us a better memory layout for attention computation, and ultimately reduces practical attention cost
# ---------------------------------------------------------------------------------------------------------------------


class RayDer(nn.Module):
    def __init__(
        self,
        width: int = 1024,
        depth: int = 24,
        d_dynamic_state: int = 256,
        dynamic_state_dropout: float = 0.5,
        down_up_configs: list[tuple[int, int, tuple[int, int, int]]] | None = None,
        main_patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> None:
        super().__init__()
        if down_up_configs is None:
            down_up_configs = [(128, 2, (1, 4, 4)), (256, 2, (1, 2, 2))]
        d_cond = width * 2
        self.width = width
        self.d_dynamic_state = d_dynamic_state
        self.dynamic_state_dropout = dynamic_state_dropout
        self.total_spatial_downsample = math.prod(ps[1] for _, _, ps in down_up_configs) * main_patch_size[1]

        self.backbone = Backbone(
            width=width, depth=depth, d_cond=d_cond, down_up_configs=down_up_configs, main_patch_size=main_patch_size
        )

        std = 1e-4
        self.camera_tokens = nn.Embedding(1, width)
        nn.init.trunc_normal_(self.camera_tokens.weight, std=std, a=-2 * std, b=2 * std)
        self.nvs_tokens = nn.Embedding(1, width)
        nn.init.trunc_normal_(self.nvs_tokens.weight, std=std, a=-2 * std, b=2 * std)

        self.camera_pose_head = MLPHead(width, Camera.NUM_EXTRINSICS_PARAMS)
        self.intrinsics_head = MLPHead(width, Camera.NUM_INTRINSICS_PARAMS, zero_init_out=True)
        self.dynamic_state_head = MLPHead(width, d_dynamic_state)
        self.dynamic_state_in_proj = nn.Linear(d_dynamic_state, width, bias=False)

        self.view_type_embedding = nn.Embedding(3, d_cond)  # 0=camera_est, 1=nvs_in, 2=nvs_out
        self.token_type_embedding = nn.Embedding(3, d_cond)  # 0=view_token, 1=camera_token, 2=register_token

    def _train_block_mask(self, n_in: int, n_t: int, n_tokens_per_view: int, device: torch.device) -> BlockMask:
        n_in = int(n_in)
        n_t = int(n_t)
        n_tokens_per_view = int(n_tokens_per_view)

        def mask_mod(batch, head, q_idx, kv_idx) -> Bool[torch.Tensor, ""]:
            q_v = torch.where(
                q_idx < (n_in + n_t) * n_tokens_per_view,
                q_idx // n_tokens_per_view,
                q_idx - (n_in + n_t) * n_tokens_per_view,
            )
            kv_v = torch.where(
                kv_idx < (n_in + n_t) * n_tokens_per_view,
                kv_idx // n_tokens_per_view,
                kv_idx - (n_in + n_t) * n_tokens_per_view,
            )
            return torch.where(
                (q_v < n_in) & (kv_v < n_in),
                q_v >= kv_v,
                (q_v == kv_v) | (q_v >= kv_v + n_in),
            )

        L = (n_in + n_t) * (n_tokens_per_view + 1)
        return create_block_mask(mask_mod, B=1, H=1, Q_LEN=L, KV_LEN=L, device=device)

    @torch.compiler.disable(recursive=False)
    def _inference_block_mask(self, n_in: int, n_t: int, n_tokens_per_view: int, device: torch.device) -> BlockMask:
        def mask_mod(batch, head, q_idx, kv_idx) -> Bool[torch.Tensor, ""]:
            q_v = torch.where(
                q_idx < (n_in + n_t) * n_tokens_per_view,
                q_idx // n_tokens_per_view,
                q_idx - (n_in + n_t) * n_tokens_per_view,
            )
            kv_v = torch.where(
                kv_idx < (n_in + n_t) * n_tokens_per_view,
                kv_idx // n_tokens_per_view,
                kv_idx - (n_in + n_t) * n_tokens_per_view,
            )
            return torch.where(
                (q_v < n_in) & (kv_v < n_in),
                q_v >= kv_v,
                (q_v == kv_v) | (kv_v < n_in),
            )

        L = (n_in + n_t) * (n_tokens_per_view + 1)
        return create_block_mask(mask_mod, B=1, H=1, Q_LEN=L, KV_LEN=L, device=device)

    def _estimate_cameras(
        self, x: Float[torch.Tensor, "b t h w c"]
    ) -> tuple[Camera["b t"], Float[torch.Tensor, "b t d_state"]]:
        B, T, H, W, C = x.shape
        pos = repeat(make_axial_pos_3d(t=T, h=H, w=W, device=x.device), "(t h w) c -> b t h w c", b=B, t=T, h=H, w=W)
        pos = pos.clone()
        pos[..., 0] = torch.argsort(torch.rand(B, T, device=x.device), dim=-1)[:, :, None, None].to(pos)
        camera_tokens = self.camera_tokens(torch.zeros(B, T, dtype=torch.long, device=x.device))
        registers_pos = einops_reduce(pos, "b t h w c -> b t c", "mean")

        ekw: dict[str, Any] = {"dtype": torch.long, "device": x.device}
        cond_norm = self.view_type_embedding(torch.zeros((B, T, 1, 1), **ekw)) + self.token_type_embedding(
            torch.zeros((B, T, 1, 1), **ekw)
        )
        registers_cond_norm = self.view_type_embedding(torch.zeros((B, T), **ekw)) + self.token_type_embedding(
            torch.ones((B, T), **ekw)
        )
        _, camera_tokens = self.backbone(
            x=torch.cat([x, x.new_zeros(B, T, H, W, PLUECKER_DIM)], dim=-1),
            pos=pos,
            registers=camera_tokens,
            registers_pos=registers_pos,
            cond_norm=cond_norm,
            registers_cond_norm=registers_cond_norm,
        )
        cameras = Camera.from_parameters(
            self.camera_pose_head(camera_tokens).to(torch.float64),
            self.intrinsics_head(camera_tokens).to(torch.float64),
        )
        return cameras, self.dynamic_state_head(camera_tokens)

    def _reconstruct(
        self,
        x_in: Float[torch.Tensor, "b n_in h w c"],
        camera_in: Camera["b n_in"],
        camera_target: Camera["b n_t"],
        state_target: Float[torch.Tensor, "b n_t d_state"],
        block_mask: BlockMask,
        drop_state: bool = False,
    ) -> Float[torch.Tensor, "b n_t h w c"]:
        (B, N_in, H, W, C), dtype, device = x_in.shape, x_in.dtype, x_in.device
        _, N_t = camera_target.shape
        N_all = N_in + N_t
        cameras = Camera.cat([camera_in, camera_target], dim=1)
        pluecker = rays_to_pluecker(cameras.get_rays(h=H, w=W)).to(dtype)

        pos = repeat(
            make_axial_pos_3d(t=N_all, h=H, w=W, device=device), "(t h w) c -> b t h w c", b=B, t=N_all, h=H, w=W
        )
        pos = pos.clone()
        pos[..., 0] = torch.argsort(torch.rand(B, N_all, device=device), dim=-1)[:, :, None, None].to(pos)
        camera_tokens = self.nvs_tokens(torch.zeros(B, N_all, dtype=torch.long, device=device))
        registers_pos = einops_reduce(pos, "b t h w c -> b t c", "mean")

        state = self.dynamic_state_in_proj(
            torch.cat([state_target.new_zeros(B, N_in, self.d_dynamic_state), state_target], dim=1)
        )
        if drop_state:
            state = torch.zeros_like(state)
        elif self.training and self.dynamic_state_dropout > 0:
            mask = (torch.rand((B, N_all, 1), device=device) >= self.dynamic_state_dropout).to(state)
            state = mask * state
        camera_tokens = camera_tokens + state

        ekw: dict[str, Any] = {"dtype": torch.long, "device": device}
        cond_norm = self.view_type_embedding(
            torch.cat([torch.ones((B, N_in, 1, 1), **ekw), torch.full((B, N_t, 1, 1), 2, **ekw)], dim=1)
        ) + self.token_type_embedding(torch.zeros((B, N_all, 1, 1), **ekw))
        registers_cond_norm = self.view_type_embedding(
            torch.cat([torch.ones((B, N_in), **ekw), torch.full((B, N_t), 2, **ekw)], dim=1)
        ) + self.token_type_embedding(torch.full((B, N_all), 2, **ekw))
        image_tokens, _ = self.backbone(
            x=torch.cat([torch.cat([x_in, x_in.new_zeros(B, N_t, H, W, C)], dim=1), pluecker], dim=-1),
            pos=pos,
            registers=camera_tokens,
            registers_pos=registers_pos,
            global_block_mask=block_mask,
            cond_norm=cond_norm,
            registers_cond_norm=registers_cond_norm,
        )
        return image_tokens[:, -N_t:]

    @torch.no_grad()
    @torch.compile(dynamic=False, fullgraph=False)
    def predict_cameras(self, x: Float[torch.Tensor, "b t h w c"]) -> Camera["b t"]:
        return self._estimate_cameras(x=x)[0]

    @torch.no_grad()
    @torch.compile(dynamic=False, fullgraph=False)
    def predict_views(
        self,
        x_in: Float[torch.Tensor, "b n_in h w c"],
        cam_in: Camera["b n_in"],
        cam_target: Camera["b n_t"],
        state_target: Float[torch.Tensor, "b n_t d_state"] | None = None,
    ) -> Float[torch.Tensor, "b n_t h w c"]:
        B, N_in, H, W, C = x_in.shape
        N_t = cam_target.shape[1]
        n_tokens = (H // self.total_spatial_downsample) * (W // self.total_spatial_downsample)
        block_mask = self._inference_block_mask(N_in, N_t, n_tokens, x_in.device)
        if state_target is None:
            state_target = x_in.new_zeros(B, N_t, self.d_dynamic_state)
        return self._reconstruct(
            x_in=x_in, camera_in=cam_in, camera_target=cam_target, state_target=state_target, block_mask=block_mask
        )


# ---------------------------------------------------------------------------------------------------------------------
# Standard Variants
# ---------------------------------------------------------------------------------------------------------------------

RayDer_XS = partial(RayDer, width=384, depth=12)
RayDer_S = partial(RayDer, width=512, depth=18)
RayDer_B = partial(RayDer, width=768, depth=24)
RayDer_L = partial(RayDer, width=1024, depth=24)
