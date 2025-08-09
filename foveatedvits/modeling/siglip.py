# Copyright 2024 Big Vision Authors.
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

"""A refactored and simplified ViT adoptation, taken from deepmind big_vision."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import foveatedvits.utils.jax_sharding as sharding

def rope_angles_2d(u, v, dim, base=10000.0):
    """
    Build interleaved 2D RoPE angles from (u, v) of shape [n, L] or [1, L].
    Returns cos, sin shaped [n, L, dim], where dim is even.
    The first half of pairs uses u, the second half uses v (x/y interleave).
    """
    assert dim % 2 == 0, "Head dim must be even for RoPE."
    half = dim // 2
    pair = half  # number of 2D pairs across x and y together
    freqs = jnp.arange(half // 2) / (half // 2 - 1 + 1e-9)
    inv_freq = 1.0 / (base ** freqs)                                  # [half//2]

    # Build angles for u (x) and v (y)
    def _angles(coord):
        # coord: [n, L]
        ang = jnp.einsum("nl,f->nlf", coord, inv_freq)                 # [n, L, half//2]
        sin = jnp.sin(ang); cos = jnp.cos(ang)
        # interleave into [n, L, half] as (cos, sin) pairs per 2 dims
        return jnp.stack([cos, sin], axis=-1).reshape(coord.shape[0], coord.shape[1], -1)  # [n,L,half]

    cos_x_sin_x = _angles(u)  # [n,L,half]
    cos_y_sin_y = _angles(v)  # [n,L,half]
    # Interleave x,y across the final dim: (x0,y0,x1,y1,...) each carrying (cos,sin) pairs implicitly
    cos = jnp.concatenate([cos_x_sin_x, cos_y_sin_y], axis=-1)  # [n,L,dim]
    sin = jnp.concatenate([cos_x_sin_x[..., ::2]*0 + cos_x_sin_x[..., 1:2]*0 + 0,  # just to match shape below
                           cos_y_sin_y], axis=-1)  # placeholder; we'll compute sin via a helper below
    # We'll actually not use this sin; rotation uses paired dims directly (see apply below).
    return cos, None  # sin isn't needed with the pairwise formula below


def rope_apply(q_or_k, u, v, base=10000.0):
    """
    Apply 2D RoPE in-place to a projected tensor shaped [n, L, num_heads, head_dim].
    We rotate every 2 dims as a complex pair; x- and y-based phases are interleaved.
    """
    n, L, H, D = q_or_k.shape
    assert D % 2 == 0
    # Build phases per head dim
    # For stability/throughput, compute inv_freq once and broadcast
    half = D // 2
    freqs = jnp.arange(half // 2) / (half // 2 - 1 + 1e-9)
    inv_freq = 1.0 / (base ** freqs)  # [half//2]

    # Angles for x and y: [n,L,half//2]
    ang_x = jnp.einsum("nl,f->nlf", u, inv_freq)
    ang_y = jnp.einsum("nl,f->nlf", v, inv_freq)

    # Expand to [n,L,1,half//2] then to pairs
    ang_x = ang_x[:, :, None, :]
    ang_y = ang_y[:, :, None, :]

    # Reshape q/k to pairs along head_dim
    q = q_or_k.reshape(n, L, H, D // 2, 2)  # [..., pair, 2]
    # Split the pair axis into two halves: first half gets x-phase, second half gets y-phase
    pair_half = q.shape[-2] // 2
    qx, qy = jnp.split(q, [pair_half], axis=-2)  # each [..., pair_half, 2]

    def rotate(qpair, ang):
        # qpair: [n,L,H,P,2], ang: [n,L,1,P]
        c = jnp.cos(ang)
        s = jnp.sin(ang)
        x = qpair[..., 0]
        y = qpair[..., 1]
        xr = x * c - y * s
        yr = x * s + y * c
        return jnp.stack([xr, yr], axis=-1)

    qx = rotate(qx, ang_x)
    qy = rotate(qy, ang_y)
    q_rot = jnp.concatenate([qx, qy], axis=-2).reshape(n, L, H, D)
    return q_rot

def rope_cache(u, v, head_dim, base=10000.0):
    """Precompute 2D RoPE cos/sin for x- and y-axes.
    u,v: [n,L]; returns cos_x, sin_x, cos_y, sin_y with shape [n,L,P],
    where P = head_dim//2 per axis (i.e., half the pairs go to x, half to y).
    """
    assert head_dim % 2 == 0
    P = (head_dim // 2) // 2  # number of pairs per axis
    freqs = jnp.arange(P) / (P - 1 + 1e-9)
    inv = 1.0 / (base ** freqs)                     # [P]
    ang_x = jnp.einsum("nl,f->nlf", u, inv)         # [n,L,P]
    ang_y = jnp.einsum("nl,f->nlf", v, inv)         # [n,L,P]
    return jnp.cos(ang_x), jnp.sin(ang_x), jnp.cos(ang_y), jnp.sin(ang_y)


def rope_apply_cached(q_or_k, cos_x, sin_x, cos_y, sin_y):
    """Apply cached RoPE phases to q_or_k shaped [n,L,H,D]."""
    n, L, H, D = q_or_k.shape
    q = q_or_k.reshape(n, L, H, D // 2, 2)           # [..., pairs, 2]
    pairs_total = q.shape[-2]
    pairs_per_axis = pairs_total // 2                # half for x, half for y
    qx, qy = jnp.split(q, [pairs_per_axis], axis=-2) # each [..., P, 2]

    def rotate(qp, c, s):
        # qp: [n,L,H,P,2]; c/s: [n,L,P] -> [n,L,1,P] for broadcast
        c = c[:, :, None, :]
        s = s[:, :, None, :]
        x = qp[..., 0]; y = qp[..., 1]
        xr = x * c - y * s
        yr = x * s + y * c
        return jnp.stack([xr, yr], axis=-1)

    qx = rotate(qx, cos_x, sin_x)
    qy = rotate(qy, cos_y, sin_y)
    return jnp.concatenate([qx, qy], axis=-2).reshape(n, L, H, D)


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Applies Transformer MlpBlock module."""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)

class RotaryMultiHeadDotProductAttention(nn.Module):
    num_heads: int
    dtype: str = "float32"
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    deterministic: bool = True

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, rope_uv=None, rope_cache=None,):
        d_model = inputs_q.shape[-1]
        head_dim = d_model // self.num_heads
        assert (d_model % self.num_heads) == 0, "d_model must be divisible by num_heads."

        proj = lambda: nn.DenseGeneral(  # noqa: E731
            features=(self.num_heads, head_dim),
            axis=-1,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=nn.initializers.zeros,
        )
        q = proj()(inputs_q)   # [n,L,H,Dh]
        k = proj()(inputs_kv)  # [n,L,H,Dh]
        v = proj()(inputs_kv)  # [n,L,H,Dh]

        if rope_cache is None:
            # Build a default (u,v) only for simple square single-tier; otherwise require rope_cache
            if rope_uv is None:
                n, L = q.shape[0], q.shape[1]
                h = w = int(jnp.sqrt(L))
                assert h * w == L, "Provide rope_cache for non-square / multi-tier sequences."
                ys, xs = jnp.mgrid[:h, :w]
                xs = (xs.reshape(-1) + 0.5) / w
                ys = (ys.reshape(-1) + 0.5) / h
                u = jnp.tile(xs[None, :], [n, 1])
                v = jnp.tile(ys[None, :], [n, 1])
            else:
                u, v = rope_uv
            cos_x, sin_x, cos_y, sin_y = rope_cache(u, v, head_dim)  # fallback compute
        else:
            cos_x, sin_x, cos_y, sin_y = rope_cache

        q = rope_apply_cached(q, cos_x, sin_x, cos_y, sin_y)
        k = rope_apply_cached(k, cos_x, sin_x, cos_y, sin_y)

        scale = 1.0 / jnp.sqrt(head_dim).astype(q.dtype)
        attn_logits = jnp.einsum("nqhd,nkhd->nhqk", q, k) * scale
        attn = nn.softmax(attn_logits, axis=-1)
        out = jnp.einsum("nhqk,nkhd->nqhd", attn, v)
        out = nn.DenseGeneral(
            features=d_model, axis=(-2, -1), dtype=self.dtype, kernel_init=self.kernel_init
        )(out)
        return out



class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    posemb: str | None = None

    @nn.compact
    def __call__(self, x, deterministic=True, rope_uv=None, rope_cache=None):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        if self.posemb == "rope2d_scale":
            y = out["sa"] = RotaryMultiHeadDotProductAttention(
                num_heads=self.num_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                dtype=self.dtype_mm,
            )(y, y, rope_uv=rope_uv, rope_cache=rope_cache)
        else:
            y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
            )(y, y)

        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    posemb: str | None = None

    @nn.compact
    def __call__(self, x, deterministic=True, rope_uv=None, rope_cache=None):  # noqa: FBT002
        out = {}

        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                posemb=self.posemb,
            )(x, deterministic, rope_uv=rope_uv, rope_cache=rope_cache)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            # Input Encoder
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic, rope_uv=rope_uv, rope_cache=rope_cache)
            out["pre_ln"] = x  # Alias for last block, but without the number in it.

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


class _Module(nn.Module):
    """ViT model."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    # posemb: str = "learn"  # Can also be "sincos2d" or "rope2d_scale"
    # posemb: str = "sincos2d"  
    posemb: str = "rope2d_scale"
    unify_tiers: bool = False
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"  # Can also be "map" or "tok"
    head_zeroinit: bool = True
    scan: bool = False
    # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image, *, train=False, rope_uv=None):
        out = {}

        # Kevin edit: do patch extraction and posemb in float32,
        # because I feel like it's a bit safer.
        image = jnp.asarray(image, jnp.float32)

        # Patch extraction
        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        if self.posemb == "rope2d_scale":
            # RoPE replaces additive pos embeddings
            out["with_posemb"] = x
        elif self.posemb in ("sincos2d", "learn"):
            # Add posemb before adding extra token.
            x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)
        else:
            raise ValueError(f"Unknown posemb type: {self.posemb}")

        # merge tiers in the batch into one token list
        if getattr(self, "unify_tiers", False) and self.posemb == "rope2d_scale":
            # Merge batch into sequence: e.g., B=2, L=64 -> [1, 128, D]
            x = x.reshape(1, n * h * w, c)
            n = 1

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape  # n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # Kevin edit: now cast back to dtype_mm (potentially half precision)
        x = x.astype(self.dtype_mm)

        rope_cache_vals = None
        if self.posemb == "rope2d_scale":
            # Expect rope_uv=(u,v) with shapes [n,L]; after unify_tiers, n=1, L=128
            assert rope_uv is not None, "rope_uv (u,v) required for rope2d_scale."
            u, v = rope_uv
            head_dim = self.width // self.num_heads
            cos_x, sin_x, cos_y, sin_y = rope_cache(u, v, head_dim)
            # Cast to compute dtype to avoid dtype churn later
            cos_x = cos_x.astype(self.dtype_mm); sin_x = sin_x.astype(self.dtype_mm)
            cos_y = cos_y.astype(self.dtype_mm); sin_y = sin_y.astype(self.dtype_mm)
            rope_cache_vals = (cos_x, sin_x, cos_y, sin_y)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train, rope_uv=rope_uv, rope_cache=rope_cache_vals)
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
