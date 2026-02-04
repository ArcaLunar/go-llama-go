import dataclasses
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

try:
    import triton
    import triton.language as tl

    _triton_available = True
except ImportError:
    _triton_available = False


@dataclasses.dataclass
class ModelConfig:
    head_dim: int

    hidden_size: int

    intermediate_size: int

    num_attention_heads: int

    num_hidden_layers: int

    num_key_value_heads: int

    rms_norm_eps: float

    rope_theta: float

    torch_dtype: str

    vocab_size: int


if _triton_available:

    @triton.jit
    def _rmsnorm_forward_kernel(x_ptr, w_ptr, o_ptr, eps, stride, hidden_size, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * stride + tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < hidden_size

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(x * x, axis=0) / hidden_size
        inv_rms = tl.rsqrt(mean + eps)

        y = x * inv_rms * w
        tl.store(o_ptr + offsets, y, mask=mask)


    @triton.jit
    def _silu_mul_down_kernel(
        gate_ptr,
        up_ptr,
        w_ptr,
        o_ptr,
        stride_gm,
        stride_gk,
        stride_um,
        stride_uk,
        stride_wk,
        stride_wn,
        stride_om,
        stride_on,
        M,
        K,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_start = k * BLOCK_K
            k_mask = k_start + offs_k < K

            gate_block = tl.load(
                gate_ptr + offs_m[:, None] * stride_gm + (k_start + offs_k)[None, :] * stride_gk,
                mask=(offs_m[:, None] < M) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            up_block = tl.load(
                up_ptr + offs_m[:, None] * stride_um + (k_start + offs_k)[None, :] * stride_uk,
                mask=(offs_m[:, None] < M) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            act = gate_block * (1.0 / (1.0 + tl.exp(-gate_block))) * up_block

            w_block = tl.load(
                w_ptr + (k_start + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                mask=k_mask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            ).to(tl.float32)

            acc += tl.dot(act, w_block)

        acc = acc.to(tl.float16)

        tl.store(
            o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def _rms_norm_triton(input, weight, eps):
    hidden_size = input.shape[-1]
    x_2d = input.view(-1, hidden_size)

    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    if not weight.is_contiguous():
        weight = weight.contiguous()

    output = torch.empty_like(x_2d)

    # Bound block size to avoid register pressure.
    block_size = int(min(4096, triton.next_power_of_2(hidden_size))) if _triton_available else hidden_size

    grid = (x_2d.shape[0],)

    _rmsnorm_forward_kernel[grid](
        x_2d,
        weight,
        output,
        eps,
        x_2d.stride(0),
        hidden_size,
        BLOCK_SIZE=block_size,
    )

    return output.view_as(input)


def _swi_glu_down_triton(gate, up, down_weight):
    # gate/up: [B,S,Inter], down_weight: [Hidden, Inter]
    M = gate.numel() // gate.shape[-1]
    K = gate.shape[-1]

    w = down_weight.t()
    N = w.shape[1]

    gate_2d = gate.reshape(M, K)
    up_2d = up.reshape(M, K)

    if not gate_2d.is_contiguous():
        gate_2d = gate_2d.contiguous()
    if not up_2d.is_contiguous():
        up_2d = up_2d.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()

    out = torch.empty((M, N), device=gate.device, dtype=gate.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _silu_mul_down_kernel[grid](
        gate_2d,
        up_2d,
        w,
        out,
        gate_2d.stride(0),
        gate_2d.stride(1),
        up_2d.stride(0),
        up_2d.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        M,
        K,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out.view_as(gate.new_empty((*gate.shape[:-1], N)))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))

        self.eps = eps

    def forward(self, input):
        if _triton_available and input.is_cuda:
            return _rms_norm_triton(input, self.weight, self.eps)

        return (
            input
            * torch.rsqrt(input.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        )


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.silu = nn.SiLU()

    def forward(self, input):
        gate = self.gate_proj(input)
        up = self.up_proj(input)

        if _triton_available and input.is_cuda:
            return _swi_glu_down_triton(gate, up, self.down_proj.weight)

        return self.down_proj(self.silu(gate) * up)


def apply_rotary_position_embedding(input, sin_table, cos_table):
    if _triton_available and input.is_cuda and os.environ.get("USE_TRITON_ROPE", "1") != "0":
        return _apply_rope_triton(input, sin_table, cos_table)

    sin_table = sin_table[None, :, None, :]
    cos_table = cos_table[None, :, None, :]

    input_0 = input[..., : input.shape[-1] // 2]
    input_1 = input[..., input.shape[-1] // 2 :]
    input_0_rotated = input_0 * cos_table - input_1 * sin_table
    input_1_rotated = input_0 * sin_table + input_1 * cos_table

    return torch.cat((input_0_rotated, input_1_rotated), dim=-1)


if _triton_available:

    @triton.jit
    def _flash_attn_fwd(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        sm_scale,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_km,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vm,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        n_heads,
        seqlen,
        head_dim,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        batch_id = pid_bh // n_heads
        head_id = pid_bh % n_heads

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)

        row_mask = offs_m < seqlen

        q_ptrs = (
            q_ptr
            + batch_id * stride_qz
            + head_id * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_k[None, :] * stride_qk
        )
        q_mask = (offs_m[:, None] < seqlen) & (offs_k[None, :] < head_dim)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        m_i = tl.where(row_mask, float("-inf"), 0.0)
        l_i = tl.where(row_mask, 0.0, 1.0)
        acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

        n_blocks = tl.cdiv(seqlen, BLOCK_N)
        for n in range(0, n_blocks):
            start_n = n * BLOCK_N
            k_ptrs = (
                k_ptr
                + batch_id * stride_kz
                + head_id * stride_kh
                + (start_n + offs_n)[None, :] * stride_km
                + offs_k[:, None] * stride_kk
            )
            v_ptrs = (
                v_ptr
                + batch_id * stride_vz
                + head_id * stride_vh
                + (start_n + offs_n)[:, None] * stride_vm
                + offs_k[None, :] * stride_vk
            )

            kv_mask = ((start_n + offs_n)[None, :] < seqlen) & (offs_k[:, None] < head_dim)

            k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=kv_mask.T, other=0.0).to(tl.float32)

            qk = tl.dot(q, k) * sm_scale

            valid_k = (start_n + offs_n)[None, :] < seqlen
            causal_mask = valid_k & ((start_n + offs_n)[None, :] <= offs_m[:, None])
            qk = tl.where(causal_mask & row_mask[:, None], qk, float("-inf"))

            m_ij = tl.max(qk, axis=1)
            m_ij_is_inf = m_ij == float("-inf")

            p = tl.where(m_ij_is_inf[:, None], 0.0, tl.exp(qk - m_ij[:, None]))
            l_ij = tl.sum(p, axis=1)
            acc_ij = tl.dot(p, v)

            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.where(m_ij_is_inf, 0.0, tl.exp(m_ij - m_new))

            l_i = l_i * alpha + l_ij * beta
            acc = acc * alpha[:, None] + acc_ij * beta[:, None]
            m_i = m_new

        acc = acc / l_i[:, None]

        o_ptrs = (
            o_ptr
            + batch_id * stride_oz
            + head_id * stride_oh
            + offs_m[:, None] * stride_om
            + offs_k[None, :] * stride_ok
        )
        store_mask = (offs_m[:, None] < seqlen) & (offs_k[None, :] < head_dim)
        tl.store(o_ptrs, acc, mask=store_mask)


    @triton.jit
    def _rope_kernel_flat(
        x_ptr,
        sin_ptr,
        cos_ptr,
        o_ptr,
        stride_xm,
        stride_xd,
        stride_sm,
        stride_sd,
        total_rows,
        seqlen,
        n_heads,
        head_dim,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)

        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = offs_m < total_rows

        offs_d = tl.arange(0, BLOCK_D)
        half_d = head_dim // 2
        d_mask = offs_d < half_d

        # Map row -> (b, s, h)
        rows = offs_m
        s_idx = (rows // n_heads) % seqlen

        first = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_d[None, :] * stride_xd,
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        second = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (offs_d + half_d)[None, :] * stride_xd,
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        sin = tl.load(
            sin_ptr + s_idx[:, None] * stride_sm + offs_d[None, :] * stride_sd,
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        cos = tl.load(
            cos_ptr + s_idx[:, None] * stride_sm + offs_d[None, :] * stride_sd,
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        out_first = first * cos - second * sin
        out_second = first * sin + second * cos

        tl.store(
            o_ptr + offs_m[:, None] * stride_xm + offs_d[None, :] * stride_xd,
            out_first,
            mask=row_mask[:, None] & d_mask[None, :],
        )
        tl.store(
            o_ptr + offs_m[:, None] * stride_xm + (offs_d + half_d)[None, :] * stride_xd,
            out_second,
            mask=row_mask[:, None] & d_mask[None, :],
        )


def _flash_attention_triton(query, key, value, scale):
    batch, n_heads, seq_len, head_dim = query.shape

    q = query.contiguous()
    k = key.contiguous()
    v = value.contiguous()

    o = torch.empty_like(q)

    BLOCK_M = 128 if seq_len <= 128 else 64
    BLOCK_N = 128 if seq_len <= 128 else 64
    block_d = int(triton.next_power_of_2(head_dim))

    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

    _flash_attn_fwd[grid](
        q,
        k,
        v,
        o,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        n_heads,
        seq_len,
        head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=block_d,
    )

    return o


def _apply_rope_triton(x, sin_table, cos_table):
    # x: [B, S, H, D]
    bsz, seqlen, n_heads, head_dim = x.shape

    assert head_dim % 2 == 0

    x_c = x.contiguous()
    sin_c = sin_table.contiguous()
    cos_c = cos_table.contiguous()

    flat = x_c.view(-1, head_dim)
    out = torch.empty_like(flat)

    BLOCK_M = 128
    BLOCK_D = min(128, head_dim)

    total_rows = flat.shape[0]

    grid = (triton.cdiv(total_rows, BLOCK_M),)

    _rope_kernel_flat[grid](
        flat,
        sin_c,
        cos_c,
        out,
        flat.stride(0),
        flat.stride(1),
        sin_c.stride(0),
        sin_c.stride(1),
        total_rows,
        seqlen,
        n_heads,
        head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D // 2,
    )

    return out.view_as(x_c)


def apply_scaled_dot_product_attention(query, key, value):
    _, num_heads_q, seq_len_q, emb_dim = query.shape
    _, num_heads_k, seq_len_k, _ = key.shape
    _, num_heads_v, _, _ = value.shape

    key = key.repeat_interleave(num_heads_q // num_heads_k, 1)
    value = value.repeat_interleave(num_heads_q // num_heads_v, 1)

    scale = 1 / math.sqrt(emb_dim)

    if _triton_available and query.is_cuda and os.environ.get("USE_TRITON_ATTENTION", "1") != "0":
        try:
            return _flash_attention_triton(query, key, value, scale)
        except RuntimeError:
            pass

    attn_mask = torch.tril(
        torch.full((seq_len_q, seq_len_k), True, device=query.device)
    )

    attn_output = torch.matmul(query, key.permute(0, 1, 3, 2)) * scale
    attn_output = torch.where(attn_mask, attn_output, float("-inf"))
    attn_output = torch.softmax(attn_output, dim=-1)
    attn_output = torch.matmul(attn_output, value)

    return attn_output


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )

        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, hidden_states, sin_table, cos_table):
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape).permute(0, 2, 1, 3)

        query_states = apply_rotary_position_embedding(
            query_states, sin_table, cos_table
        ).permute(0, 2, 1, 3)
        key_states = apply_rotary_position_embedding(
            key_states, sin_table, cos_table
        ).permute(0, 2, 1, 3)

        attn_output = apply_scaled_dot_product_attention(
            query_states, key_states, value_states
        )

        return self.o_proj(
            attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        )


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.self_attn = Attention(config)

        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, sin_table, cos_table):
        hidden_states += self.self_attn(
            self.input_layernorm(hidden_states), sin_table, cos_table
        )

        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states


def generate_sin_and_cos_tables(seq_len, emb_dim, base, dtype, device):
    theta = base ** (
        -2 * (torch.arange(emb_dim // 2, dtype=dtype, device=device) / emb_dim)
    )

    positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    sin_table = torch.sin(positions * theta)
    cos_table = torch.cos(positions * theta)

    return sin_table, cos_table


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_hidden_layers = config.num_hidden_layers

        self.rms_norm_eps = config.rms_norm_eps

        self.rope_theta = config.rope_theta

        self.torch_dtype = config.torch_dtype

        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(self.vocab_size, self.hidden_size)

        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(self.num_hidden_layers)
        )

        self.norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        seq_len = hidden_states.shape[1]

        sin_table, cos_table = generate_sin_and_cos_tables(
            seq_len,
            self.head_dim,
            base=self.rope_theta,
            dtype=getattr(torch, self.torch_dtype),
            device=input_ids.device,
        )

        for i in range(self.num_hidden_layers):
            hidden_states = self.layers[i](hidden_states, sin_table, cos_table)

        return self.norm(hidden_states)


class ModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def generate(self, input_ids, max_new_tokens=20):
        for _ in range(max_new_tokens):
            hidden_states = self.model(input_ids)

            logits = self.lm_head(hidden_states[:, -1, :])

            next = torch.argmax(logits, dim=-1).unsqueeze(-1)

            input_ids = torch.cat((input_ids, next), dim=-1)

        return input_ids

    @staticmethod
    def from_pretrained(model_path):
        model_path = Path(model_path)

        with open(model_path / "config.json") as f:
            config = json.load(f)

        if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

        config = ModelConfig(
            **{
                key: value
                for key, value in config.items()
                if key in ModelConfig.__annotations__
            }
        )

        model = ModelForCausalLM(config).to(getattr(torch, config.torch_dtype))

        state_dict = load_file(model_path / "model.safetensors")

        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        model.load_state_dict(state_dict)

        return model
