# Basically taken from Karpathy's llm.c repo
# https://github.com/karpathy/llm.c/blob/master/train_gpt2.py
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import math


# -----------------------
# RoPE helpers
# -----------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., d)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    q, k: (B, nh, T, hs)
    cos,sin: (1, 1, T, hs) broadcastable to q/k
    """
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))
        ))

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash_attention = config.flash_attention
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if self.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads for grouped-query attention (GQA)."""
    if n_rep == 1:
        return hidden_states
    B, num_kv_heads, T, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(B, num_kv_heads, n_rep, T, head_dim)
    return hidden_states.reshape(B, num_kv_heads * n_rep, T, head_dim)


# ---------------------------------------------------------------------------
# Llama building blocks
# ---------------------------------------------------------------------------

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)        # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)              # (T, head_dim)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)  # (1,1,T,head_dim)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        return (
            self.cos_cached[:, :, :seq_len].to(dtype=dtype, device=device),
            self.sin_cached[:, :, :seq_len].to(dtype=dtype, device=device),
        )


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.flash_attention = config.flash_attention

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        if not self.flash_attention:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings))
                     .view(1, 1, config.max_position_embeddings, config.max_position_embeddings),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(T, dtype=q.dtype, device=q.device)
        q, k = apply_rope(q, k, cos, sin)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        if self.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        return self.o_proj(y)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Config & top-level model
# ---------------------------------------------------------------------------

@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32   # == num_attention_heads for MHA; < for GQA
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    flash_attention: bool = True    # set by LlamaModel.__init__


class _LlamaInnerModel(nn.Module):
    """Matches HuggingFace LlamaModel (no lm_head)."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class LlamaModel(nn.Module):
    """Matches HuggingFace LlamaForCausalLM structure."""

    def __init__(self, config: LlamaConfig, device, flash_attention: bool = True):
        super().__init__()
        config.flash_attention = flash_attention
        self.config = config

        self.model = _LlamaInnerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # No weight tying (standard Llama keeps embed_tokens and lm_head independent)

        self.init_rng = torch.Generator(device="cpu")
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx: torch.Tensor, labels=None, return_logits: bool = True):
        b, t = idx.size()
        assert t <= self.config.max_position_embeddings, (
            f"Cannot forward sequence of length {t}, "
            f"max_position_embeddings is {self.config.max_position_embeddings}"
        )

        hidden_states = self.model(idx)

        if labels is not None:
            logits = self.lm_head(hidden_states)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(hidden_states[:, [-1], :])
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_position_embeddings else idx[:, -self.config.max_position_embeddings:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx




# -----------------------------------------------------------------------------
# GPT building blocks

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = NewGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config, device, flash_attention=True):
        super().__init__()
        config.flash_attention = flash_attention
        self.config = config
        self.device = device
       
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([AttentionBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.RMSNorm(config.n_embd, eps=1e-6),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator(device="cpu") #self.device)
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, labels=None, return_logits=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if labels is not None:
            # if we are given some desired labels also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
