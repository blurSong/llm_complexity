import os
import re
import csv
import json
import math
from scipy.stats import binom
from tabulate import tabulate
from huggingface_hub import hf_hub_download, list_repo_files


def axwy_to_bytes(axwy: str):
    match = re.match(r"a(\d+)w(\d+)", axwy)
    assert match
    ab, wb = match.groups()
    return float(ab) / 8, float(wb) / 8


def how_many_experts(e: int, t: int, k: int, l: int = 0):
    # e: shared experts
    # t: tokens
    # k: number of experts per token
    # FIXME: Need refine. l: topk group limit (deepseek). 0 means no limit.
    geese = t * k
    # e = min(e, l) if l > 0 else e
    prob_used = 1 - (1 - 1.0 / e) ** geese
    n_probs = binom.pmf(range(e + 1), e, prob_used)
    most_likely_n = n_probs.argmax()

    return most_likely_n


def get_model_config(path_or_hf_repo: str, cache_dir: str = None):

    if os.path.exists(path_or_hf_repo):
        local_path = os.path.join(path_or_hf_repo, "config.json")
    else:
        if cache_dir:
            local_path = os.path.join(cache_dir, path_or_hf_repo.split("/")[-1])
        else:
            local_path = None
        local_path = hf_hub_download(repo_id=path_or_hf_repo, filename="config.json", local_dir=local_path)

    try:
        with open(local_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise

    return config


def download_model_cache(hf_repo: str, cache_dir: str = None):

    weight_exts = ["bin", "safetensors", "pt", "pth", "ckpt", "npz"]

    local_path = os.path.join(cache_dir, hf_repo.split("/")[-1]) if cache_dir else None
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    files = list_repo_files(hf_repo)
    for f in files:
        if not any(f.endswith(ext) for ext in weight_exts):
            hf_hub_download(repo_id=hf_repo, filename=f, local_dir=local_path)

    return local_path


class llama:
    def __init__(self, config: dict, name: str = None):
        self.config = config
        self.name = name
        self.num_layers = self.config["num_hidden_layers"]
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = (
            self.config["num_key_value_heads"] if "num_key_value_heads" in self.config else self.num_heads
        )
        self.head_dim = self.config["head_dim"] if "head_dim" in self.config else self.hidden_size // self.num_heads
        self.intermediate_size = self.config["intermediate_size"]
        self.vocab_size = self.config["vocab_size"]
        self.tie_word_embeddings = self.config["tie_word_embeddings"] if "tie_word_embeddings" in self.config else False

    def calc_inference_math_ops(
        self, tokens: int, past_tokens: int = 0, batch: int = 1, return_break_down: bool = False
    ):
        lm_head_macs = self.hidden_size * self.vocab_size * tokens
        q_proj_macs = self.hidden_size * self.num_heads * self.head_dim * tokens
        kv_proj_macs = 2 * self.hidden_size * self.num_kv_heads * self.head_dim * tokens
        out_proj_macs = self.hidden_size * self.num_heads * self.head_dim * tokens

        attention_tokens = tokens + past_tokens
        attention_qk_macs = self.num_heads * tokens * self.head_dim * attention_tokens
        attention_softmax_macs = self.num_heads * tokens * attention_tokens
        attention_qkv_macs = self.num_heads * tokens * attention_tokens * self.head_dim

        mlp_ffn_macs = self.intermediate_size * self.hidden_size * tokens
        mlp_matdot_macs = self.intermediate_size * tokens

        transformer_block_macs = (
            q_proj_macs
            + kv_proj_macs
            + out_proj_macs
            + attention_qk_macs
            + attention_softmax_macs
            + attention_qkv_macs
            + mlp_ffn_macs * 3
            + mlp_matdot_macs
        )

        scale = 2 * batch
        lyr = self.num_layers

        complexity = (transformer_block_macs * lyr + lm_head_macs) * scale
        if not return_break_down:
            return complexity

        lyr = self.num_layers
        break_down = {}
        break_down.update({"q_proj": q_proj_macs * lyr * scale})
        break_down.update({"kv_proj": kv_proj_macs * lyr * scale})
        break_down.update({"out_proj": out_proj_macs * lyr * scale})
        break_down.update(
            {"self_attention": (attention_qk_macs + attention_softmax_macs + attention_qkv_macs) * lyr * scale}
        )
        break_down.update({"mlp": (mlp_ffn_macs * 3 + mlp_matdot_macs) * lyr * scale})
        break_down.update({"lm_head": lm_head_macs * scale})
        return break_down

    def calc_inference_dram_bytes(
        self, tokens: int, past_tokens: int = 0, batch: int = 1, axwy: str = "a16w4", return_break_down: bool = False
    ):
        embedding_params = self.vocab_size * self.hidden_size
        lm_head_params = self.hidden_size * self.vocab_size

        q_proj_params = self.hidden_size * self.num_heads * self.head_dim
        kv_proj_params = 2 * self.hidden_size * self.num_kv_heads * self.head_dim
        out_proj_params = self.hidden_size * self.num_heads * self.head_dim
        mlp_ffn_params = self.intermediate_size * self.hidden_size

        q_activations = tokens * self.num_heads * self.head_dim
        kv_activations = 2 * tokens * self.num_kv_heads * self.head_dim
        kv_cache = 2 * past_tokens * self.num_kv_heads * self.head_dim
        hidden_states = tokens * self.hidden_size
        hidden_states_mlp = tokens * self.intermediate_size
        output_states = 1 * self.vocab_size

        ab, wb = axwy_to_bytes(axwy)
        scale_a, scale_w = ab * batch, wb
        lyr = self.num_layers

        # Calc DRAM
        transformer_block_params = q_proj_params + kv_proj_params + out_proj_params + mlp_ffn_params * 3
        transformer_params = transformer_block_params * lyr
        head_and_tail_params = embedding_params + (lm_head_params if not self.tie_word_embeddings else 0)
        total_kv_cache = (kv_activations + kv_cache) * lyr

        capacity = (transformer_params + head_and_tail_params) * scale_w + (
            total_kv_cache + hidden_states + hidden_states_mlp
        ) * scale_a

        # Calc IO
        embedding_io_bytes = embedding_params * scale_w + hidden_states * scale_a
        lm_head_io_bytes = lm_head_params * scale_w + (hidden_states + output_states) * scale_a
        q_proj_io_bytes = q_proj_params * scale_w + (hidden_states + q_activations) * scale_a
        kv_proj_io_bytes = kv_proj_params * scale_w + (hidden_states + kv_activations) * scale_a
        self_attention_io_bytes = (q_activations + kv_activations + kv_cache + hidden_states) * scale_a
        out_proj_io_bytes = out_proj_params * scale_w + (hidden_states + q_activations) * scale_a
        mlp_ffn_io_bytes = mlp_ffn_params * scale_w + (hidden_states + hidden_states_mlp) * scale_a
        mlp_gate_io_bytes = hidden_states_mlp * 3 * scale_a

        total_attention_io_bytes = (
            q_proj_io_bytes + kv_proj_io_bytes + out_proj_io_bytes + self_attention_io_bytes
        ) * lyr
        total_mlp_io_bytes = (mlp_ffn_io_bytes * 3 + mlp_gate_io_bytes) * lyr
        head_tail_io_bytes = embedding_io_bytes + lm_head_io_bytes

        io = head_tail_io_bytes + total_attention_io_bytes + total_mlp_io_bytes

        if not return_break_down:
            return capacity, io

        lyr = self.num_layers
        io_break_down = {}
        io_break_down.update({"embedding": embedding_io_bytes})
        io_break_down.update({"q_proj": q_proj_io_bytes * lyr})
        io_break_down.update({"kv_proj": kv_proj_io_bytes * lyr})
        io_break_down.update({"self_attention": self_attention_io_bytes * lyr})
        io_break_down.update({"out_proj": out_proj_io_bytes * lyr})
        io_break_down.update({"mlp": total_mlp_io_bytes})
        io_break_down.update({"lm_head": lm_head_io_bytes})
        return capacity, io_break_down

    def calc_inference_comm_bytes(self):
        pass


class llama4:
    def __init__(self, config: dict, name: str = None):
        # llama4 config contains text_config and vision_config.
        if "text_config" in config:
            self.config = config["text_config"]
        else:
            self.config = config
        self.name = name
        self.num_layers = self.config["num_hidden_layers"]
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = (
            self.config["num_key_value_heads"] if "num_key_value_heads" in self.config else self.num_heads
        )
        self.head_dim = self.config["head_dim"] if "head_dim" in self.config else self.hidden_size // self.num_heads
        self.num_experts_per_tok = self.config["num_experts_per_tok"]
        self.num_experts = self.config["num_local_experts"]
        self.intermediate_size = self.config["intermediate_size"]
        self.intermediate_size_mlp = self.config["intermediate_size_mlp"]
        self.vocab_size = self.config["vocab_size"]
        self.attn_temperature_tuning = self.config["attn_temperature_tuning"]
        self.tie_word_embeddings = self.config["tie_word_embeddings"] if "tie_word_embeddings" in self.config else False

        interleave_moe_layer_step = self.config["interleave_moe_layer_step"]
        no_rope_step = 4  # FIXME hyper param = attn_temperature_tuning
        self.moe_layers = math.floor(self.num_layers / interleave_moe_layer_step)
        self.no_rope_layers = math.floor(self.num_layers / no_rope_step)
        # assert self.moe_layers == len(self.config["moe_layers"])
        # assert self.no_rope_layers == self.config["no_rope_layers"].count(0)

    def calc_inference_math_ops(
        self, tokens: int, past_tokens: int = 0, batch: int = 1, return_break_down: bool = False
    ):
        lm_head_macs = self.hidden_size * self.vocab_size * tokens
        q_proj_macs = self.hidden_size * self.num_heads * self.head_dim * tokens
        kv_proj_macs = 2 * self.hidden_size * self.num_kv_heads * self.head_dim * tokens
        out_proj_macs = self.hidden_size * self.num_heads * self.head_dim * tokens

        attention_tokens = tokens + past_tokens
        attn_scales_macs = self.num_heads * self.head_dim * attention_tokens
        attention_qk_macs = self.num_heads * tokens * self.head_dim * attention_tokens
        attention_softmax_macs = self.num_heads * tokens * attention_tokens
        attention_qkv_macs = self.num_heads * tokens * attention_tokens * self.head_dim

        mlp_ffn_macs = self.intermediate_size_mlp * self.hidden_size * tokens
        mlp_matdot_macs = self.intermediate_size_mlp * tokens

        moe_router_macs = self.num_experts * self.hidden_size * tokens
        per_tok_experts = self.num_experts_per_tok + 1
        moe_experts_ffn_macs = self.intermediate_size * self.hidden_size * per_tok_experts * tokens
        moe_experts_matdot_macs = self.intermediate_size * per_tok_experts * tokens

        layer_attention_macs = (
            q_proj_macs + kv_proj_macs + out_proj_macs + attention_qk_macs + attention_softmax_macs + attention_qkv_macs
        )
        layer_moe_macs = moe_router_macs + moe_experts_ffn_macs * 3 + moe_experts_matdot_macs
        layer_mlp_macs = mlp_ffn_macs * 3 + mlp_matdot_macs

        scale = 2 * batch
        lyr, dense_lyr, moe_lyr, nope_lyr = (
            self.num_layers,
            self.num_layers - self.moe_layers,
            self.moe_layers,
            self.no_rope_layers,
        )

        complexity = (
            layer_attention_macs * lyr
            + attn_scales_macs * nope_lyr
            + layer_moe_macs * moe_lyr
            + layer_mlp_macs * dense_lyr
            + lm_head_macs
        ) * scale

        if not return_break_down:
            return complexity

        break_down = {}
        break_down.update({"q_proj": q_proj_macs * lyr * scale})
        break_down.update({"kv_proj": kv_proj_macs * lyr * scale})
        break_down.update({"out_proj": out_proj_macs * lyr * scale})
        break_down.update(
            {
                "self_attention": (attention_qk_macs + attention_softmax_macs + attention_qkv_macs) * lyr * scale
                + attn_scales_macs * nope_lyr * scale
            }
        )
        break_down.update({"mlp": layer_mlp_macs * dense_lyr * scale})
        break_down.update({"moe": layer_moe_macs * moe_lyr * scale})
        break_down.update({"lm_head": lm_head_macs * scale})
        return break_down

    def calc_inference_dram_bytes(
        self, tokens: int, past_tokens: int = 0, batch: int = 1, axwy: str = "a16w4", return_break_down: bool = False
    ):
        embedding_params = self.vocab_size * self.hidden_size
        lm_head_params = self.hidden_size * self.vocab_size

        q_proj_params = self.hidden_size * self.num_heads * self.head_dim
        kv_proj_params = 2 * self.hidden_size * self.num_kv_heads * self.head_dim
        out_proj_params = self.hidden_size * self.num_heads * self.head_dim
        mlp_ffn_params = self.intermediate_size_mlp * self.hidden_size
        moe_ffn_params = self.intermediate_size * self.hidden_size
        moe_router_params = self.hidden_size * self.num_experts

        q_activations = tokens * self.num_heads * self.head_dim
        kv_activations = 2 * tokens * self.num_kv_heads * self.head_dim
        kv_cache = 2 * past_tokens * self.num_kv_heads * self.head_dim
        hidden_states = tokens * self.hidden_size
        hidden_states_mlp = tokens * self.intermediate_size_mlp
        hidden_states_moe = tokens * self.intermediate_size
        output_states = 1 * self.vocab_size

        ab, wb = axwy_to_bytes(axwy)
        scale_a, scale_w = ab * batch, wb
        lyr, dense_lyr, moe_lyr = self.num_layers, self.num_layers - self.moe_layers, self.moe_layers
        activated_experts = how_many_experts(self.num_experts, tokens * batch, self.num_experts_per_tok) + 1

        # Calc DRAM
        layer_attention_params = q_proj_params + kv_proj_params + out_proj_params
        layer_mlp_params = mlp_ffn_params * 3
        layer_moe_params = moe_router_params + moe_ffn_params * 3 * activated_experts

        transformer_params = layer_attention_params * lyr + layer_moe_params * moe_lyr + layer_mlp_params * dense_lyr
        head_and_tail_params = embedding_params + (lm_head_params if not self.tie_word_embeddings else 0)
        total_kv_cache = (kv_activations + kv_cache) * lyr

        max_hidden_states = max(hidden_states_mlp, hidden_states_moe)

        capacity = (transformer_params + head_and_tail_params) * scale_w + (
            total_kv_cache + hidden_states + max_hidden_states
        ) * scale_a

        # Calc IO
        embedding_io_bytes = embedding_params * scale_w + hidden_states * scale_a
        q_proj_io_bytes = q_proj_params * scale_w + (hidden_states + q_activations) * scale_a
        kv_proj_io_bytes = kv_proj_params * scale_w + (hidden_states + kv_activations) * scale_a
        self_attention_io_bytes = (q_activations + kv_activations + kv_cache + hidden_states) * scale_a
        out_proj_io_bytes = out_proj_params * scale_w + (hidden_states + q_activations) * scale_a
        mlp_ffn_io_bytes = mlp_ffn_params * scale_w + (hidden_states + hidden_states_mlp) * scale_a
        moe_ffn_io_bytes = moe_ffn_params * activated_experts * scale_w + (hidden_states + hidden_states_moe) * scale_a
        mlp_gate_io_bytes = hidden_states_mlp * 3 * scale_a
        moe_gate_io_bytes = hidden_states_moe * 3 * scale_a
        moe_router_io_bytes = moe_router_params * scale_w + hidden_states * scale_a
        lm_head_io_bytes = lm_head_params * scale_w + (hidden_states + output_states) * scale_a

        total_attention_io_bytes = (
            q_proj_io_bytes + kv_proj_io_bytes + out_proj_io_bytes + self_attention_io_bytes
        ) * lyr
        total_mlp_io_bytes = (mlp_ffn_io_bytes * 3 + mlp_gate_io_bytes) * dense_lyr
        total_moe_io_bytes = (moe_ffn_io_bytes * 3 + moe_gate_io_bytes + moe_router_io_bytes) * moe_lyr
        head_tail_io_bytes = embedding_io_bytes + lm_head_io_bytes

        io = total_attention_io_bytes + total_mlp_io_bytes + total_moe_io_bytes + head_tail_io_bytes

        if not return_break_down:
            return capacity, io

        io_break_down = {}
        io_break_down.update({"embedding": embedding_io_bytes})
        io_break_down.update({"q_proj": q_proj_io_bytes * lyr})
        io_break_down.update({"kv_proj": kv_proj_io_bytes * lyr})
        io_break_down.update({"self_attention": self_attention_io_bytes * lyr})
        io_break_down.update({"out_proj": out_proj_io_bytes * lyr})
        io_break_down.update({"mlp": total_mlp_io_bytes})
        io_break_down.update({"moe": total_moe_io_bytes})
        io_break_down.update({"lm_head": lm_head_io_bytes})

        return capacity, io_break_down

    def calc_inference_comm_bytes(self):
        pass


class deepseek_v3:
    def __init__(self, config: dict, name: str = None):
        # config https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/configuration_deepseek_v3.py#L26
        # model https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
        # Deepseek has 2 MLA impls, naive and absorb.
        # Here we use the convenient naive impl to compute tops. But use the kvcache-efficient absorb impl to compute dram gbs.
        self.config = config
        self.name = name
        self.num_layers = self.config["num_hidden_layers"]
        self.first_k_dense_replace = self.config["first_k_dense_replace"]
        self.moe_layer_freq = self.config["moe_layer_freq"]
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = (
            self.config["num_key_value_heads"] if "num_key_value_heads" in self.config else self.num_heads
        )
        self.v_head_dim = self.config["v_head_dim"]
        self.kv_lora_rank = self.config["kv_lora_rank"]
        self.q_lora_rank = self.config["q_lora_rank"] if "q_lora_rank" in self.config else None
        self.qk_nope_head_dim = self.config["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.config["qk_rope_head_dim"]
        self.intermediate_size = self.config["intermediate_size"]
        self.moe_intermediate_size = self.config["moe_intermediate_size"]
        self.n_routed_experts = self.config["n_routed_experts"]
        self.n_shared_experts = self.config["n_shared_experts"]
        self.num_experts_per_tok = self.config["num_experts_per_tok"]
        self.n_group = self.config["n_group"]
        self.topk_group = self.config["topk_group"]
        self.vocab_size = self.config["vocab_size"]
        self.tie_word_embeddings = self.config["tie_word_embeddings"] if "tie_word_embeddings" in self.config else False

        self.num_moe_layers = (self.num_layers - self.first_k_dense_replace) / self.moe_layer_freq
        self.num_dense_layers = self.num_layers - self.num_moe_layers

    def calc_inference_math_ops(
        self, tokens: int, past_tokens: int = 0, batch: int = 1, return_break_down: bool = False
    ):
        lm_head_macs = self.hidden_size * self.vocab_size * tokens

        q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        kv_a_proj_dim = self.kv_lora_rank + self.qk_rope_head_dim
        kv_b_proj_dim = self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        if self.q_lora_rank:
            q_proj_macs = (
                self.hidden_size * self.q_lora_rank + self.q_lora_rank * self.num_heads * q_head_dim
            ) * tokens
        else:
            q_proj_macs = self.hidden_size * self.num_heads * q_head_dim * tokens
        kv_a_proj_macs = self.hidden_size * kv_a_proj_dim * tokens
        kv_b_proj_macs = self.kv_lora_rank * kv_b_proj_dim * tokens
        out_proj_macs = self.num_heads * self.v_head_dim * self.hidden_size * tokens

        attention_tokens = tokens + past_tokens
        attention_qk_macs = self.num_heads * tokens * q_head_dim * attention_tokens
        attention_softmax_macs = self.num_heads * tokens * attention_tokens
        attention_qkv_macs = self.num_heads * tokens * attention_tokens * self.v_head_dim

        mlp_ffn_macs = self.intermediate_size * self.hidden_size * tokens
        mlp_matdot_macs = self.intermediate_size * tokens

        moe_router_macs = self.n_routed_experts * self.hidden_size * tokens
        per_tok_experts = self.num_experts_per_tok + self.n_shared_experts
        moe_experts_ffn_macs = self.moe_intermediate_size * self.hidden_size * per_tok_experts * tokens
        moe_matdot_macs = self.moe_intermediate_size * per_tok_experts * tokens

        layer_attention_macs = (
            q_proj_macs
            + kv_a_proj_macs
            + kv_b_proj_macs
            + out_proj_macs
            + attention_qk_macs
            + attention_softmax_macs
            + attention_qkv_macs
        )
        layer_mlp_macs = mlp_ffn_macs * 3 + mlp_matdot_macs
        layer_moe_macs = moe_router_macs + moe_experts_ffn_macs * 3 + moe_matdot_macs

        scale = 2 * batch
        lyr, dense_lyr, moe_lyr = self.num_layers, self.num_dense_layers, self.num_moe_layers

        complexity = (
            layer_attention_macs * lyr + layer_moe_macs * moe_lyr + layer_mlp_macs * dense_lyr + lm_head_macs
        ) * scale

        if not return_break_down:
            return complexity

        break_down = {}
        break_down.update({"q_proj": q_proj_macs * lyr * scale})
        break_down.update({"kv_proj": (kv_a_proj_macs + kv_b_proj_macs) * lyr * scale})
        break_down.update({"out_proj": out_proj_macs * lyr * scale})
        break_down.update(
            {"self_attention": (attention_qk_macs + attention_softmax_macs + attention_qkv_macs) * lyr * scale}
        )
        break_down.update({"mlp": layer_mlp_macs * dense_lyr * scale})
        break_down.update({"moe": layer_moe_macs * moe_lyr * scale})
        break_down.update({"lm_head": lm_head_macs * scale})
        return break_down

    def calc_inference_dram_bytes(
        self, tokens: int, past_tokens: int = 0, batch: int = 1, axwy: str = "a16w4", return_break_down: bool = False
    ):
        embedding_params = self.vocab_size * self.hidden_size
        lm_head_params = self.hidden_size * self.vocab_size

        q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.q_lora_rank:
            q_proj_params = self.hidden_size * self.q_lora_rank + self.q_lora_rank * self.num_heads * q_head_dim
        else:
            q_proj_params = self.hidden_size * self.num_heads * q_head_dim
        kv_a_proj_params = self.hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim)
        kv_b_proj_params = self.kv_lora_rank * self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        out_proj_params = self.hidden_size * self.num_heads * self.v_head_dim
        mlp_ffn_params = self.intermediate_size * self.hidden_size
        moe_ffn_params = self.moe_intermediate_size * self.hidden_size
        moe_router_params = self.hidden_size * self.n_routed_experts

        # absorb mla
        q_activations = tokens * self.num_heads * q_head_dim
        kv_activations = tokens * self.kv_lora_rank
        k_pe_activations = tokens * self.qk_rope_head_dim
        attention_activations = tokens * self.num_heads * self.v_head_dim
        kv_cache = past_tokens * self.kv_lora_rank
        pe_cache = past_tokens * self.qk_rope_head_dim
        hidden_states = tokens * self.hidden_size
        hidden_states_mlp = tokens * self.intermediate_size
        hidden_states_moe = tokens * self.moe_intermediate_size
        output_states = 1 * self.vocab_size

        ab, wb = axwy_to_bytes(axwy)
        scale_a, scale_w = ab * batch, wb
        lyr, dense_lyr, moe_lyr = self.num_layers, self.num_dense_layers, self.num_moe_layers
        _e, _t, _k, _l = (
            self.n_routed_experts,
            tokens * batch,
            self.num_experts_per_tok,
            self.n_routed_experts // self.n_group * self.topk_group,
        )
        activated_experts = how_many_experts(_e, _t, _k, _l) + self.n_shared_experts

        # Calc DRAM
        layer_moe_params = moe_router_params + moe_ffn_params * 3 * activated_experts
        layer_mlp_params = mlp_ffn_params * 3
        layer_attention_params = q_proj_params + kv_b_proj_params + kv_a_proj_params + out_proj_params

        transformer_params = layer_attention_params * lyr + layer_mlp_params * dense_lyr + layer_moe_params * moe_lyr
        head_and_tail_params = embedding_params + (lm_head_params if not self.tie_word_embeddings else 0)
        total_kv_pe_cache = (kv_activations + k_pe_activations + kv_cache + pe_cache) * lyr

        max_hidden_states = max(hidden_states_mlp, hidden_states_moe)

        capacity = (transformer_params + head_and_tail_params) * scale_w + (
            total_kv_pe_cache + hidden_states + max_hidden_states
        ) * scale_a

        # Calc IO
        embedding_io_bytes = embedding_params * scale_w + hidden_states * scale_a
        q_proj_io_bytes = q_proj_params * scale_w + (hidden_states + q_activations) * scale_a
        kv_a_proj_io_bytes = kv_a_proj_params * scale_w + (hidden_states + kv_activations + k_pe_activations) * scale_a
        absorb_attention_io_bytes = (
            kv_b_proj_params * scale_w
            + (q_activations + kv_activations + k_pe_activations + kv_cache + pe_cache + attention_activations)
            * scale_a
        )
        out_proj_io_bytes = out_proj_params * scale_w + (hidden_states + attention_activations) * scale_a
        mlp_ffn_io_bytes = mlp_ffn_params * scale_w + (hidden_states + hidden_states_mlp) * scale_a
        moe_ffn_io_bytes = moe_ffn_params * activated_experts * scale_w + (hidden_states + hidden_states_moe) * scale_a
        mlp_gate_io_bytes = hidden_states_mlp * 3 * scale_a
        moe_gate_io_bytes = hidden_states_moe * 3 * scale_a
        moe_router_io_bytes = moe_router_params * scale_w + hidden_states * scale_a
        lm_head_io_bytes = lm_head_params * scale_w + (hidden_states + output_states) * scale_a

        total_attention_io_bytes = (q_proj_io_bytes + kv_a_proj_io_bytes + absorb_attention_io_bytes) * lyr
        total_mlp_io_bytes = (mlp_ffn_io_bytes * 3 + mlp_gate_io_bytes) * dense_lyr
        total_moe_io_bytes = (moe_ffn_io_bytes * 3 + moe_gate_io_bytes + moe_router_io_bytes) * moe_lyr
        head_tail_io_bytes = embedding_io_bytes + lm_head_io_bytes

        io = total_attention_io_bytes + total_mlp_io_bytes + total_moe_io_bytes + head_tail_io_bytes

        if not return_break_down:
            return capacity, io

        io_break_down = {}
        io_break_down.update({"embedding": embedding_io_bytes})
        io_break_down.update({"q_proj": q_proj_io_bytes * lyr})
        io_break_down.update({"kv_a_proj": kv_a_proj_io_bytes * lyr})
        io_break_down.update({"absorb_self_attention": absorb_attention_io_bytes * lyr})
        io_break_down.update({"out_proj": out_proj_io_bytes * lyr})
        io_break_down.update({"mlp": total_mlp_io_bytes})
        io_break_down.update({"moe": total_moe_io_bytes})
        io_break_down.update({"lm_head": lm_head_io_bytes})

        return capacity, io_break_down

    def calc_inference_comm_bytes(self):
        pass


def auto_model(path_or_hf_repo: str, cache_dir: str = None, custom_config: dict = None):
    config = get_model_config(path_or_hf_repo, cache_dir)
    model_type = config.get("model_type", None)

    # special config postprocess
    if model_type == "llama4":
        config = config["text_config"]

    if custom_config:
        config.update(custom_config)

    name = path_or_hf_repo.split("/")[-1]

    if model_type == "llama":
        return llama(config, name)
    elif model_type == "llama4":
        return llama4(config, name)
    elif model_type == "deepseek_v3":
        return deepseek_v3(config, name)
    else:
        raise NotImplementedError(f"Unsupported model: {model_type}")


def calc_inference_complexity(
    model,
    prompt: int = 1024,
    output: int = 128,
    batch: int = 1,
    axwy: str = "a16w4",
    verbose: bool = False,
):
    headers = ["Model", "Phase", "Precision", "Batch", "Prompt", "Output"]
    pvalues = [model.name, "prefill", axwy, batch, prompt, output]
    dvaules = [model.name, "decode (once)", axwy, batch, prompt, output]

    # p.
    p_math = model.calc_inference_math_ops(prompt, 0, batch, verbose)
    p_dram, p_io = model.calc_inference_dram_bytes(prompt, 0, batch, axwy, verbose)

    # d. use output/2 for average
    past_token = prompt + output / 2
    d_math = model.calc_inference_math_ops(1, past_token, batch, verbose)
    d_dram, d_io = model.calc_inference_dram_bytes(1, past_token, batch, axwy, verbose)

    # csv. TEST + CAPACITY (GBs) + OPS + IO (Bytes)
    if verbose:
        math_headers, dram_io_headers = list(p_math.keys()), list(p_io.keys())
        headers += (
            ["Required DRAM GBs"]
            + [f"{mh.capitalize()} OPs" for mh in math_headers]
            + [f"{dh.capitalize()} Bytes" for dh in dram_io_headers]
        )
        pvalues += [p_dram / 1e9] + [int(v) for v in p_math.values()] + [int(v) for v in p_io.values()]
        dvaules += [d_dram / 1e9] + [int(v) for v in d_math.values()] + [int(v) for v in d_io.values()]
    else:
        headers += ["Required DRAM GBs", "Math OPs", "Total IO Bytes"]
        pvalues += [p_dram / 1e9, p_math, p_io]
        dvaules += [d_dram / 1e9, d_math, d_io]

    table = [headers, pvalues, dvaules]
    if verbose:
        name = f"results/{model.name}_{axwy}_in{prompt}_out{output}_b{batch}.csv"
        with open(name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(table)
        print(f"Report saved to {name}")
    else:
        print(tabulate(table, headers="firstrow", tablefmt="rounded_grid", stralign="left", numalign="left"))
