import os
import re
import json
import math
import requests
from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

import benchmark_k8s

load_dotenv()

app = Flask(__name__)

# ---------- constants ----------
GPU_DATABASE = [
    # ---- NV-series (A10) — Visualization / light inference ----
    # bandwidth_gbs = per-GPU memory bandwidth in GB/s
    # bf16_tflops   = per-GPU BF16 tensor TFLOPS (without sparsity)
    {"name": "NV6ads A10 v5",       "vram_gb": 4,    "gpu": "1/6 A10",    "gpus": 1, "series": "NV A10",    "bandwidth_gbs": 100,  "bf16_tflops": 20.8},
    {"name": "NV12ads A10 v5",      "vram_gb": 8,    "gpu": "1/3 A10",    "gpus": 1, "series": "NV A10",    "bandwidth_gbs": 200,  "bf16_tflops": 41.6},
    {"name": "NV18ads A10 v5",      "vram_gb": 12,   "gpu": "1/2 A10",    "gpus": 1, "series": "NV A10",    "bandwidth_gbs": 300,  "bf16_tflops": 62.5},
    {"name": "NV36ads A10 v5",      "vram_gb": 24,   "gpu": "1× A10",     "gpus": 1, "series": "NV A10",    "bandwidth_gbs": 600,  "bf16_tflops": 125},
    {"name": "NV36adms A10 v5",     "vram_gb": 24,   "gpu": "1× A10",     "gpus": 1, "series": "NV A10",    "bandwidth_gbs": 600,  "bf16_tflops": 125},
    {"name": "NV72ads A10 v5",      "vram_gb": 48,   "gpu": "2× A10",     "gpus": 2, "series": "NV A10",    "bandwidth_gbs": 600,  "bf16_tflops": 125},

    # ---- NC-series T4 — Entry-level inference ----
    {"name": "NC4as T4 v3",         "vram_gb": 16,   "gpu": "1× T4",      "gpus": 1, "series": "NC T4",     "bandwidth_gbs": 300,  "bf16_tflops": 65},
    {"name": "NC8as T4 v3",         "vram_gb": 16,   "gpu": "1× T4",      "gpus": 1, "series": "NC T4",     "bandwidth_gbs": 300,  "bf16_tflops": 65},
    {"name": "NC16as T4 v3",        "vram_gb": 16,   "gpu": "1× T4",      "gpus": 1, "series": "NC T4",     "bandwidth_gbs": 300,  "bf16_tflops": 65},
    {"name": "NC64as T4 v3",        "vram_gb": 64,   "gpu": "4× T4",      "gpus": 4, "series": "NC T4",     "bandwidth_gbs": 300,  "bf16_tflops": 65},

    # ---- NC-series A100 — LLM inference / fine-tuning ----
    # NC A100 v4 uses A100 80GB PCIe: ~1935 GB/s HBM2e, ~312 BF16 dense TFLOPS
    {"name": "NC24ads A100 v4",     "vram_gb": 80,   "gpu": "1× A100 80G","gpus": 1, "series": "NC A100",   "bandwidth_gbs": 1935, "bf16_tflops": 312},
    {"name": "NC48ads A100 v4",     "vram_gb": 160,  "gpu": "2× A100 80G","gpus": 2, "series": "NC A100",   "bandwidth_gbs": 1935, "bf16_tflops": 312},
    {"name": "NC96ads A100 v4",     "vram_gb": 320,  "gpu": "4× A100 80G","gpus": 4, "series": "NC A100",   "bandwidth_gbs": 1935, "bf16_tflops": 312},

    # ---- NC-series H100 — High-perf LLM inference ----
    # H100 NVL (94 GB HBM3): ~3.9 TB/s memory bandwidth, ~835 BF16 dense TFLOPS
    {"name": "NC40ads H100 v5",     "vram_gb": 94,   "gpu": "1× H100 NVL","gpus": 1, "series": "NC H100",   "bandwidth_gbs": 3900, "bf16_tflops": 835},
    {"name": "NC80adis H100 v5",    "vram_gb": 188,  "gpu": "2× H100 NVL","gpus": 2, "series": "NC H100",   "bandwidth_gbs": 3900, "bf16_tflops": 835},

    # ---- NC-series RTX PRO 6000 Blackwell Server Edition (v6, Preview) — Inference / RAG / VDI ----
    # Per-GPU: 96 GB GDDR7, ~1792 GB/s, BF16 Tensor (dense) ~503 TFLOPS
    {"name": "NC128ds_xl RTXPRO6000 v6",   "vram_gb": 96,   "gpu": "1× RTX PRO 6000 96G", "gpus": 1, "series": "NC RTX PRO 6000", "bandwidth_gbs": 1792, "bf16_tflops": 503},
    {"name": "NC256ds_xl RTXPRO6000 v6",   "vram_gb": 192,  "gpu": "2× RTX PRO 6000 96G", "gpus": 2, "series": "NC RTX PRO 6000", "bandwidth_gbs": 1792, "bf16_tflops": 503},
    {"name": "NC320ds_xl RTXPRO6000 v6",   "vram_gb": 192,  "gpu": "2× RTX PRO 6000 96G", "gpus": 2, "series": "NC RTX PRO 6000", "bandwidth_gbs": 1792, "bf16_tflops": 503},
    {"name": "NC128lds_xl RTXPRO6000 v6",  "vram_gb": 96,   "gpu": "1× RTX PRO 6000 96G", "gpus": 1, "series": "NC RTX PRO 6000", "bandwidth_gbs": 1792, "bf16_tflops": 503},
    {"name": "NC256lds_xl RTXPRO6000 v6",  "vram_gb": 192,  "gpu": "2× RTX PRO 6000 96G", "gpus": 2, "series": "NC RTX PRO 6000", "bandwidth_gbs": 1792, "bf16_tflops": 503},
    {"name": "NC320lds_xl RTXPRO6000 v6",  "vram_gb": 192,  "gpu": "2× RTX PRO 6000 96G", "gpus": 2, "series": "NC RTX PRO 6000", "bandwidth_gbs": 1792, "bf16_tflops": 503},

    # ---- ND-series A100 — Training / large-scale inference ----
    {"name": "ND96asr A100 v4",     "vram_gb": 320,  "gpu": "8× A100 40G","gpus": 8, "series": "ND A100",   "bandwidth_gbs": 1555, "bf16_tflops": 312},
    {"name": "ND96amsr A100 v4",    "vram_gb": 640,  "gpu": "8× A100 80G","gpus": 8, "series": "ND A100",   "bandwidth_gbs": 2039, "bf16_tflops": 312},

    # ---- ND-series H100 — Frontier training / inference ----
    {"name": "ND96isr H100 v5",     "vram_gb": 640,  "gpu": "8× H100 SXM","gpus": 8, "series": "ND H100",   "bandwidth_gbs": 3350, "bf16_tflops": 989},

    # ---- ND-series H200 — Next-gen large models ----
    {"name": "ND96isr H200 v5",     "vram_gb": 1128, "gpu": "8× H200 141G","gpus": 8,"series": "ND H200",   "bandwidth_gbs": 4800, "bf16_tflops": 989},

    # ---- ND-series MI300X — AMD HPC ----
    {"name": "ND96isr MI300X v5",   "vram_gb": 1536, "gpu": "8× MI300X 192G","gpus": 8,"series": "ND MI300X","bandwidth_gbs": 5300, "bf16_tflops": 1307},
]

QUANTIZATION_FORMATS = {
    "FP32":  {"bits": 32, "label": "FP32 (Full Precision)",   "description": "32-bit floating point – maximum precision, highest memory"},
    "FP16":  {"bits": 16, "label": "FP16 (Half Precision)",   "description": "16-bit floating point – standard training / inference precision"},
    "BF16":  {"bits": 16, "label": "BF16 (Brain Float 16)",   "description": "16-bit bfloat – same size as FP16, better dynamic range"},
    "FP8":   {"bits": 8,  "label": "FP8 (Float 8)",           "description": "8-bit floating point – hardware-accelerated on H100/MI300X, near-FP16 quality"},
    "INT8":  {"bits": 8,  "label": "INT8 (8-bit Integer)",    "description": "8-bit integer quantization – good quality, ~2× smaller than FP16"},
    "INT4":  {"bits": 4,  "label": "INT4 (4-bit Integer)",    "description": "4-bit integer quantization – ~4× smaller than FP16, slight quality loss"},
    "GPTQ":  {"bits": 4,  "label": "GPTQ (4-bit)",            "description": "GPU-optimised 4-bit post-training quantization"},
    "AWQ":   {"bits": 4,  "label": "AWQ (4-bit)",             "description": "Activation-aware weight quantization, 4-bit"},
    "GGUF_Q8":  {"bits": 8,  "label": "GGUF Q8_0",           "description": "llama.cpp 8-bit quantization"},
    "GGUF_Q6":  {"bits": 6,  "label": "GGUF Q6_K",           "description": "llama.cpp 6-bit quantization"},
    "GGUF_Q5":  {"bits": 5,  "label": "GGUF Q5_K_M",         "description": "llama.cpp 5-bit quantization"},
    "GGUF_Q4":  {"bits": 4,  "label": "GGUF Q4_K_M",         "description": "llama.cpp 4-bit quantization"},
    "GGUF_Q3":  {"bits": 3,  "label": "GGUF Q3_K_M",         "description": "llama.cpp 3-bit quantization – aggressive compression"},
    "GGUF_Q2":  {"bits": 2,  "label": "GGUF Q2_K",           "description": "llama.cpp 2-bit quantization – extreme compression, noticeable quality loss"},
}

# ---------- Azure OpenAI client (DefaultAzureCredential) ----------

def _get_aoai_client() -> AzureOpenAI:
    """
    Create an Azure OpenAI client authenticated via DefaultAzureCredential.
    Requires env vars:
      AZURE_OPENAI_ENDPOINT     – e.g. https://<resource>.services.ai.azure.com
      AZURE_OPENAI_API_VERSION  – (optional, defaults to 2024-12-01-preview)
    """
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=os.getenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://jianshn-eastus2-foundry.services.ai.azure.com",
        ),
        azure_ad_token_provider=token_provider,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )

# ---------- helpers ----------

def parse_model_id(user_input: str) -> str:
    """Extract an org/model style id from a HuggingFace URL or plain text."""
    user_input = user_input.strip().rstrip("/")
    m = re.match(r"https?://huggingface\.co/([^/]+/[^/]+)", user_input)
    if m:
        return m.group(1)
    if "/" in user_input:
        return user_input
    return user_input


def _hf_headers() -> dict:
    """Return Authorization header if HF_TOKEN is set."""
    token = os.getenv("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch_model_info(model_id: str) -> dict | None:
    """Fetch model metadata from the Hugging Face Hub API."""
    url = f"https://huggingface.co/api/models/{model_id}"
    resp = requests.get(url, headers=_hf_headers(), timeout=15)
    if resp.status_code == 200:
        return resp.json()
    return None


def fetch_model_config(model_id: str) -> tuple[dict | None, str | None]:
    """Try to grab config.json (then params.json) for architecture details.
    Returns (config_dict, warning_message)."""
    for filename in ("config.json", "params.json"):
        url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
        try:
            resp = requests.get(url, headers=_hf_headers(), timeout=15)
        except Exception:
            continue
        if resp.status_code == 200:
            try:
                return resp.json(), None
            except Exception:
                continue
        if resp.status_code in (401, 403):
            return None, ("This is a gated model — config files are not accessible. "
                          "Set HF_TOKEN in your .env file with a token that has access. "
                          "Architecture values are estimated from the parameter count.")
    return None, ("config.json / params.json not found for this model. "
                  "Architecture values are estimated from the parameter count.")


def estimate_params_from_config(config: dict) -> int | None:
    """Rough parameter estimation from a transformers config.json."""
    try:
        h = config.get("hidden_size", config.get("d_model", config.get("dim")))
        n_layers = config.get("num_hidden_layers", config.get("n_layer", config.get("num_layers", config.get("n_layers"))))
        v = config.get("vocab_size")
        i = config.get("intermediate_size", config.get("d_ff"))
        if h and n_layers and v:
            if i is None:
                i = 4 * h
            params = v * h
            params += n_layers * (4 * h * h)
            params += n_layers * (2 * h * i)
            params += v * h
            return int(params)
    except Exception:
        pass
    return None


def extract_param_count(model_info: dict, config: dict | None) -> float | None:
    """Try every method to find parameter count. Returns billions."""
    # 1. safetensors metadata
    safetensors = model_info.get("safetensors")
    if safetensors and isinstance(safetensors, dict):
        params = safetensors.get("total")
        if params:
            return params / 1e9

    # 2. model card metadata
    card = model_info.get("cardData", {}) or {}
    for key in ("parameter_count", "num_parameters", "parameters"):
        val = card.get(key)
        if val:
            if isinstance(val, (int, float)):
                return val / 1e9 if val > 1e6 else val
            if isinstance(val, str):
                val_clean = val.lower().replace(",", "").strip()
                m = re.search(r"([\d.]+)\s*[bB]", val_clean)
                if m:
                    return float(m.group(1))
                try:
                    return float(val_clean) / 1e9
                except ValueError:
                    pass

    # 3. model name heuristic
    model_id = model_info.get("modelId", "")
    m = re.search(r"(\d+\.?\d*)\s*[bB]", model_id)
    if m:
        return float(m.group(1))

    # 4. Estimate from config.json
    if config:
        est = estimate_params_from_config(config)
        if est:
            return est / 1e9

    return None


def estimate_arch_from_params(param_billions: float) -> dict:
    """Estimate architecture details from parameter count when config.json is unavailable."""
    params = param_billions * 1e9
    if params < 1e9:
        return {"num_hidden_layers": 12, "hidden_size": 768, "num_kv_heads": 12, "num_attention_heads": 12, "head_dim": 64}
    elif params < 3e9:
        return {"num_hidden_layers": 24, "hidden_size": 2048, "num_kv_heads": 16, "num_attention_heads": 16, "head_dim": 128}
    elif params < 10e9:
        return {"num_hidden_layers": 32, "hidden_size": 4096, "num_kv_heads": 8, "num_attention_heads": 32, "head_dim": 128}
    elif params < 20e9:
        return {"num_hidden_layers": 40, "hidden_size": 5120, "num_kv_heads": 8, "num_attention_heads": 40, "head_dim": 128}
    elif params < 40e9:
        return {"num_hidden_layers": 48, "hidden_size": 6144, "num_kv_heads": 8, "num_attention_heads": 48, "head_dim": 128}
    elif params < 80e9:
        return {"num_hidden_layers": 64, "hidden_size": 8192, "num_kv_heads": 8, "num_attention_heads": 64, "head_dim": 128}
    else:
        return {"num_hidden_layers": 80, "hidden_size": 8192, "num_kv_heads": 8, "num_attention_heads": 64, "head_dim": 128}


def get_arch_params(config: dict | None, param_billions: float) -> dict:
    """Extract architecture params from config, falling back to estimates.
    Also detects MoE and MLA (Multi-head Latent Attention) properties."""
    if config:
        n_layers = config.get("num_hidden_layers", config.get("n_layer", config.get("num_layers", config.get("n_layers"))))
        h = config.get("hidden_size", config.get("d_model", config.get("dim")))
        n_heads = config.get("num_attention_heads", config.get("n_head", config.get("n_heads")))
        n_kv_heads = config.get("num_key_value_heads", config.get("n_kv_heads", n_heads))
        explicit_head_dim = config.get("head_dim")
        if n_layers and h and n_heads:
            arch = {
                "num_hidden_layers": n_layers,
                "hidden_size": h,
                "num_kv_heads": n_kv_heads or n_heads,
                "num_attention_heads": n_heads,
                "head_dim": explicit_head_dim if explicit_head_dim else h // n_heads,
            }
            # --- MLA detection (DeepSeek-V2/V3 style) ---
            kv_lora_rank = config.get("kv_lora_rank")
            qk_rope_head_dim = config.get("qk_rope_head_dim")
            if kv_lora_rank and qk_rope_head_dim:
                arch["mla"] = {
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "v_head_dim": config.get("v_head_dim", 128),
                }
            # --- MoE detection ---
            n_routed = config.get("n_routed_experts", config.get("num_local_experts"))
            n_selected = config.get("num_experts_per_tok", config.get("num_selected_experts"))
            if n_routed and n_selected:
                n_shared = config.get("n_shared_experts", 0)
                first_dense = config.get("first_k_dense_replace", 0)
                moe_inter = config.get("moe_intermediate_size", config.get("intermediate_size", 4 * h))
                dense_inter = config.get("intermediate_size", 4 * h)
                arch["moe"] = {
                    "n_routed_experts": n_routed,
                    "num_experts_per_tok": n_selected,
                    "n_shared_experts": n_shared,
                    "first_k_dense_replace": first_dense,
                    "moe_intermediate_size": moe_inter,
                    "dense_intermediate_size": dense_inter,
                }
            return arch
    return estimate_arch_from_params(param_billions)


def calculate_kv_cache_gb(arch: dict, sequence_length: int, batch_size: int,
                          concurrent_users: int, kv_bits: int = 16) -> float:
    """
    Calculate KV-cache memory in GB.

    Standard MHA/GQA:
        kv_cache = 2 × num_layers × num_kv_heads × head_dim × seq_len × seqs × bytes

    MLA (Multi-head Latent Attention, e.g. DeepSeek-V2/V3):
        The KV cache stores a single compressed latent per token per layer:
        kv_cache = num_layers × (kv_lora_rank + qk_rope_head_dim) × seq_len × seqs × bytes
    """
    n_layers = arch["num_hidden_layers"]
    total_sequences = batch_size * concurrent_users
    bytes_per_elem = kv_bits / 8

    mla = arch.get("mla")
    if mla:
        # MLA: single compressed vector per token per layer
        latent_dim = mla["kv_lora_rank"] + mla["qk_rope_head_dim"]
        kv_bytes = n_layers * latent_dim * sequence_length * total_sequences * bytes_per_elem
    else:
        # Standard MHA/GQA: separate K and V tensors
        n_kv_heads = arch["num_kv_heads"]
        n_heads = arch["num_attention_heads"]
        hidden = arch["hidden_size"]
        head_dim = arch.get("head_dim", hidden // n_heads)
        kv_bytes = 2 * n_layers * n_kv_heads * head_dim * sequence_length * total_sequences * bytes_per_elem

    return kv_bytes / (1024 ** 3)


def calculate_vram(param_billions: float, config: dict | None,
                   sequence_length: int = 2048, batch_size: int = 1,
                   concurrent_users: int = 1) -> tuple[list[dict], dict]:
    """
    Calculate VRAM requirements for every quantisation format.
    Returns (vram_table, kv_info) where kv_info has the architecture used.
    """
    arch = get_arch_params(config, param_billions)
    kv_cache_gb = calculate_kv_cache_gb(arch, sequence_length, batch_size, concurrent_users)
    activation_overhead_gb = 0.5 if param_billions < 15 else (1.0 if param_billions < 70 else 2.0)

    results = []
    for key, fmt in QUANTIZATION_FORMATS.items():
        model_size_gb = (param_billions * 1e9 * fmt["bits"] / 8) / (1024 ** 3)
        total_vram_gb = model_size_gb + kv_cache_gb + activation_overhead_gb
        results.append({
            "format": key,
            "label": fmt["label"],
            "description": fmt["description"],
            "bits": fmt["bits"],
            "model_size_gb": round(model_size_gb, 2),
            "kv_cache_gb": round(kv_cache_gb, 2),
            "activation_overhead_gb": round(activation_overhead_gb, 2),
            "vram_required_gb": round(total_vram_gb, 2),
        })

    active_billions = estimate_active_params(arch, param_billions)
    kv_info = {
        "kv_cache_gb": round(kv_cache_gb, 2),
        "activation_overhead_gb": round(activation_overhead_gb, 2),
        "arch_source": "config" if config else "estimated",
        "num_layers": arch["num_hidden_layers"],
        "hidden_size": arch["hidden_size"],
        "num_kv_heads": arch["num_kv_heads"],
        "num_attention_heads": arch["num_attention_heads"],
        "head_dim": arch.get("head_dim", arch["hidden_size"] // arch["num_attention_heads"]),
        "total_sequences": batch_size * concurrent_users,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "concurrent_users": concurrent_users,
        "is_moe": "moe" in arch,
        "is_mla": "mla" in arch,
        "active_param_billions": round(active_billions, 2),
    }
    if "moe" in arch:
        kv_info["moe_info"] = arch["moe"]
    if "mla" in arch:
        kv_info["mla_info"] = arch["mla"]

    return results, kv_info


def estimate_active_params(arch: dict, param_billions: float) -> float:
    """Estimate the number of *active* parameters (in billions) per decode step.

    For dense models this equals total params.
    For MoE models, only the selected experts (+ shared experts) and the
    non-expert layers are read per step, which is much less than total params.
    """
    moe = arch.get("moe")
    if not moe:
        return param_billions

    h = arch["hidden_size"]
    n_layers = arch["num_hidden_layers"]
    n_routed = moe["n_routed_experts"]
    n_selected = moe["num_experts_per_tok"]
    n_shared = moe["n_shared_experts"]
    first_dense = moe["first_k_dense_replace"]
    moe_inter = moe["moe_intermediate_size"]
    dense_inter = moe["dense_intermediate_size"]

    n_moe_layers = n_layers - first_dense

    # Per-expert params: gate_proj + up_proj + down_proj = 3 × h × moe_inter
    params_per_expert = 3 * h * moe_inter

    # Total expert params across all MoE layers (stored in VRAM)
    total_expert_params = n_moe_layers * n_routed * params_per_expert

    # Active expert params per step: selected + shared experts
    active_expert_params = n_moe_layers * (n_selected + n_shared) * params_per_expert

    # Non-expert params (attention, embeddings, dense FFN layers, norms, etc.)
    non_expert_params = (param_billions * 1e9) - total_expert_params

    active_total = non_expert_params + active_expert_params
    return max(active_total / 1e9, 0.1)  # safety floor


def calculate_theoretical_tpm(vm: dict, model_size_gb: float,
                              kv_cache_per_seq_gb: float,
                              param_billions: float,
                              total_sequences: int,
                              active_param_billions: float | None = None) -> dict:
    """
    Calculate theoretical throughput metrics for a model on a given GPU VM.

    For MoE models, `active_param_billions` should be the number of params
    actually read per decode step (selected + shared experts + non-expert layers).
    The `model_size_gb` still reflects total weights for VRAM fitting, but
    bandwidth uses the active weight size.

    The decode phase is memory-bandwidth-bound:
        steps/s = total_bandwidth / (active_weight_bytes + total_kv_cache)
        total_tok/s = steps/s × total_sequences

    The compute ceiling is:
        max_tok/s = (bf16_tflops × 1e12) / (2 × active_params)
    """
    if active_param_billions is None:
        active_param_billions = param_billions

    n_gpus = vm["gpus"]
    bw_per_gpu = vm["bandwidth_gbs"]
    tflops_per_gpu = vm["bf16_tflops"]

    # Total bandwidth across all GPUs in the VM (tensor-parallel)
    total_bandwidth_gbs = bw_per_gpu * n_gpus

    # Total KV cache for all concurrent sequences
    total_kv_gb = kv_cache_per_seq_gb * total_sequences

    # Active weight bytes read per decode step (MoE: only active experts)
    # Use same bits-per-param ratio as model_size_gb / param_billions
    if param_billions > 0:
        bytes_per_param = (model_size_gb * 1024**3) / (param_billions * 1e9)
    else:
        bytes_per_param = 2  # default FP16
    active_weight_gb = (active_param_billions * 1e9 * bytes_per_param) / (1024**3)

    bytes_per_step_gb = active_weight_gb + total_kv_gb

    if bytes_per_step_gb <= 0 or total_bandwidth_gbs <= 0:
        return {"output_tok_per_sec": 0, "tpm": 0, "per_user_tok_per_sec": 0,
                "compute_ceiling_tok_per_sec": 0, "bottleneck": "unknown"}

    # Bandwidth-limited throughput
    steps_per_sec = total_bandwidth_gbs / bytes_per_step_gb
    bw_limited_tok_s = steps_per_sec * total_sequences

    # Compute-limited ceiling (using active params for FLOPs)
    total_tflops = tflops_per_gpu * n_gpus
    flops_per_token = 2 * active_param_billions * 1e9
    compute_ceiling_tok_s = (total_tflops * 1e12) / flops_per_token if flops_per_token > 0 else float("inf")

    # Actual throughput is the minimum of bandwidth-limited and compute-limited
    effective_tok_s = min(bw_limited_tok_s, compute_ceiling_tok_s)
    bottleneck = "memory_bandwidth" if bw_limited_tok_s <= compute_ceiling_tok_s else "compute"

    per_user = effective_tok_s / total_sequences if total_sequences > 0 else effective_tok_s
    tpm = effective_tok_s * 60

    # Max sequences this VM can fit in remaining VRAM (use total model_size for VRAM)
    available_vram = vm["vram_gb"] - model_size_gb - 3  # 3 GB overhead
    max_sequences = max(0, int(available_vram / kv_cache_per_seq_gb)) if kv_cache_per_seq_gb > 0 else 0

    return {
        "output_tok_per_sec": round(effective_tok_s, 1),
        "tpm": round(tpm),
        "per_user_tok_per_sec": round(per_user, 1),
        "total_bandwidth_gbs": round(total_bandwidth_gbs),
        "bytes_per_step_gb": round(bytes_per_step_gb, 2),
        "active_weight_gb": round(active_weight_gb, 2),
        "compute_ceiling_tok_per_sec": round(compute_ceiling_tok_s, 1),
        "bottleneck": bottleneck,
        "max_sequences": max_sequences,
    }


def gpu_compatibility(vram_required: float, model_size_gb: float = 0,
                      kv_cache_per_seq_gb: float = 0,
                      param_billions: float = 0,
                      total_sequences: int = 1,
                      active_param_billions: float | None = None) -> list[dict]:
    """Which Azure VM SKUs can run a given VRAM requirement, with theoretical TPM."""
    result = []
    for vm in GPU_DATABASE:
        fits = vm["vram_gb"] >= vram_required
        headroom = vm["vram_gb"] - vram_required
        entry = {
            "name": vm["name"],
            "vram_gb": vm["vram_gb"],
            "gpu": vm["gpu"],
            "gpus": vm["gpus"],
            "series": vm["series"],
            "fits": fits,
            "headroom_gb": round(headroom, 2),
        }
        # Add theoretical TPM if the model fits
        if fits and model_size_gb > 0 and param_billions > 0:
            entry["throughput"] = calculate_theoretical_tpm(
                vm, model_size_gb, kv_cache_per_seq_gb,
                param_billions, total_sequences,
                active_param_billions=active_param_billions,
            )
        result.append(entry)
    result.sort(key=lambda g: g["vram_gb"])
    return result


def get_llm_analysis(model_id: str, model_info: dict, config: dict | None,
                     param_billions: float | None, vram_table: list[dict] | None,
                     kv_info: dict | None = None) -> str:
    """Use Azure OpenAI to provide a rich, contextual analysis of the model's memory footprint."""
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    try:
        client = _get_aoai_client()
    except Exception as e:
        return f"⚠️ Azure OpenAI client init failed: {str(e)}"

    # Build a context payload for the LLM
    context_parts = [f"Model: {model_id}"]
    if model_info:
        context_parts.append(f"Tags: {model_info.get('tags', [])}")
        context_parts.append(f"Pipeline: {model_info.get('pipeline_tag', 'unknown')}")
        context_parts.append(f"Library: {model_info.get('library_name', 'unknown')}")
    if config:
        for k in ("architectures", "model_type", "hidden_size", "num_hidden_layers",
                   "num_attention_heads", "intermediate_size", "vocab_size",
                   "num_key_value_heads", "max_position_embeddings"):
            if k in config:
                context_parts.append(f"{k}: {config[k]}")
    if param_billions:
        context_parts.append(f"Estimated parameters: {param_billions:.2f}B")
    if kv_info:
        context_parts.append(f"\nInference parameters:")
        context_parts.append(f"  Batch size: {kv_info['batch_size']}")
        context_parts.append(f"  Sequence length: {kv_info['sequence_length']} tokens")
        context_parts.append(f"  Concurrent users: {kv_info['concurrent_users']}")
        context_parts.append(f"  Total active sequences: {kv_info['total_sequences']}")
        context_parts.append(f"  KV-cache memory: {kv_info['kv_cache_gb']} GB")
        context_parts.append(f"  Architecture ({kv_info['arch_source']}): {kv_info['num_layers']}L, hidden={kv_info['hidden_size']}, kv_heads={kv_info['num_kv_heads']}, head_dim={kv_info['head_dim']}")
    if vram_table:
        context_parts.append("\nVRAM estimates (GB) by quantisation:")
        for row in vram_table:
            context_parts.append(f"  {row['label']}: weights={row['model_size_gb']}GB + KV-cache={row['kv_cache_gb']}GB + overhead={row['activation_overhead_gb']}GB = total {row['vram_required_gb']}GB")

    context_str = "\n".join(context_parts)

    system_prompt = """You are an expert ML engineer specialising in LLM deployment and GPU memory optimisation.
Given metadata about a model and its calculated VRAM estimates, provide a concise but insightful analysis covering:

1. **Parameter Estimate Confidence** – how confident you are in the parameter count and why.
2. **Recommended Quantisation** – which quantisation(s) offer the best quality/memory trade-off for this specific architecture.
3. **Deployment Tips** – practical advice for running this model (e.g. tensor parallelism, offloading, KV-cache optimisation, context length impact).
4. **Context Length Impact** – how increasing context length affects VRAM (KV-cache scaling).
5. **Multi-GPU Guidance** – when multi-GPU or model-parallelism is required and how to set it up.
6. **Azure VM Compatibility** – Always show the cheapest Azure VM SKUs that can run the model at each quantisation level, based on the VRAM requirements. For example, NC first than ND.
7. **Model-Specific Insights** – Determine if the model is an MoE model and how many parameters are in the experts vs. the base. Surface any exceptions for example. gpt-oss-120b can be run on a single NC40ads_H100_v5 because the experts are not loaded into memory, but a 70B dense model cannot.

Keep the response under 400 words. Use markdown formatting. Be specific to THIS model."""

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_str},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM analysis unavailable: {str(e)}"


# ---------- routes ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    user_input = data.get("model", "").strip()
    if not user_input:
        return jsonify({"error": "Please provide a model name, ID, or HuggingFace URL."}), 400

    model_id = parse_model_id(user_input)

    # Inference parameters (with sensible defaults)
    sequence_length = max(1, int(data.get("sequence_length", 2048)))
    batch_size = max(1, int(data.get("batch_size", 1)))
    concurrent_users = max(1, int(data.get("concurrent_users", 1)))

    # Precision format filter (optional — defaults to all)
    requested_formats = data.get("formats")
    valid_formats = set(QUANTIZATION_FORMATS.keys())
    if requested_formats and isinstance(requested_formats, list):
        selected_formats = [f for f in requested_formats if f in valid_formats]
    else:
        selected_formats = list(valid_formats)

    if not selected_formats:
        return jsonify({"error": "Please select at least one precision mode."}), 400

    try:
        # Fetch from HuggingFace
        model_info = fetch_model_info(model_id)
        if not model_info:
            return jsonify({"error": f"Could not find model '{model_id}' on Hugging Face. Check the name and try again."}), 404

        config, config_warning = fetch_model_config(model_id)
        param_billions = extract_param_count(model_info, config)

        if param_billions is None:
            return jsonify({
                "error": f"Could not determine parameter count for '{model_id}'. "
                         "The model may not have published parameter metadata."
            }), 422

        # Calculate VRAM with KV-cache
        vram_table, kv_info = calculate_vram(
            param_billions, config,
            sequence_length=sequence_length,
            batch_size=batch_size,
            concurrent_users=concurrent_users,
        )

        # Filter to only requested precision formats
        vram_table = [entry for entry in vram_table if entry["format"] in selected_formats]

        # GPU compatibility for selected quantizations only (with theoretical TPM)
        total_sequences = batch_size * concurrent_users
        arch = get_arch_params(config, param_billions)
        active_billions = estimate_active_params(arch, param_billions)
        # KV cache for a single sequence (used for per-GPU max_sequences calc)
        kv_per_seq_gb = calculate_kv_cache_gb(arch, sequence_length, 1, 1)

        gpu_compat = {}
        for entry in vram_table:
            gpu_compat[entry["format"]] = gpu_compatibility(
                entry["vram_required_gb"],
                model_size_gb=entry["model_size_gb"],
                kv_cache_per_seq_gb=kv_per_seq_gb,
                param_billions=param_billions,
                total_sequences=total_sequences,
                active_param_billions=active_billions,
            )

        # LLM analysis (included in response but not rendered in UI)
        llm_analysis = get_llm_analysis(model_id, model_info, config, param_billions, vram_table, kv_info)

        return jsonify({
            "model_id": model_info.get("modelId", model_id),
            "pipeline_tag": model_info.get("pipeline_tag", "unknown"),
            "library": model_info.get("library_name", "unknown"),
            "tags": model_info.get("tags", [])[:15],
            "param_billions": round(param_billions, 2),
            "vram_table": vram_table,
            "kv_info": kv_info,
            "gpu_compatibility": gpu_compat,
            "llm_analysis": llm_analysis,
            "architecture": config.get("architectures", ["unknown"])[0] if config else "unknown",
            "context_length": config.get("max_position_embeddings") if config else None,
            "config_warning": config_warning,
        })

    except requests.exceptions.Timeout:
        return jsonify({"error": f"Request to Hugging Face timed out for '{model_id}'. Please try again."}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not connect to Hugging Face. Check your internet connection."}), 502
    except Exception as e:
        app.logger.exception(f"Unexpected error analyzing model '{model_id}'")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/api/benchmark", methods=["POST"])
def submit_benchmark():
    """Provision vLLM Deployment + Service + benchmark Job in Kubernetes."""
    payload = request.get_json(force=True) or {}
    try:
        result = benchmark_k8s.submit_benchmark(payload)
        return jsonify({"status": "submitted", **result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Failed to submit benchmark job")
        return jsonify({"error": f"Failed to submit benchmark: {e}"}), 500


@app.route("/api/benchmark/<run_id>", methods=["GET"])
def benchmark_status(run_id: str):
    try:
        return jsonify(benchmark_k8s.get_status(run_id))
    except Exception as e:
        app.logger.exception("Failed to get benchmark status")
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmark/<run_id>", methods=["DELETE"])
def benchmark_cleanup(run_id: str):
    try:
        return jsonify(benchmark_k8s.cleanup(run_id))
    except Exception as e:
        app.logger.exception("Failed to clean up benchmark")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
