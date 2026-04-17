# LLM VRAM Calculator

A web tool that estimates GPU memory (VRAM) requirements and **theoretical throughput (TPM)** for any Hugging Face model on Azure GPU VMs.  
Supports dense models, **GQA**, **MoE**, and **MLA** architectures automatically.

---

## Features

- **Any HuggingFace model** — enter a model ID, name, or URL (e.g. `meta-llama/Llama-3.1-8B-Instruct`)
- **Automatic architecture detection** — reads `config.json` or `params.json` from hugging face model card to detect layers, hidden size, KV heads, MoE experts, MLA latent dims
- **14 quantisation formats** — FP32, FP16, BF16, FP8, INT8, INT4, GPTQ, AWQ, GGUF Q2–Q8
- **MoE & MLA aware** — correctly calculates active parameters and compressed KV cache
- **Theoretical TPM** — estimates tokens/minute for each Azure GPU VM based on memory bandwidth
- **20+ Azure VM SKUs** — NV A10, NC T4, NC A100, NC H100 NVL, ND A100, ND H100 SXM, ND H200, ND MI300X
- **Gated model support** — Reach out to @ngjason to access private/gated model configs

---

## Quick Start

### 1. Install

```bash
cd llm-calculator
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements-aoai.txt
```

### 2. Configure environment

Create a `.env` file:

```env
# Azure OpenAI (used for AI analysis section)
AZURE_OPENAI_ENDPOINT=https://<your-resource>.services.ai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4.1

# HuggingFace token (optional — needed for gated models like Llama-2)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

### 3. Run

```bash
python3 app_aoai.py
```

Open [http://localhost:5000](http://localhost:5000).

### 4. Docker

```bash
docker build --platform linux/amd64 --provenance=false --sbom=false -t llm-sizing .
docker run -p 5000:5000 llm-sizing
docker push <YOUR REGISTRY>
```

---

## How to Use

1. **Enter a model** — type a HuggingFace model ID (e.g. `kakaocorp/kanana-2-30b-a3b-thinking-2601`) or paste a URL
2. **Set inference parameters** — adjust sequence length, batch size, and concurrent users to match your workload
3. **Select precision formats** — choose which quantisation levels to evaluate (FP16 is selected by default)
4. **Click Analyze** — the tool fetches the model config, calculates VRAM and TPM, and shows results
5. **Review GPU compatibility** — use the format dropdown to see which Azure VMs fit at each precision level

---

## Understanding the Results

### Model Summary

Shows the model name, architecture, parameter count, and context length. If the model uses **MoE** or **MLA**, colored tags appear:

- **MoE** (purple) — Mixture of Experts model, with the active parameter count shown (e.g. "MoE · 3.37B active")
- **MLA** (green) — Multi-head Latent Attention, which compresses the KV cache

### KV-Cache Breakdown

Shows the memory consumed by the key-value cache for your configured workload:

| Field | Meaning |
|-------|---------|
| **KV Cache** | Total KV cache memory for all sequences |
| **Activation Overhead** | Fixed overhead for intermediate tensors (0.5–2 GB depending on model size) |
| **Architecture** | Layers, KV heads, and head dimension detected from config |

### VRAM Table

For each selected precision format:

| Column | Formula |
|--------|---------|
| **Model Size** | `total_params × bits / 8` |
| **KV Cache** | Depends on architecture (see below) |
| **Total VRAM** | `Model Size + KV Cache + Activation Overhead` |

### GPU Compatibility Cards

For each Azure VM that can fit the model, shows:

| Metric | Meaning |
|--------|---------|
| **TPM** | Theoretical tokens per minute (total across all sequences) |
| **tok/s** | Tokens per second per user |
| **Bottleneck** | Whether the limit is `memory_bandwidth` (typical) or `compute` |
| **Headroom** | Remaining VRAM after loading the model |

---

## Architecture Differences: GQA vs MoE vs MLA

### GQA (Grouped Query Attention)

Used by: Llama-3, Mistral, Qwen, most modern dense models

- Uses fewer KV heads than query heads (e.g. 8 KV heads for 32 query heads)
- KV cache per token per layer: `2 × n_kv_heads × head_dim`
- All parameters are active every step → VRAM ≈ speed bottleneck

**Example:** Llama-3.1-8B — 8B params, all active, 8 KV heads

### MoE (Mixture of Experts)

Used by: Mixtral, DeepSeek-V2/V3, Kanana-2, Qwen-MoE

- Has many expert FFN blocks but only a few activate per token (e.g. 2 of 64)
- **VRAM**: must load ALL experts (total params)
- **Speed**: only reads ACTIVE expert weights per decode step → much faster than a dense model of the same total size
- The tool shows "active params" which drives the TPM calculation

**Example:** Kanana-2-30b — 30B total params, but only ~3.4B active per token → runs like a 3.4B model for throughput while having 30B quality

### MLA (Multi-head Latent Attention)

Used by: DeepSeek-V2/V3, Kanana-2 (any DeepSeekV3ForCausalLM architecture)

- Compresses K and V into a single small latent vector via a learned projection
- KV cache per token per layer: `kv_lora_rank + qk_rope_head_dim` (typically 512 + 64 = 576)
- Compared to GQA's `2 × n_kv_heads × head_dim` (e.g. 3072), this is **5× smaller**
- Means you can serve more concurrent users or longer contexts in the same VRAM

### Combined Impact (MoE + MLA)

Models like Kanana-2-30b use both:

| Metric | Without MoE+MLA | With MoE+MLA | Improvement |
|--------|-----------------|--------------|-------------|
| Active weight read/step | 55.88 GB | 6.28 GB | **8.9×** |
| KV cache per seq (4K) | 1.5 GB | 0.21 GB | **7.1×** |
| Theoretical TPM (A100) | ~17K | ~143K | **8.4×** |

---

## Interpreting TPM Numbers

The theoretical TPM is calculated assuming the **decode phase** is **memory-bandwidth-bound** (which it is for most LLM inference):

```
steps/sec = GPU_bandwidth / (active_weight_bytes + total_KV_cache)
TPM = steps/sec × total_sequences × 60
```

**Key points:**

- These are **theoretical maximums** — real-world throughput is typically 50–80% of this due to scheduling overhead, attention computation, and framework inefficiency
- **Higher batch size** → higher total TPM (GPU is better utilized), but lower per-user tok/s
- **Longer sequence length** → more KV cache → fewer steps/sec → lower TPM
- The **compute ceiling** (shown when bottleneck = "compute") means the GPU's FLOPS limit is reached before bandwidth — this is rare for decode but can happen with very small models on powerful GPUs

**Rules of thumb:**

| TPM | Rough interpretation |
|-----|---------------------|
| < 5K | Very slow, consider quantisation or a faster GPU |
| 5K–20K | Acceptable for single-user or low-traffic scenarios |
| 20K–100K | Good for production workloads |
| > 100K | Excellent throughput — MoE models or quantised small models |

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Backend | Python 3.12, Flask, Gunicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Model API | HuggingFace Hub REST API |
| LLM Analysis | Azure OpenAI (gpt-4.1) via DefaultAzureCredential |
| Auth | Azure Identity (DefaultAzureCredential), optional HF_TOKEN |
| Deployment | Docker → Azure Container Registry → Azure Container Apps |

---

## License

MIT
