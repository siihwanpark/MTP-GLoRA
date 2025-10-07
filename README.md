<div align="center">

# MTP-GLoRA

**Production-Ready Multi-Token Prediction with Gated LoRA**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![PyTorch>=2.8](https://img.shields.io/badge/pytorch-%E2%89%A52.8-orange?logo=pytorch) ![Transformers>=4.56](https://img.shields.io/badge/transformers-%E2%89%A54.56-green?logo=huggingface)

*Efficient Training Framework for multi-token prediction through gated LoRA*

[Features](#-key-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture-overview) • [Documentation](#-detailed-features)

---

</div>

## Overview

**MTP-GLoRA** is a high-performance training framework for **Multiple Token Prediction (MTP)** with **Gated LoRA** adaptation on Hugging Face Transformer models. Inspired by the paper *"Your LLM Knows the Future"*, this implementation brings cutting-edge non-autoregressive decoding research into production with:

- **Triton-accelerated kernels** for stable soft cross-entropy at scale
- **Chunked Training with Streaming KV cache** for processing ultra-long sequences (100k+ tokens)
- **Length-grouped batching** for better GPU utilization
- **Production-ready** checkpoint management and distributed training

> **Status:** Research-purpose code under active development. Interfaces may evolve. Contributions, issues, and PRs are welcome!

---

## Key Features

### Core Architecture
- **MTP Training Pipeline** – Complete end-to-end training with `mtp_glora/train.py` (single/multi-GPU via DDP)
- **Gated LoRA Injection** – Selective adaptation of LoRA with optional fused QKV and Gate-Up projections (Llama, Qwen3)
- **Flex Attention** – Custom block masks for complex MTP attention patterns with FlashAttention-level performance

### Performance Optimizations
- **Triton-Accelerated Loss** – Liger-kernel based stable soft cross-entropy for large vocabularies (152k+ tokens) without materializing probabilities
- **Chunked Training with Streaming KV Cache** – Memory-efficient processing of sequences up to 100k+ tokens through chunked computation
- **Length-Grouped Batching** – Dynamic batching by sequence length for better GPU utilization across ranks

### Developer Experience
- **Automatic Dataset Caching** – Intelligent caching with versioning and sharded writes for large corpora
- **Checkpoint Management** – Automatic rotation, resumable training, and distributed-safe I/O
- **Data Preparation Tools** – Complete pipeline with prompt extraction and vLLM generation (`prepare_data/`)

---

## Installation

### Prerequisites

- **Python 3.11+** with `pip`
- **PyTorch 2.8+** (for Flex Attention and modern kernels)
- **CUDA 12.8+** (for PyTorch 2.8)
- **Linux** recommended for NCCL distributed training

### Core Training Environment

```bash
# Clone the repository
git clone https://github.com/siihwanpark/MTP-GLoRA.git
cd MTP-GLoRA

# Install core dependencies
pip install -r requirements.txt
```

### Optional: Data Preparation Tools

For prompt extraction and vLLM-based response generation:

```bash
pip install -r prepare_data/requirements.txt
```

> **Note:** Triton is required for the custom loss kernel and is typically bundled with PyTorch. If not, install with `pip install triton`.

---

## Quick Start

Get training in 4 simple steps:

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Training Data

Extract prompts from existing datasets:

```bash
# OpenThoughts-114k dataset
bash scripts/extract_prompts.sh openthoughts

# Or AM-Qwen3-Distilled dataset
bash scripts/extract_prompts.sh am_qwen3
```

### Step 3: Generate Model Responses

Use vLLM for efficient inference:

```bash
bash scripts/run_vllm.sh am_qwen3
```

### Step 4: Train MTP Model

Launch distributed training:

```bash
torchrun --standalone --nproc_per_node=8 mtp_glora.train \
  --model_path Qwen/Qwen3-8B \
  --train_data_path data/am_qwen3_vllm_output.json \
  --save_dir checkpoints/mtp_experiment \
  --report_to tensorboard \
  --fuse_weights \
  --lr 2e-4 --warmup_steps 5000 --max_steps 50000 \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --chunk_size 4096 --min_chunk_size 1024 \
  --per_device_batch_size 1
```

<details>
<summary>📊 What to expect during training</summary>

- **First few steps**: Triton kernel autotuning (slower)
- **After warmup**: Stable training speed
- **TensorBoard**: Logs in `checkpoints/mtp_experiment/tensorboard/`
- **Checkpoints**: Auto-saved every 1000 steps with rotation

</details>

---

## Repository Structure

```
MTP-GLoRA/
├── mtp_glora/
│   ├── core/              # MTP model wrapper and Triton loss kernels
│   ├── data_utils/        # Dataset builder, chunked collator, intelligent caching
│   ├── models/            # Llama/Qwen3 adapters, Gated LoRA layers
│   ├── trainer/           # Training loop, checkpoints, distributed coordination
│   └── utils/             # Distributed helpers, logging, statistics
├── prepare_data/          # Data preparation pipeline
│   ├── extract_prompts.py # Extract prompts from datasets
│   └── run_vllm.py        # Generate responses with vLLM
├── scripts/               # Ready-to-use shell scripts
└── data/                  # Your datasets and outputs (git-ignored)
```

---

## Data Preparation

Prepare MTP training pairs with the provided helpers:

### Prompt Extraction

Extract prompts from popular datasets:

```bash
# OpenThoughts-114k dataset
bash scripts/extract_prompts.sh openthoughts

# AM-Qwen3-Distilled dataset
bash scripts/extract_prompts.sh am_qwen3
```

### Response Generation

Generate model responses using vLLM for efficient inference:

```bash
bash scripts/run_vllm.sh am_qwen3
```

### Expected Data Format

The dataset builder (`mtp_glora/data_utils/dataset.py`) expects JSON with `input`/`output` pairs:

**Option 1: Simple list**
```json
[
  {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
  {"input": "Explain quantum computing", "output": "Quantum computing uses..."}
]
```

**Option 2: Nested structure**
```json
{
  "results": [
    {"input": "prompt text", "output": "model response"},
    ...
  ]
}
```

---

## Training

### Basic Multi-GPU Training

Launch distributed training on a single node (see `scripts/train.sh` for reference):

```bash
torchrun --standalone --nproc_per_node=8 mtp_glora.train \
  --model_path meta-llama/Llama-3.1-8B-Instruct \
  --train_data_path data/training_data.json \
  --save_dir checkpoints/llama_mtp \
  --report_to tensorboard \
  --fuse_weights \
  --lr 2e-4 --warmup_steps 5000 --max_steps 50000 \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --chunk_size 4096 --min_chunk_size 1024 \
  --per_device_batch_size 1
```

### Resume from Checkpoint

Seamlessly resume training from any checkpoint:

```bash
torchrun --standalone --nproc_per_node=8 mtp_glora.train \
  --model_path meta-llama/Llama-3.1-8B-Instruct \
  --train_data_path data/training_data.json \
  --resume --checkpoint_dir checkpoints/llama_mtp
```

> The trainer automatically loads the latest checkpoint (via `latest.json`) and restores the optimizer, scheduler, and RNG states for exact continuation.

---

## Configuration Reference

<details>
<summary><b>Click to expand full configuration options</b></summary>

### Model Configuration
| Parameter | Description | Example |
|-----------|-------------|---------|
| `--model_path` | HuggingFace model ID or local path | `meta-llama/Llama-3.1-8B` |
| `--cache_dir` | Model cache directory | `~/.cache/huggingface` |
| `--dtype` | Training precision | `bfloat16` (default), `float16`, `float32` |
| `--fuse_weights` | Fuse QKV/Gate-Up projections | `--fuse_weights` |

### Data Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train_data_path` | Path to training JSON file | Required |
| `--eval_data_path` | Path to evaluation JSON file | `None` |
| `--dataset_cache_dir` | Dataset cache location | Auto |
| `--dataset_cache_rebuild` | Force rebuild cache | `False` |
| `--num_workers` | DataLoader workers | `4` |
| `--group_by_length` | Length-grouped batching | `True` |

### MTP Configuration
| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--draft_length` | Number of draft tokens | `4` |
| `--chunk_size` | Chunk size for memory efficiency | `2048` - `4096` |
| `--min_chunk_size` | Minimum chunk size | `1024` |

### LoRA Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lora_rank` | LoRA rank | `16` |
| `--lora_alpha` | LoRA alpha (scaling) | `32` |
| `--lora_dropout` | LoRA dropout rate | `0.05` |
| `--lora_use_rslora` | Use RS-LoRA scaling | `False` |
| `--lora_modules` | Target modules | `q_proj,k_proj,v_proj,o_proj,...` |

### Training Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max_steps` | Total training steps | `50000` |
| `--warmup_steps` | Warmup steps | `5000` |
| `--lr` | Learning rate | `2e-4` |
| `--per_device_batch_size` | Batch size per device | `1` |
| `--grad_accumulation_steps` | Gradient accumulation | `1` |
| `--max_grad_norm` | Gradient clipping | `1.0` |
| `--save_steps` | Checkpoint interval | `1000` |
| `--save_limit` | Max checkpoints to keep | `3` |

### Logging Configuration
| Parameter | Description | Options |
|-----------|-------------|---------|
| `--report_to` | Logging backend | `wandb`, `tensorboard`, `none` |
| `--wandb_project` | W&B project name | Your project |
| `--wandb_name` | W&B run name | Auto-generated |

</details>

---

## Architecture Overview

### System Components

```
┌──────────────────────────────────────────────────────────────┐
│                      MTP Training Pipeline                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input  →   Tokenizer   →   Collator   →   Model   →   Loss  │
│                  ↓             ↓            ↓                │
│               <mask>        Chunking      GatedLoRA          │
│               token        + KV Cache     Layers             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Function | Key Features |
|-----------|----------|--------------|
| **Tokenizer** | Token management | Deterministic `<mask>` embedding initialization |
| **Collator** | Data preprocessing | `MTPChunkedDataCollator` - mask insertion & chunking |
| **Sampler** | Batch formation | `DistributedLengthGroupedBatchSampler` - length-grouped global batching |
| **Attention** | Attention mechanism | Flex Attention with custom `BlockMask` for MTP rules |
| **Model** | Neural network | Llama/Qwen3 + GatedLoRA layers + optional fused projections |
| **Sampler Head** | Draft prediction | 2-layer MLP with residual connections |
| **Loss** | Training objective | Triton `StableSoftCrossEntropy` - numerically stable, memory-efficient |
| **Trainer** | Training orchestration | Chunked forward/backward, DDP sync, checkpoint management |

---

## Detailed Features

### Chunked Training: Handling Ultra-Long Sequences

**The Challenge:** MTP inserts several mask tokens between output tokens, creating sequences of 10k-100k+ tokens — impractical for standard GPU memory.

**Our Solution:** Memory-efficient chunked training with four key techniques:

#### 1️⃣ Chunking with Gradient Accumulation
Sequences are split into fixed-size chunks by `MTPChunkedDataCollator`. The trainer processes chunks sequentially, computing loss and gradients per chunk, then accumulates before optimizer step.

#### 2️⃣ Streaming KV Cache Across Chunks
Only `<mask>` tokens are learned; regular tokens provide teacher context. We preserve KV cache for regular tokens and reuse across chunks:
- `StreamingKVCacheManager.prepare_data_with_kv_cache` builds block masks
- `extract_regular_kv_cache_for_next_chunk` propagates only regular-token KV

#### 3️⃣ Smart DDP Synchronization
Gradients sync only on the last valid chunk of the last micro-batch via `sync_on_last_step`, avoiding excessive communication when devices see different chunk counts.

#### 4️⃣ Length-Grouped Global Batching
`DistributedLengthGroupedBatchSampler` groups sequences by length before sharding across ranks, reducing stragglers and improving GPU utilization.

### StableSoftCrossEntropyLoss: Triton-Accelerated Training

Custom loss kernel (`mtp_glora/core/loss.py`) for efficient, numerically stable cross-entropy with large vocabularies (152k+ tokens).

**Key Innovations:**

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Logits-Space Computation** | $CE = -S/dt + mx + log(dx)$ with LSE statistics | No probability materialization |
| **Memory Efficiency** | Per-row scalars only (mx, dx, mt, dt) | Massive memory savings vs softmax |
| **Triton Kernels** | Autotuned tile sizes (32K/64K), coalesced loads | 3-5x faster than naive PyTorch |
| **Stable Backward** | Gradient: softmax(x-mx) - softmax(t-mt) | Numerical stability at fp16/bf16 |
| **Verified Correctness** | Unit tests vs naive CE & PyTorch reference | Tested up to V≈152k |

**Usage in MTP:** Applied per draft step in `mtp_glora/core/mtp.py`:
```python
loss = StableSoftCrossEntropy.apply(sampler_logits, teacher_logits_step)
```

### GatedLoRA: Selective Adaptation

Selective parameter-efficient fine-tuning that activates LoRA updates only for specific tokens.

#### GatedLoRALinear Layer

Wraps `nn.Linear` with LoRA A/B matrices, scaled by `gate_mask` for selective activation:

```python
output = base_linear(x) + lora_scaling * lora_B(lora_A(x)) * gate_mask
```

**Features:**
- ✅ **Selective Updates** – LoRA activates only at `<mask>` positions
- ✅ **Fusion-Aware** – Auto-scales for fused projections (×3 for QKV, ×2 for Gate-Up)
- ✅ **Efficient Training** – Base weights frozen, only LoRA params trainable
- ✅ **Flexible Scaling** – Standard (`α/r`) or RS-LoRA (`α/√r`) scaling

#### SamplerHead: Draft Token Prediction

Lightweight 2-layer MLP predicting draft tokens from teacher features:

```
Input: concat(prev_token_embedding, teacher_hidden_state)
  ↓
Linear → SiLU → RMSNorm
  ↓
Linear → SiLU → RMSNorm (+ residual)
  ↓
Output: draft_hidden_state → logits
```

**End-to-End Flow:**
1. Base model generates teacher hidden states
2. SamplerHead predicts draft tokens from teacher features
3. GatedLoRA selectively adapts only at `<mask>` positions
4. StableSoftCE trains sampler to match teacher predictions

### Flex Attention: Custom MTP Masking

PyTorch Flex Attention provides FlashAttention-level speed with custom masking logic impossible in standard FlashAttention.

**Why Flex Attention?**
- High-performance attention with user-defined `BlockMask`
- Necessary for complex MTP gating rules between regular (`x`) and mask (`m`) tokens
- Supports KV cache streaming across chunks

**MTP Attention Rules:**
```
x (regular) queries: attend only to x tokens (teacher context)
m (mask) queries:    attend to x tokens + m tokens in same block (draft prediction)
```

**Implementation:**
`StreamingKVCacheManager._create_block_mask_for_chunk` builds masks enforcing:
- ✅ Causal ordering
- ✅ Padding token filtering  
- ✅ MTP-specific visibility rules

### Deterministic `<mask>` Token Initialization

Reproducible mask token embeddings without external files.

**Initialization Strategy:**
```python
# When adding <mask> token to vocabulary
new_embedding = mean(existing_embeddings)  # Deterministic & stable
```

**Benefits:**
- **Reproducible** – Same initialization across all runs
- **Self-Contained** – No separate embedding files needed
- **Inference-Ready** – Works identically at inference time

**Implementation:** `LlamaForCausalLM._init_added_embeddings_weights_with_mean` in `mtp_glora/models/modeling_llama.py`

---

## Checkpoint Management

### Automatic Checkpointing

**Directory Structure:**
```
save_dir/
├── step-1000/
│   ├── model.safetensors    # Trainable parameters only
│   ├── state.pt             # Optimizer, scheduler, RNG states
│   └── config.json          # Training configuration
├── step-2000/
├── step-3000/
└── latest.json              # Points to most recent checkpoint
```

**Features:**
- ✅ **Automatic Rotation** – Keeps only last N checkpoints (`--save_limit`)
- ✅ **Exact Resumption** – Restores optimizer, scheduler, RNG, and sampler state
- ✅ **Distributed-Safe** – Rank 0 writes, all ranks wait at barriers
- ✅ **SafeTensors Format** – Fast, secure model weight serialization

### Intelligent Dataset Caching

MTP dataset building is expensive — cache once, reuse everywhere.

**Cache Strategy:**
```
<train_file_folder>/.mtp_cache/mtp_ds_<hash>/
├── shard_0000.arrow    # ~4 GiB per shard
├── shard_0001.arrow
├── shards.json         # Shard manifest
├── meta.json           # Dataset metadata
└── _SUCCESS            # Completion marker
```

**Smart Invalidation:**
Cache hash includes:
- Training file fingerprint (path, mtime, size)
- Tokenizer fingerprint (vocab size, `<mask>` ID, added tokens)
- Build parameters (`draft_length`, `shuffle_seed`)
- Build version (for code changes)

**Distributed Building:**
1. Rank 0 builds and writes cache
2. Other ranks wait at barrier
3. All ranks load from shared cache
4. Force rebuild with `--dataset_cache_rebuild`

**Cached Data:**
- `input_ids`, `position_ids`, `gate_mask`, `regular_token_mask`, `total_len`
- Arrow format → Torch format at load time

---

## Performance Tips

### Optimization Checklist

| Optimization | Command/Setting | Impact |
|--------------|-----------------|--------|
| **Fused Projections** | `--fuse_weights` | 15-20% speedup (Llama/Qwen3) |
| **Gradient Accumulation** | `--grad_accumulation_steps=4` | Larger effective batch size |
| **Length Grouping** | Enabled by default | 20-30% reduction in padding |
| **Triton Warmup** | First 5-10 steps | Auto-tuning overhead |
| **BF16 Precision** | `--dtype=bfloat16` | Best stability/speed balance |
| **Pin Memory** | `--pin_memory` (default) | Faster host-device transfers |

### Expected Performance

**Reference Setup:** 8×H100 (80GB), Qwen3-8B, draft_length=4, chunk_size=5120
- **Training Speed**: ~70 hours for 30000 steps
- **Memory**: ~80 GB per GPU
- **First step**: Slower due to Triton autotuning

---

## Known Limitations

### Current Constraints

- **Batch Size**: Collator optimized for `per_device_batch_size=1`. Use `--grad_accumulation_steps` to scale.
- **PyTorch Version**: Requires PyTorch 2.8+ for Flex Attention. Eager attention fallback is slower.
- **Platform**: NCCL multi-GPU recommended on Linux. Windows NCCL support is limited.
- **Model Support**: Pre-built adapters for Llama and Qwen3. Other architectures need custom adaptation.

### Roadmap Items

- [ ] Multi-sample batch collation
- [ ] Additional model architectures (Mistral, Phi, etc.)
- [ ] Inference pipeline for trained models
- [ ] Mixed-precision training optimizations

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This project builds upon groundbreaking research and excellent open-source tools:

### Research Foundations
- [**"Your LLM Knows the Future"**](https://arxiv.org/abs/2507.11851) – Core MTP methodology inspiration
- Prior work on **speculative decoding** and **multi-token prediction**

### Technology Stack
- [**PyTorch**](https://pytorch.org/) – Deep learning framework with Flex Attention
- [**Hugging Face Transformers**](https://huggingface.co/transformers/) – Model architectures and utilities
- [**Triton**](https://github.com/openai/triton) – GPU kernel programming
- [**vLLM**](https://github.com/vllm-project/vllm) – High-performance inference engine
- [**SpecForge**](https://github.com/sgl-project/SpecForge) - Training speculative decoding models

Special thanks to the open-source community for tools, kernels, and inspiration that made this work possible.

---

## Contributing

We welcome contributions from the community! Here's how to get involved:

### Contribution Guidelines

1. **Discuss First** – Open an issue for major changes before submitting a PR
2. **Stay Focused** – Keep PRs targeted and well-documented
3. **Benchmark Changes** – Include performance metrics for kernel/training loop changes
4. **Code Quality** – Ensure formatting and lints pass
5. **Documentation** – Update docs for user-facing changes

### Areas for Contribution

- Model architecture adapters (Mistral, Phi, Gemma, etc.)
- Performance optimizations
- Extended testing and validation
- Documentation improvements
- Bug fixes and issue resolution

---

## Citation

If you use MTP-GLoRA in your research or project, please cite:

```bibtex
@software{mtp_glora_2025,
  title        = {MTP-GLoRA: Training Framework for Multi-Token Prediction with Gated LoRA},
  author       = {Park, Sihwan and contributors},
  year         = {2025},
  url          = {https://github.com/siihwanpark/MTP-GLoRA},
  note         = {Efficient training framework for multi-token prediction}
}
```

---

<div align="center">

**Made with ❤️ by the open-source community**

[⭐ Star us on GitHub](https://github.com/siihwanpark/MTP-GLoRA) • [🐛 Report Bug](https://github.com/siihwanpark/MTP-GLoRA/issues) • [💡 Request Feature](https://github.com/siihwanpark/MTP-GLoRA/issues)

</div>