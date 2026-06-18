<div align="center">

# 🧠 OmniLiquid AGI

### A Continuous-Time Neural ODE Architecture for Language Modeling

*Replacing discrete Transformer attention with biologically-inspired continuous dynamical systems*

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: AGPL 3.0](https://img.shields.io/badge/License-AGPL%203.0-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Liquid%20Neural%20ODE-purple.svg)](#architecture)

---

**OmniLiquid AGI** is a from-scratch implementation of a **Liquid Time-Constant Neural Network** applied to autoregressive language modeling. Unlike GPT, Llama, or Claude which use discrete self-attention layers, OmniLiquid models cognition as a **continuous dynamical system** governed by ordinary differential equations (ODEs) — inspired by the neural circuits of biological organisms.

</div>

---

## ⚠️ Experimental Status

> [!CAUTION]
> **This project is highly experimental.** The architecture has been validated and proven to converge at the `nano` scale (1.7M parameters, 24M tokens). Larger configurations (`small` through `ultra`) have **not been trained** due to the extreme computational requirements of sequential ODE integration and the lack of available hardware (multi-GPU clusters / supercomputer access). The scaling configurations represent theoretical architectural definitions based on the validated ODE physics.

> [!IMPORTANT]
> **Hardware Reality:** Training the `ultra` (70B) configuration would require approximately 16,384 A100 GPUs running for several months. The `nano` (1.7M) configuration trains in ~90 minutes on a single Kaggle T4 GPU and serves as the proof-of-concept that the underlying mathematics work.

---

## 🔬 Architecture

### Core Innovation: Liquid Time-Constant ODEs

Traditional Transformers process tokens through discrete attention layers. OmniLiquid replaces this with a **continuous-time state evolution**:

```
d(state)/dt = (-state + σ(W_inp · x + b)) / τ(x)

where:
  τ(x) = σ(W_tau · x + b) × 10 + 0.1    (learnable time constant)
  σ = sigmoid activation
```

Each layer maintains a **persistent hidden state** that evolves smoothly through time as new tokens are processed. The learnable time constant `τ(x)` allows the network to adaptively control how quickly it integrates new information — fast for syntax, slow for long-range semantics.

### Why This Matters

| Property | Transformer (GPT/Llama) | OmniLiquid (This Project) |
|---|---|---|
| **Context Memory** | KV Cache grows linearly with context | Fixed-size state vector (constant memory) |
| **Theoretical Context** | Limited by KV cache size | Theoretically infinite |
| **Inference Memory** | O(n) where n = context length | O(1) — constant regardless of input |
| **Training Speed** | Highly parallel (fast) | Sequential (slow) |
| **Precision Required** | FP16/BF16 safe | FP32 required (gradient stability) |
| **Biological Inspiration** | None (pure attention) | Neural circuit dynamics |

### Scaling Configurations

```
┌─────────────┬──────────┬────────┬────────────────┬──────────────────────────┐
│ Config      │ dim      │ layers │ Parameters     │ Training Requirement     │
├─────────────┼──────────┼────────┼────────────────┼──────────────────────────┤
│ nano        │ 32       │ 4      │ ~1.7M          │ 1x T4 GPU (90 min)      │
│ micro       │ 512      │ 16     │ ~35M           │ 1x A100 GPU (60 hrs)    │
│ small       │ 2,048    │ 32     │ ~500M          │ 8x A100 cluster         │
│ medium      │ 4,096    │ 48     │ ~3.5B          │ 64x A100 cluster        │
│ large       │ 8,192    │ 64     │ ~14B           │ 256x A100 cluster       │
│ xl          │ 12,288   │ 80     │ ~35B           │ 2048x A100 cluster      │
│ ultra       │ 16,384   │ 96     │ ~70B           │ 16384x A100 cluster     │
└─────────────┴──────────┴────────┴────────────────┴──────────────────────────┘
```

---

## 🧪 Validated Results (Nano Scale)

The `nano` configuration has been fully trained and validated:

| Metric | Value |
|---|---|
| **Parameters** | 1,616,928 (~1.7M) |
| **Training Tokens** | 24,000,000 (24M) |
| **Final Training Loss** | ~6.84 (converged) |
| **Training Time** | ~90 minutes |
| **Hardware** | 1x NVIDIA T4 (Kaggle) |
| **Precision** | Pure FP32 |
| **Speed** | 2.35 steps/second |

### Training Loss Curve

The loss successfully converged from `10.82` (random initialization) to `6.84`, proving the ODE dynamics are mathematically stable and the architecture learns:

```
Step   10/5860 | Loss: 10.8258    ← Random initialization
Step  100/5860 | Loss:  9.9264    ← Learning basic token statistics
Step  300/5860 | Loss:  6.9698    ← Rapid convergence phase
Step  500/5860 | Loss:  6.9196    ← Entering saturation
Step 1000/5860 | Loss:  6.85      ← Converged
Step 5860/5860 | Loss:  6.84      ← Fully saturated
```

### Dataset Composition

```
┌──────────────┬─────────────────┬──────┬──────────────────────────┐
│ Category     │ Tokens          │  %   │ Source                   │
├──────────────┼─────────────────┼──────┼──────────────────────────┤
│ Reasoning    │ 12,000,000      │ 50%  │ MetaMathQA               │
│ English      │  8,400,000      │ 35%  │ OpenOrca                 │
│ Code         │  3,600,000      │ 15%  │ CodeAlpaca-20k           │
├──────────────┼─────────────────┼──────┼──────────────────────────┤
│ TOTAL        │ 24,000,000      │ 100% │                          │
└──────────────┴─────────────────┴──────┴──────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Bshdhorrhh/AGI-.git
cd AGI-
pip install -r requirements.txt
```

### Train the Nano Model (Free on Kaggle)

```bash
# Train from scratch (~90 minutes on T4)
python train.py --config nano

# Resume from checkpoint
python train.py --config nano --resume omniliquid_nano_step500.pth
```

### Chat with the Model

```bash
# Interactive chat
python chat.py --config nano

# With custom sampling parameters
python chat.py --config nano --temperature 0.7 --top_k 50 --max_tokens 150
```

### Verify Architecture

```bash
# Print all scaling configurations and run forward pass test
python model.py
```

---

## 📁 Repository Structure

```
AGI-/
├── model.py              # Core OmniLiquid architecture (nano → 70B)
├── train.py              # Training pipeline with auto-configuration
├── chat.py               # Interactive chat interface
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🧬 Technical Deep Dive

### The ODE Integration Loop

The heart of OmniLiquid is a JIT-compiled (`@torch.jit.script`) ODE integration loop that processes tokens sequentially through continuous-time dynamics:

```python
for t in range(seq_len):
    x = embedded_tokens[:, t, :]          # Current input
    
    for layer in range(n_layers):
        f = sigmoid(W_inp @ x + b)        # Input gate
        τ = sigmoid(W_tau @ x + b) * 10   # Time constant
        state += (-state + f) / τ          # Euler integration
        x = LayerNorm(state)               # Normalize
```

### Why FP32 is Required

The sequential state accumulation in ODE integration causes tiny floating-point errors to compound over hundreds of timesteps. In FP16, these errors snowball exponentially and cause gradient explosion (observed at ~Step 1500 in our experiments). FP32 provides sufficient precision for stable convergence.

### Weight Tying

The output projection head shares weights with the input embedding layer (`head.weight = emb.weight`), reducing the parameter count significantly while maintaining a consistent token representation space.

---

## 🔮 Future Directions

- [ ] **Adaptive ODE Solvers**: Replace Euler integration with higher-order methods (RK4, Dormand-Prince) for more accurate state trajectories
- [ ] **Chunked Parallel Training**: Process multiple independent chunks in parallel while maintaining sequential ODE within each chunk
- [ ] **Mixture of Liquid Experts (MoLE)**: Route prompts to specialized Liquid ODE expert models using a gating network
- [ ] **External Scratchpad Memory**: Integrate RAG-style retrieval to extend effective context beyond the state vector capacity
- [ ] **Scale Validation**: Train and evaluate `micro` (35M) and `small` (500M) configurations with sufficient compute

---

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **[Liquid AI / MIT CSAIL](https://www.liquid.ai/)** — Pioneering research on Liquid Time-Constant Networks
- **[Meta AI](https://ai.meta.com/)** — MetaMathQA dataset
- **[Open-Orca](https://huggingface.co/Open-Orca)** — OpenOrca instruction dataset  
- **[Databricks](https://www.databricks.com/)** — Dolly-15k human-written dataset

---

<div align="center">

*Built from scratch with ❤️ and pure mathematics*

**OmniLiquid AGI** — *Where cognition flows like water*

</div>
