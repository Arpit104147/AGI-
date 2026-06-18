"""
OmniLiquid AGI — Training Script
==================================
Multi-category dataset collection, packing, and training loop
for the OmniLiquid continuous-time neural ODE.

Supports:
  - Automatic dataset downloading from HuggingFace Hub
  - Multi-category data mixing (Reasoning / English / Code)
  - Packed sequence training with configurable context length
  - Cosine LR scheduling with linear warmup
  - Checkpoint saving and resumption
  - Pure FP32 training (required for ODE gradient stability)

Usage:
  python train.py                              # Train from scratch
  python train.py --resume checkpoint.pth      # Resume training
  python train.py --config nano                # Use a preset config
"""
import os
import sys
import glob
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from model import OmniLiquidAGI, MODEL_CONFIGS


# ═══════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="OmniLiquid AGI Training")
parser.add_argument("--config", type=str, default="nano",
                    choices=list(MODEL_CONFIGS.keys()),
                    help="Model size preset (default: nano)")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
parser.add_argument("--bs", type=int, default=None,
                    help="Batch size (auto-selected if not set)")
parser.add_argument("--sl", type=int, default=None,
                    help="Sequence length (auto-selected if not set)")
parser.add_argument("--lr", type=float, default=None,
                    help="Learning rate (auto-selected if not set)")
parser.add_argument("--steps", type=int, default=None,
                    help="Total training steps (auto-calculated if not set)")
args = parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# AUTO-CONFIGURATION BASED ON MODEL SIZE
# ═══════════════════════════════════════════════════════════════

# Optimal hyperparameters for each scale
SCALE_DEFAULTS = {
    "nano":   {"bs": 32,  "sl": 128, "lr": 5e-4,  "tokens": 24_000_000},
    "micro":  {"bs": 32,  "sl": 256, "lr": 1e-4,  "tokens": 400_000_000},
    "small":  {"bs": 16,  "sl": 512, "lr": 3e-4,  "tokens": 10_000_000_000},
    "medium": {"bs": 8,   "sl": 1024,"lr": 3e-4,  "tokens": 70_000_000_000},
    "large":  {"bs": 4,   "sl": 2048,"lr": 1.5e-4,"tokens": 280_000_000_000},
    "xl":     {"bs": 2,   "sl": 2048,"lr": 1e-4,  "tokens": 700_000_000_000},
    "ultra":  {"bs": 1,   "sl": 2048,"lr": 6e-5,  "tokens": 1_400_000_000_000},
}

defaults = SCALE_DEFAULTS[args.config]
BS = args.bs or defaults["bs"]
SL = args.sl or defaults["sl"]
LR = args.lr or defaults["lr"]
TARGET_TOKENS = defaults["tokens"]

ACCUM = 1
TOTAL_STEPS = args.steps or (TARGET_TOKENS // (BS * SL))
SAVE_EVERY = max(500, TOTAL_STEPS // 10)
WARMUP_STEPS = max(100, TOTAL_STEPS // 50)

# Data mix: 50% Reasoning, 35% English, 15% Code
TARGET_REASONING = int(TARGET_TOKENS * 0.50)
TARGET_ENGLISH   = int(TARGET_TOKENS * 0.35)
TARGET_CODE      = int(TARGET_TOKENS * 0.15)

PACKED_CACHE = f"packed_{args.config}.pt"


# ═══════════════════════════════════════════════════════════════
# DEVICE SETUP
# ═══════════════════════════════════════════════════════════════

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*65}")
print(f"  OmniLiquid AGI — Training Pipeline")
print(f"{'='*65}")
print(f"  Config:     {args.config} ({MODEL_CONFIGS[args.config]['desc']})")
print(f"  Device:     {device.upper()}")
print(f"  Tokens:     {TARGET_TOKENS:,}")
print(f"  BS={BS}  SL={SL}  LR={LR}  Steps={TOTAL_STEPS:,}")
print(f"{'='*65}\n")

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
EOS_ID = tokenizer.eos_token_id


# ═══════════════════════════════════════════════════════════════
# DATASET COLLECTION
# ═══════════════════════════════════════════════════════════════

def collect_tokens_from_dataset(dataset_iter, target, format_fn, name, allow_loop=False):
    """Stream dataset and collect tokens into a numpy buffer."""
    buf = np.zeros(target + 500_000, dtype=np.int32)
    pos = 0
    loops = 0
    
    while pos < target:
        for example in dataset_iter:
            text = format_fn(example)
            if text is None:
                continue
            encoded = tokenizer.encode(text)
            encoded.append(EOS_ID)
            n = len(encoded)
            if pos + n > len(buf):
                break
            buf[pos:pos + n] = encoded
            pos += n
            if pos % 5_000_000 < n:
                print(f"   [{name}] {pos:,}/{target:,} tokens ({pos/target*100:.1f}%)")
            if pos >= target:
                break
        
        loops += 1
        if not allow_loop or loops >= 35:
            break
        print(f"   [{name}] Looping dataset (pass {loops+1})...")
    
    print(f"   [{name}] DONE: {pos:,} tokens collected")
    return buf[:pos]


if os.path.exists(PACKED_CACHE):
    print(f">> Found cached data: {PACKED_CACHE}")
    packed_data = torch.load(PACKED_CACHE, weights_only=True)
    num_chunks = len(packed_data)
    print(f">> Loaded {num_chunks:,} chunks of {SL} tokens")
else:
    print(f">> Building {TARGET_TOKENS:,} token dataset from scratch...\n")
    
    # --- REASONING (50%) ---
    print(f">> [1/3] Collecting REASONING data ({TARGET_REASONING:,} tokens)...")
    try:
        math_ds = load_dataset("meta-math/MetaMathQA", split="train")
        print(f"   MetaMathQA loaded: {len(math_ds)} examples")
    except Exception as e:
        print(f"   MetaMathQA failed: {e}")
        math_ds = load_dataset("openai/gsm8k", "main", split="train")
    
    def format_reasoning(x):
        q = x.get("query", x.get("question", "")).strip()
        a = x.get("response", x.get("answer", "")).strip()
        if not q or not a: return None
        return f"### Instruction:\n{q}\n\n### Response:\n{a}"
    
    reasoning_tokens = collect_tokens_from_dataset(
        math_ds, TARGET_REASONING, format_reasoning, "REASONING", allow_loop=True
    )
    
    # --- ENGLISH (35%) ---
    print(f"\n>> [2/3] Collecting ENGLISH data ({TARGET_ENGLISH:,} tokens)...")
    try:
        orca_ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        def format_english(x):
            q = x.get("question", "").strip()
            a = x.get("response", "").strip()
            if not q or not a or len(a) < 20: return None
            return f"### Instruction:\n{q}\n\n### Response:\n{a}"
        english_tokens = collect_tokens_from_dataset(
            orca_ds, TARGET_ENGLISH, format_english, "ENGLISH"
        )
    except Exception as e:
        print(f"   OpenOrca failed: {e}")
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        def format_dolly(x):
            q = x.get("instruction", "").strip()
            ctx = x.get("context", "").strip()
            a = x.get("response", "").strip()
            if not q or not a: return None
            if ctx: return f"### Instruction:\n{q}\nContext: {ctx}\n\n### Response:\n{a}"
            return f"### Instruction:\n{q}\n\n### Response:\n{a}"
        english_tokens = collect_tokens_from_dataset(
            dolly, TARGET_ENGLISH, format_dolly, "ENGLISH", allow_loop=True
        )
    
    # --- CODE (15%) ---
    print(f"\n>> [3/3] Collecting CODE data ({TARGET_CODE:,} tokens)...")
    try:
        code_ds = load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True)
        def format_code(x):
            p = x.get("prompt", x.get("instruction", "")).strip()
            r = x.get("response", x.get("output", "")).strip()
            if not p or not r: return None
            return f"### Instruction:\n{p}\n\n### Response:\n{r}"
        code_tokens = collect_tokens_from_dataset(
            code_ds, TARGET_CODE, format_code, "CODE"
        )
    except Exception as e:
        print(f"   tiny-codes failed: {e}")
        code_ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        def format_code_fb(x):
            q = x.get("instruction", "").strip()
            a = x.get("output", "").strip()
            if not q or not a: return None
            return f"### Instruction:\n{q}\n\n### Response:\n{a}"
        code_tokens = collect_tokens_from_dataset(
            code_ds, TARGET_CODE, format_code_fb, "CODE", allow_loop=True
        )
    
    # --- COMBINE AND PACK ---
    print(f"\n>> Combining all categories...")
    total_tokens = len(reasoning_tokens) + len(english_tokens) + len(code_tokens)
    print(f"   Reasoning: {len(reasoning_tokens):,} tokens ({len(reasoning_tokens)/total_tokens*100:.0f}%)")
    print(f"   English:   {len(english_tokens):,} tokens ({len(english_tokens)/total_tokens*100:.0f}%)")
    print(f"   Code:      {len(code_tokens):,} tokens ({len(code_tokens)/total_tokens*100:.0f}%)")
    print(f"   TOTAL:     {total_tokens:,} tokens")
    
    print(">> Interleaving and packing...")
    all_tokens = np.concatenate([reasoning_tokens, english_tokens, code_tokens])
    num_chunks = len(all_tokens) // SL
    all_tokens = all_tokens[:num_chunks * SL]
    packed_data = torch.tensor(all_tokens, dtype=torch.long).reshape(num_chunks, SL)
    perm = torch.randperm(num_chunks)
    packed_data = packed_data[perm]
    
    print(f">> Packed into {num_chunks:,} chunks of {SL} tokens")
    print(f">> Saving cache to {PACKED_CACHE}...")
    torch.save(packed_data, PACKED_CACHE)
    print(f">> Cache saved ({os.path.getsize(PACKED_CACHE) / 1e9:.2f} GB)")
    
    del reasoning_tokens, english_tokens, code_tokens, all_tokens
    import gc; gc.collect()

num_chunks = len(packed_data)


def get_packed_batch(bs):
    indices = random.sample(range(num_chunks), min(bs, num_chunks))
    return packed_data[indices]


# ═══════════════════════════════════════════════════════════════
# MODEL INITIALIZATION
# ═══════════════════════════════════════════════════════════════

model = OmniLiquidAGI(config=args.config).to(device)
counts = model.count_parameters()
print(f"\n>> Model loaded: {counts['total']:,} parameters")
print(f">> Estimated VRAM: {model.estimate_memory_gb():.2f} GB")

if args.resume and os.path.exists(args.resume):
    print(f">> Resuming from: {args.resume}")
    model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# Pure FP32 — ODE requires 32-bit for gradient stability
print(f">> Pure FP32 training (ODE requires 32-bit stability)")


def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return LR * 0.5 * (1 + math.cos(math.pi * progress))


def manage_checkpoints(step_num):
    prefix = f"omniliquid_{args.config}"
    files = glob.glob(f"{prefix}_step*.pth")
    for f in files:
        try: os.remove(f)
        except: pass
    path = f"{prefix}_step{step_num}.pth"
    torch.save(model.state_dict(), path)
    print(f">> [SAVED] {path}")


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

tokens_per_step = BS * SL
total_train_tokens = TOTAL_STEPS * tokens_per_step

print(f"\n{'='*65}")
print(f"  >>> TRAINING: {args.config.upper()} | FP32 | Pure ODE")
print(f"  >>> Steps: {TOTAL_STEPS:,} | Tokens: {total_train_tokens:,}")
print(f"  >>> BS={BS} | SL={SL} | LR={LR} | Warmup={WARMUP_STEPS}")
print(f"  >>> Chunks: {num_chunks:,} | Coverage: {total_train_tokens/num_chunks/SL*100:.0f}%")
print(f"{'='*65}\n")

model.train()
optimizer.zero_grad()
t0 = time.time()
running_loss = 0.0
loss_count = 0

for step in range(1, TOTAL_STEPS + 1):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    tokens = get_packed_batch(BS).to(device)
    
    logits, _ = model(tokens=tokens)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    loss = criterion(shift_logits.view(-1, model.vocab), shift_labels.view(-1))

    (loss / ACCUM).backward()
    running_loss += loss.item()
    loss_count += 1

    if step % ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if step % 10 == 0:
        dt = time.time() - t0
        sps = 10 / dt
        eta = (TOTAL_STEPS - step) / max(sps, 0.01) / 60
        avg = running_loss / loss_count
        tps = sps * tokens_per_step
        print(f"[{args.config}] Step {step}/{TOTAL_STEPS} | "
              f"Loss: {avg:.4f} | LR: {lr:.2e} | "
              f"{sps:.2f} s/s | {tps:.0f} tok/s | ETA: {eta:.1f}m")
        t0 = time.time()
        running_loss = 0.0
        loss_count = 0

    if step % SAVE_EVERY == 0:
        manage_checkpoints(step)

manage_checkpoints(TOTAL_STEPS)
print(f"\n{'='*65}")
print(f"  >>> TRAINING COMPLETE!")
print(f"  >>> Model: omniliquid_{args.config}_step{TOTAL_STEPS}.pth")
print(f"  >>> Trained on ~{total_train_tokens:,} tokens")
print(f"{'='*65}")
