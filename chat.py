"""
OmniLiquid AGI — Interactive Chat Interface
=============================================
Loads a trained OmniLiquid checkpoint and provides an interactive
terminal chat with Temperature, Top-K sampling, and Repetition Penalty.

Usage:
  python chat.py                                    # Auto-detect checkpoint
  python chat.py --checkpoint omniliquid_nano.pth   # Specific checkpoint
  python chat.py --config micro                     # Specify model config
"""
import os
import glob
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import OmniLiquidAGI, MODEL_CONFIGS


parser = argparse.ArgumentParser(description="OmniLiquid AGI Chat")
parser.add_argument("--config", type=str, default="nano",
                    choices=list(MODEL_CONFIGS.keys()),
                    help="Model config preset (default: nano)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint file")
parser.add_argument("--temperature", type=float, default=0.8,
                    help="Sampling temperature (default: 0.8)")
parser.add_argument("--top_k", type=int, default=40,
                    help="Top-K sampling (default: 40)")
parser.add_argument("--max_tokens", type=int, default=200,
                    help="Max tokens to generate (default: 200)")
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print(f"Loading OmniLiquid [{args.config}] on {device.upper()}...")
model = OmniLiquidAGI(config=args.config).to(device)

# Find checkpoint
if args.checkpoint:
    ckpt_path = args.checkpoint
else:
    files = glob.glob(f"omniliquid_{args.config}_step*.pth")
    if not files:
        files = glob.glob("liquid_*step*.pth")
    if not files:
        print("ERROR: No trained weights found!")
        print("Run `python train.py` first, or specify --checkpoint")
        exit(1)
    ckpt_path = max(files, key=os.path.getctime)

print(f"Loading weights: {ckpt_path}")
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
model.eval()

params = model.count_parameters()
print(f"Parameters: {params['total']:,}")

print(f"""
╔═══════════════════════════════════════════╗
║     OmniLiquid AGI — Chat Interface       ║
║  Config: {args.config:<8}  Device: {device.upper():<6}         ║
║  Params: {params['total']:>12,}                ║
╠═══════════════════════════════════════════╣
║  Type 'exit' to quit                      ║
║  Type 'clear' to reset context            ║
╚═══════════════════════════════════════════╝
""")

while True:
    try:
        prompt = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break
        
    if not prompt:
        continue
    if prompt.lower() == 'exit':
        print("Goodbye!")
        break
    
    # Format in the exact training template
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    tokens = tokenizer.encode(formatted)
    input_tensor = torch.tensor([tokens]).to(device)
    
    print("AI: ", end="", flush=True)
    
    generated_tokens = []
    with torch.no_grad():
        for _ in range(args.max_tokens):
            logits, _ = model(input_tensor)
            next_logits = logits[0, -1, :] / args.temperature
            
            # Repetition penalty
            seen = set(input_tensor[0].tolist() + generated_tokens)
            for tid in seen:
                if next_logits[tid] > 0:
                    next_logits[tid] /= 1.5
                else:
                    next_logits[tid] *= 1.5
            
            # Top-K filtering
            v, _ = torch.topk(next_logits, args.top_k)
            next_logits[next_logits < v[-1]] = -float('Inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token)
            word = tokenizer.decode([next_token])
            print(word, end="", flush=True)
            
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token]]).to(device)], dim=1
            )
    
    print("\n")
