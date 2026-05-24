"""
training/export_model.py

Standalone model export and quantization script for Trace.

Exports trained models in multiple formats using Unsloth's optimized
save methods:
  - Merged 16-bit (full precision safetensors)
  - Merged 4-bit (quantized safetensors)
  - GGUF (Q4_K_M, Q8_0) for Ollama / llama.cpp deployment
  - LoRA-only (adapter weights only)
  - HuggingFace Hub push

Usage:
    # Export from GRPO checkpoint:
    python -m training.export_model --input ./outputs/trace-grpo

    # Export with specific formats:
    python -m training.export_model --input ./outputs/trace-grpo \\
        --format merged_16bit gguf_q4_k_m gguf_q8_0

    # Push to HuggingFace Hub:
    python -m training.export_model --input ./outputs/trace-grpo \\
        --push-to-hub --hub-repo username/trace-model

    # Dry run (just verify model loads):
    python -m training.export_model --input ./outputs/trace-grpo --dry-run
"""

import argparse
import os
import sys
import time
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Trace Model Export & Quantization")
    parser.add_argument("--input", required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", default=None,
                        help="Output base path (default: <input>-exported)")
    parser.add_argument("--format", nargs="+",
                        default=["merged_16bit", "lora", "gguf_q4_k_m"],
                        choices=[
                            "merged_16bit", "merged_4bit",
                            "gguf_q4_k_m", "gguf_q8_0", "gguf_q5_k_m",
                            "gguf_f16",
                            "lora",
                        ],
                        help="Export formats")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default="",
                        help="HuggingFace Hub repository name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just load the model, don't export")
    return parser.parse_args()


def main():
    args = parse_args()

    output_base = args.output or f"{args.input}-exported"

    print("=" * 60)
    print("[Trace] Model Export & Quantization (Unsloth)")
    print("=" * 60)
    print(f"  Input:   {args.input}")
    print(f"  Output:  {output_base}")
    print(f"  Formats: {args.format}")
    print("=" * 60)

    # ── Load model with Unsloth ──────────────────────────────────────────
    from unsloth import FastLanguageModel

    print(f"[Trace] Loading model from {args.input}...")
    start = time.time()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.input,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    elapsed = time.time() - start
    print(f"[Trace] Model loaded in {elapsed:.1f}s")

    if args.dry_run:
        print("[Trace] Dry run — model loaded successfully, skipping export.")
        _print_model_info(model)
        return

    # ── Export in each requested format ───────────────────────────────────
    os.makedirs(output_base, exist_ok=True)

    for fmt in args.format:
        print(f"\n[Trace] Exporting: {fmt}")
        start = time.time()

        try:
            if fmt == "merged_16bit":
                path = os.path.join(output_base, "merged-16bit")
                model.save_pretrained_merged(
                    path, tokenizer,
                    save_method="merged_16bit",
                )
                print(f"  ✅ Saved to {path}")

            elif fmt == "merged_4bit":
                path = os.path.join(output_base, "merged-4bit")
                model.save_pretrained_merged(
                    path, tokenizer,
                    save_method="merged_4bit_forced",
                )
                print(f"  ✅ Saved to {path}")

            elif fmt.startswith("gguf_"):
                quant = fmt.replace("gguf_", "")
                path = os.path.join(output_base, f"gguf-{quant}")
                model.save_pretrained_gguf(
                    path, tokenizer,
                    quantization_method=quant,
                )
                print(f"  ✅ Saved GGUF ({quant}) to {path}")
                _print_gguf_usage(path, quant)

            elif fmt == "lora":
                path = os.path.join(output_base, "lora-adapters")
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                print(f"  ✅ Saved LoRA adapters to {path}")

        except Exception as e:
            print(f"  ❌ Export failed for {fmt}: {e}")
            if "gguf" in fmt:
                print(f"     Install: pip install llama-cpp-python")

        elapsed = time.time() - start
        print(f"  ⏱  Took {elapsed:.1f}s")

    # ── Push to Hub ──────────────────────────────────────────────────────
    if args.push_to_hub:
        if not args.hub_repo:
            print("\n[Trace] ERROR: --hub-repo required with --push-to-hub")
            return

        print(f"\n[Trace] Pushing to HuggingFace Hub: {args.hub_repo}")
        try:
            # Push merged 16-bit
            model.push_to_hub_merged(
                args.hub_repo, tokenizer,
                save_method="merged_16bit",
            )
            print(f"  ✅ Pushed to {args.hub_repo}")
        except Exception as e:
            print(f"  ❌ Push failed: {e}")
            print(f"     Set HF_TOKEN environment variable")

        # Push GGUF if requested
        if any("gguf" in f for f in args.format):
            try:
                quant = [f.replace("gguf_", "") for f in args.format if "gguf" in f][0]
                model.push_to_hub_gguf(
                    args.hub_repo, tokenizer,
                    quantization_method=quant,
                )
                print(f"  ✅ Pushed GGUF ({quant}) to {args.hub_repo}")
            except Exception as e:
                print(f"  ❌ GGUF push failed: {e}")

    print(f"\n[Trace] Export complete! All outputs in: {output_base}")


def _print_model_info(model):
    """Print model parameter information."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters:")
    print(f"    Total:     {total:,}")
    print(f"    Trainable: {trainable:,} ({100*trainable/total:.1f}%)")


def _print_gguf_usage(path: str, quant: str):
    """Print how to use the exported GGUF model."""
    # Find the .gguf file
    gguf_files = [f for f in os.listdir(path) if f.endswith(".gguf")] if os.path.isdir(path) else []
    gguf_file = gguf_files[0] if gguf_files else f"model-{quant}.gguf"

    print(f"\n  📋 Usage with Ollama:")
    print(f"     1. Create a Modelfile:")
    print(f"        FROM {os.path.join(path, gguf_file)}")
    print(f"     2. Create the model:")
    print(f"        ollama create trace-model -f Modelfile")
    print(f"     3. Run:")
    print(f"        ollama run trace-model")

    print(f"\n  📋 Usage with llama.cpp:")
    print(f"     ./llama-cli -m {os.path.join(path, gguf_file)} -p \"<prompt>\"")


if __name__ == "__main__":
    main()
