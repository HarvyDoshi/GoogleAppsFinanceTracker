"""
training/train_sft.py

Supervised Fine-Tuning (SFT) warm-start script for Trace.

Per hackathon guide: "In many practical cases, do a little SFT first, then RL"

SFT is more sample-efficient than RL and helps the model learn:
  - Output format (valid JSON with action_type, content, source)
  - Task scaffolding (PLAN → RETRIEVE → VERIFY → ANSWER sequence)
  - Domain vocabulary (financial terms, data source names)

This produces a warm-start checkpoint that train_grpo.py can build on.

Usage:
    # Generate SFT data first:
    python -m training.generate_sft_data --env-url http://localhost:8000

    # Run SFT training:
    python -m training.train_sft --config configs/grpo_config.yaml

    # Then continue with GRPO:
    python -m training.train_grpo --config configs/grpo_config.yaml \\
        --sft-checkpoint ./outputs/trace-sft
"""

import argparse
import json
import os
import sys
import yaml
import torch
from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Trace SFT Warm-Start Training")
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    parser.add_argument("--dataset", default=None,
                        help="Override SFT dataset path")
    parser.add_argument("--model", default=None,
                        help="Override model name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 2 steps then exit")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sft_cfg = cfg.get("sft", {})
    model_name = args.model or cfg.get("model_name", "unsloth/Qwen2.5-3B-Instruct")
    dataset_path = args.dataset or sft_cfg.get("dataset_path", "./data/sft_demos.jsonl")

    print("=" * 60)
    print("[Trace] SFT Warm-Start Training (Unsloth + TRL)")
    print("=" * 60)
    print(f"  Model:   {model_name}")
    print(f"  Dataset: {dataset_path}")
    print("=" * 60)

    # ── Check dataset exists ─────────────────────────────────────────────
    if not os.path.exists(dataset_path):
        print(f"[Trace] ERROR: SFT dataset not found at {dataset_path}")
        print(f"[Trace] Generate it first:")
        print(f"  python -m training.generate_sft_data --env-url http://localhost:8000")
        sys.exit(1)

    # ── Load model with Unsloth ──────────────────────────────────────────
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cfg.get("max_seq_length", 4096),
        load_in_4bit=cfg.get("load_in_4bit", True),
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    target_modules = cfg.get("lora_target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.get("lora_r", 32),
        target_modules=target_modules,
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load SFT dataset ─────────────────────────────────────────────────
    sft_data = []
    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                sft_data.append(json.loads(line))

    dataset = Dataset.from_list(sft_data)
    print(f"[Trace] SFT dataset loaded: {len(dataset)} examples")

    if len(dataset) == 0:
        print("[Trace] ERROR: Dataset is empty!")
        sys.exit(1)

    # ── Configure SFT Trainer ────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig

    output_dir = sft_cfg.get("checkpoint_path", "./outputs/trace-sft")

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1 if args.dry_run else sft_cfg.get("epochs", 2),
        per_device_train_batch_size=sft_cfg.get("batch_size", 4),
        gradient_accumulation_steps=sft_cfg.get("grad_accum", 2),
        learning_rate=sft_cfg.get("lr", 2e-4),
        max_seq_length=sft_cfg.get("max_seq_length", 2048),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        dataset_text_field="text",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )

    if args.dry_run:
        sft_config.max_steps = 2

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    # ── Train ────────────────────────────────────────────────────────────
    print("[Trace] Starting SFT training...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"[Trace] Saving SFT model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Also save merged version for direct use
    merged_path = f"{output_dir}-merged"
    print(f"[Trace] Saving merged 16-bit to {merged_path}")
    model.save_pretrained_merged(
        merged_path, tokenizer,
        save_method="merged_16bit",
    )

    print("[Trace] SFT training complete!")
    print(f"[Trace] Use this checkpoint for GRPO:")
    print(f"  python -m training.train_grpo --sft-checkpoint {output_dir}")


if __name__ == "__main__":
    main()
