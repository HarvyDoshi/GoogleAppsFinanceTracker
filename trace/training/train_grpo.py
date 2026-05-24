"""
training/train_grpo.py

Main RL training script for Trace using TRL (GRPO) + Unsloth.

This implements the full Unsloth + TRL + GRPO/RLVR pipeline per the
hackathon guide:
  - Unsloth FastLanguageModel for 2-4x faster rollouts
  - GRPO with multiple independent reward functions (anti-hacking)
  - Chat template formatting for instruct models
  - Optional SFT warm-start before RL
  - Curriculum learning: easy → medium → hard
  - Periodic output inspection (anti-reward-hacking)
  - Quantization export (GGUF, merged 16-bit, merged 4-bit)

Usage:
    # Direct GRPO training:
    python -m training.train_grpo --config configs/grpo_config.yaml

    # With SFT warm-start:
    python -m training.train_grpo --config configs/grpo_config.yaml --sft-first

    # From SFT checkpoint:
    python -m training.train_grpo --config configs/grpo_config.yaml \\
        --sft-checkpoint ./outputs/trace-sft

    # Dry run (2 steps, then exit):
    python -m training.train_grpo --config configs/grpo_config.yaml --dry-run

Requirements:
    pip install trl unsloth transformers datasets requests pyyaml
"""

import argparse
import json
import os
import sys
import time
import yaml
import requests
from typing import Optional

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import GRPOConfig, GRPOTrainer

from .dataset import TaskCurriculum, format_prompt_for_chat
from .callbacks import TraceRewardCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Trace GRPO Training (Unsloth + TRL)")
    parser.add_argument("--config", default="configs/grpo_config.yaml",
                        help="Path to GRPO config YAML")
    parser.add_argument("--env-url", default="http://localhost:8000",
                        help="URL of the running Trace OpenEnv server")
    parser.add_argument("--model", default=None,
                        help="Override model name from config")
    parser.add_argument("--sft-first", action="store_true",
                        help="Run SFT warm-start before GRPO")
    parser.add_argument("--sft-checkpoint", default=None,
                        help="Load from existing SFT checkpoint instead of base model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 2 training steps then exit (for testing)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip model export/quantization after training")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model loading (Unsloth) ─────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, cfg: dict):
    """
    Load model with Unsloth's FastLanguageModel.

    Per guide: "Unsloth reduces memory use and improves efficiency on top of TRL"
    """
    from unsloth import FastLanguageModel

    print(f"[Trace] Loading model with Unsloth: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cfg.get("max_seq_length", 4096),
        load_in_4bit=cfg.get("load_in_4bit", True),
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # Apply LoRA adapters
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
        use_gradient_checkpointing="unsloth",  # Unsloth optimized
        random_state=42,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Trace] Model loaded. LoRA r={cfg.get('lora_r', 32)}, "
          f"modules={target_modules}")

    return model, tokenizer


# ── Dataset / prompt builder ─────────────────────────────────────────────────

def build_dataset(
    curriculum: TaskCurriculum,
    env_url: str,
    tokenizer,
    n_tasks: int = 200,
) -> Dataset:
    """
    Generate a dataset of prompts by resetting the environment.
    Each item is a fresh episode starting prompt, formatted with
    the model's chat template.

    Per guide: "Use the same prompt repeated many times, routed through
    an environment, with TRL driving training."
    """
    prompts = []

    print(f"[Trace] Generating {n_tasks} training prompts...")
    for i in range(n_tasks):
        task = curriculum.sample()

        try:
            resp = requests.post(f"{env_url}/reset", json=task, timeout=10)
            if resp.status_code != 200:
                continue
        except requests.RequestException:
            continue

        obs = resp.json()

        # Format using chat template
        messages = format_prompt_for_chat(
            instruction=obs["instruction"],
            sources=obs.get("available_sources", ["gmail"]),
            step=0,
            context="",
            memory=obs.get("memory_summary", ""),
        )

        # Apply tokenizer's chat template to get the formatted prompt string
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback if chat template not available
            prompt_text = (
                f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        prompts.append({
            "prompt": prompt_text,
            "task": json.dumps(task),
        })

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_tasks} prompts")

    print(f"[Trace] Dataset size: {len(prompts)}")
    return Dataset.from_list(prompts)


# ── Reward functions ─────────────────────────────────────────────────────────

def make_reward_functions(env_url: str, cfg: dict) -> list:
    """
    Build the list of independent reward functions for GRPO.

    Per guide: "Use multiple independent reward functions, not just one.
    If you only have a single reward signal, it is easier for the model
    to hack it. Multiple independent checks reduce that risk."
    """
    from environments.trace_env.rewards.reward_fn import make_trl_reward_functions
    return make_trl_reward_functions(env_url)


# ── SFT warm-start ───────────────────────────────────────────────────────────

def run_sft_warmstart(model, tokenizer, cfg: dict):
    """
    Run light SFT before GRPO.

    Per guide: "In many practical cases, do a little SFT first, then RL"
    """
    from trl import SFTTrainer, SFTConfig

    sft_cfg = cfg.get("sft", {})
    dataset_path = sft_cfg.get("dataset_path", "./data/sft_demos.jsonl")

    if not os.path.exists(dataset_path):
        print(f"[Trace] SFT dataset not found at {dataset_path}")
        print(f"[Trace] Generate it first: python -m training.generate_sft_data")
        print(f"[Trace] Skipping SFT warm-start.")
        return model

    print(f"[Trace] Running SFT warm-start from {dataset_path}")

    # Load SFT dataset
    sft_data = []
    with open(dataset_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            sft_data.append(item)

    dataset = Dataset.from_list(sft_data)
    print(f"[Trace] SFT dataset: {len(dataset)} examples")

    sft_config = SFTConfig(
        output_dir=sft_cfg.get("checkpoint_path", "./outputs/trace-sft"),
        num_train_epochs=sft_cfg.get("epochs", 2),
        per_device_train_batch_size=sft_cfg.get("batch_size", 4),
        gradient_accumulation_steps=sft_cfg.get("grad_accum", 2),
        learning_rate=sft_cfg.get("lr", 2e-4),
        max_seq_length=sft_cfg.get("max_seq_length", 2048),
        logging_steps=10,
        save_steps=100,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    print("[Trace] Starting SFT training...")
    trainer.train()

    # Save SFT checkpoint
    save_path = sft_cfg.get("checkpoint_path", "./outputs/trace-sft")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[Trace] SFT checkpoint saved to {save_path}")

    return model


# ── Main training loop ───────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_name = args.model or args.sft_checkpoint or cfg.get("model_name", "unsloth/Qwen2.5-3B-Instruct")
    env_url = args.env_url

    print("=" * 60)
    print("[Trace] GRPO Training Pipeline (Unsloth + TRL + RLVR)")
    print("=" * 60)
    print(f"  Model:       {model_name}")
    print(f"  Environment: {env_url}")
    print(f"  SFT first:   {args.sft_first}")
    print(f"  Dry run:     {args.dry_run}")
    print("=" * 60)

    # ── Load model with Unsloth ──────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_name, cfg)

    # ── Optional SFT warm-start ──────────────────────────────────────────
    if args.sft_first and cfg.get("sft", {}).get("enabled", True):
        model = run_sft_warmstart(model, tokenizer, cfg)

    # ── Curriculum & dataset ─────────────────────────────────────────────
    curriculum = TaskCurriculum(cfg.get("curriculum", {}))

    print("[Trace] Generating training prompts from environment...")
    dataset = build_dataset(
        curriculum, env_url, tokenizer,
        n_tasks=2 if args.dry_run else cfg.get("n_tasks", 200),
    )

    if len(dataset) == 0:
        print("[Trace] ERROR: No prompts generated. Is the environment running?")
        print(f"[Trace] Check: {env_url}/health")
        sys.exit(1)

    # ── Build reward functions ───────────────────────────────────────────
    reward_fns = make_reward_functions(env_url, cfg)
    print(f"[Trace] Reward functions: {[fn.__name__ for fn in reward_fns]}")

    # ── GRPO config ──────────────────────────────────────────────────────
    grpo_kwargs = dict(
        output_dir=cfg.get("output_dir", "./outputs/trace-grpo"),
        num_train_epochs=1 if args.dry_run else cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 2),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        learning_rate=cfg.get("lr", 5e-6),
        max_completion_length=cfg.get("max_completion_length", 512),
        max_prompt_length=cfg.get("max_prompt_length", 1536),
        temperature=cfg.get("temperature", 0.9),
        num_generations=cfg.get("num_generations", 8),
        logging_steps=cfg.get("logging_steps", 1),
        save_steps=cfg.get("save_steps", 50),
        report_to=cfg.get("report_to", "none"),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
    )

    # Add KL coefficient if specified
    kl_coef = cfg.get("kl_coef")
    if kl_coef is not None:
        grpo_kwargs["beta"] = kl_coef  # TRL uses 'beta' for KL coefficient

    # Add vLLM if enabled
    if cfg.get("use_vllm", False):
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_gpu_memory_utilization"] = cfg.get("vllm_gpu_memory_fraction", 0.5)

    grpo_config = GRPOConfig(**grpo_kwargs)

    # ── Training callbacks ───────────────────────────────────────────────
    callbacks = [
        TraceRewardCallback(
            env_url=env_url,
            curriculum=curriculum,
            sample_every_n_steps=20,
            output_dir=cfg.get("output_dir", "./outputs/trace-grpo"),
        )
    ]

    # ── Train ────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fns,
        args=grpo_config,
        train_dataset=dataset,
        callbacks=callbacks,
    )

    print("[Trace] Starting GRPO training...")
    start_time = time.time()

    if args.dry_run:
        # Run just a couple of steps
        print("[Trace] DRY RUN — running 2 steps then exiting")
        trainer.args.max_steps = 2
        trainer.train()
    else:
        trainer.train()

    elapsed = time.time() - start_time
    print(f"[Trace] Training completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ── Export ────────────────────────────────────────────────────────────
    if not args.no_export:
        _export_model(model, tokenizer, cfg)

    print("[Trace] Done!")


def _export_model(model, tokenizer, cfg: dict):
    """Export model in configured formats using Unsloth."""
    save_path = cfg.get("save_path", "./outputs/trace-final")
    export_cfg = cfg.get("export", {})

    print(f"\n[Trace] Exporting model to {save_path}")

    # ── Merged 16-bit (always recommended) ───────────────────────────────
    if export_cfg.get("merged_16bit", True):
        path_16 = f"{save_path}-16bit"
        print(f"[Trace] Saving merged 16-bit to {path_16}")
        model.save_pretrained_merged(
            path_16, tokenizer,
            save_method="merged_16bit",
        )

    # ── Merged 4-bit ─────────────────────────────────────────────────────
    if export_cfg.get("merged_4bit", False):
        path_4 = f"{save_path}-4bit"
        print(f"[Trace] Saving merged 4-bit to {path_4}")
        model.save_pretrained_merged(
            path_4, tokenizer,
            save_method="merged_4bit_forced",
        )

    # ── GGUF export (for Ollama / llama.cpp) ─────────────────────────────
    if export_cfg.get("gguf", False):
        quant_methods = export_cfg.get("gguf_quant_methods", ["q4_k_m"])
        for method in quant_methods:
            path_gguf = f"{save_path}-gguf-{method}"
            print(f"[Trace] Saving GGUF ({method}) to {path_gguf}")
            try:
                model.save_pretrained_gguf(
                    path_gguf, tokenizer,
                    quantization_method=method,
                )
            except Exception as e:
                print(f"[Trace] GGUF export failed for {method}: {e}")
                print(f"[Trace] You may need: pip install llama-cpp-python")

    # ── Push to HuggingFace Hub ──────────────────────────────────────────
    if export_cfg.get("push_to_hub", False):
        hub_repo = export_cfg.get("hub_repo", "")
        if hub_repo:
            print(f"[Trace] Pushing to HuggingFace Hub: {hub_repo}")
            try:
                model.push_to_hub_merged(
                    hub_repo, tokenizer,
                    save_method="merged_16bit",
                )
            except Exception as e:
                print(f"[Trace] Hub push failed: {e}")
                print(f"[Trace] Set HF_TOKEN environment variable")

    # ── LoRA-only save (lightweight) ─────────────────────────────────────
    lora_path = f"{save_path}-lora"
    print(f"[Trace] Saving LoRA adapters to {lora_path}")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    print(f"[Trace] Export complete!")


if __name__ == "__main__":
    main()
