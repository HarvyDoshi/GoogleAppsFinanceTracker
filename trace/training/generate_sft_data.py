"""
training/generate_sft_data.py

Generate SFT demonstration data by running the hardcoded expert policy
against the live Trace environment.

Per hackathon guide: "SFT is generally more sample-efficient"

This captures (prompt, expert_action) pairs formatted with chat templates,
producing a JSONL dataset for the SFT warm-start trainer.

Usage:
    # Make sure the environment server is running:
    uvicorn environments.trace_env.app:app --port 8000

    # Generate demonstrations:
    python -m training.generate_sft_data \\
        --env-url http://localhost:8000 \\
        --n-episodes 50 \\
        --output data/sft_demos.jsonl

    # Then train SFT:
    python -m training.train_sft --config configs/grpo_config.yaml
"""

import argparse
import json
import os
import random
import sys
import requests
from typing import Optional

# Import instruction pools for diverse tasks
from .dataset import (
    EASY_INSTRUCTIONS, MEDIUM_INSTRUCTIONS, HARD_INSTRUCTIONS,
    _make_ground_truth, format_prompt_for_chat,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SFT demonstrations for Trace")
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="Number of demonstration episodes to generate")
    parser.add_argument("--output", default="./data/sft_demos.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-3B-Instruct",
                        help="Model name (for chat template)")
    return parser.parse_args()


# ── Expert policy (rule-based, high-quality demonstrations) ──────────────────

def expert_step(obs: dict, step: int, task: dict) -> dict:
    """
    A hand-crafted expert policy that demonstrates the ideal
    action sequence: PLAN → RETRIEVE → MEMORIZE → VERIFY → ANSWER.

    This produces training signal for format compliance, process ordering,
    and domain vocabulary.
    """
    instruction = task.get("instruction", "")
    sources = task.get("available_sources", ["gmail"])
    difficulty = task.get("difficulty", "easy")

    if step == 0:
        # PLAN: decompose the instruction into steps
        plan_steps = [
            f"Step 1: Retrieve relevant financial data from {sources[0]}",
        ]
        if len(sources) > 1:
            plan_steps.append(f"Step 2: Cross-reference with {sources[1]} records")
        plan_steps.extend([
            f"Step {len(plan_steps)+1}: Analyze and categorize all transactions",
            f"Step {len(plan_steps)+2}: Verify totals and flag discrepancies",
            f"Step {len(plan_steps)+3}: Produce a summary answer with exact amounts",
        ])
        return {
            "action_type": "PLAN",
            "content": ". ".join(plan_steps),
            "source": None,
        }

    elif step == 1:
        # RETRIEVE from primary source
        query = instruction[:80] if len(instruction) > 80 else instruction
        return {
            "action_type": "RETRIEVE",
            "content": query,
            "source": sources[0],
        }

    elif step == 2 and len(sources) > 1:
        # RETRIEVE from secondary source
        return {
            "action_type": "RETRIEVE",
            "content": f"Read all existing transactions from the ledger.",
            "source": sources[1],
        }

    elif step == 2 or (step == 3 and len(sources) > 1):
        # MEMORIZE key findings
        return {
            "action_type": "MEMORIZE",
            "content": (
                f"Retrieved data from {', '.join(sources)}. "
                f"Key task: {instruction[:60]}. "
                f"Need to analyze totals and verify before answering."
            ),
            "source": None,
        }

    elif step == 3 or (step == 4 and len(sources) > 1):
        # VERIFY if medium/hard
        if difficulty in ("medium", "hard"):
            return {
                "action_type": "VERIFY",
                "content": f"Verify that retrieved data covers the full scope: {instruction[:60]}",
                "source": None,
            }
        else:
            # Easy: skip to ANSWER
            return _make_answer(instruction, sources)

    else:
        # ANSWER
        return _make_answer(instruction, sources)


def _make_answer(instruction: str, sources: list) -> dict:
    """Generate a well-formatted answer action."""
    return {
        "action_type": "ANSWER",
        "content": (
            f"Based on analysis of {', '.join(sources)}, "
            f"I found relevant records matching the query: {instruction[:80]}. "
            f"Summary: Retrieved and verified data across all available sources. "
            f"Financial audit complete with exact amounts extracted and categorized."
        ),
        "source": None,
    }


# ── Main generation loop ────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Try loading tokenizer for chat template formatting
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        print(f"[SFT-Gen] Loaded tokenizer: {args.model_name}")
    except Exception as e:
        print(f"[SFT-Gen] Could not load tokenizer ({e}), using fallback format")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Build task pool
    task_pool = []
    for inst in EASY_INSTRUCTIONS:
        task_pool.append({
            "instruction": inst, "difficulty": "easy",
            "available_sources": ["gmail"],
            "ground_truth": _make_ground_truth(inst, "easy"),
        })
    for inst in MEDIUM_INSTRUCTIONS:
        task_pool.append({
            "instruction": inst, "difficulty": "medium",
            "available_sources": ["gmail", "sheets"],
            "ground_truth": _make_ground_truth(inst, "medium"),
        })
    for inst in HARD_INSTRUCTIONS:
        task_pool.append({
            "instruction": inst, "difficulty": "hard",
            "available_sources": ["gmail", "sheets"],
            "ground_truth": _make_ground_truth(inst, "hard"),
        })

    print(f"[SFT-Gen] Generating {args.n_episodes} demonstration episodes...")
    print(f"[SFT-Gen] Task pool: {len(task_pool)} unique tasks")
    print(f"[SFT-Gen] Output: {args.output}")

    all_examples = []
    success_count = 0

    for ep in range(args.n_episodes):
        task = random.choice(task_pool)

        # Reset environment
        try:
            resp = requests.post(f"{args.env_url}/reset", json=task, timeout=10)
            if resp.status_code != 200:
                print(f"  Episode {ep+1}: reset failed ({resp.status_code})")
                continue
            obs = resp.json()
        except requests.RequestException as e:
            print(f"  Episode {ep+1}: connection error ({e})")
            continue

        # Run expert policy for this episode
        max_steps = 6 if task["difficulty"] == "easy" else 8
        for step in range(max_steps):
            # Build prompt from current observation
            messages = format_prompt_for_chat(
                instruction=obs.get("instruction", task["instruction"]),
                sources=task["available_sources"],
                step=step,
                context=obs.get("context", "")[:300],
                memory=obs.get("memory_summary", ""),
            )

            # Get expert action
            action = expert_step(obs, step, task)
            action_json = json.dumps(action, ensure_ascii=False)

            # Format as chat template text
            if tokenizer is not None:
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                    full_text = prompt_text + action_json + tokenizer.eos_token
                except Exception:
                    full_text = _fallback_format(messages, action_json)
            else:
                full_text = _fallback_format(messages, action_json)

            all_examples.append({"text": full_text})

            # Execute action in environment to get next observation
            try:
                resp = requests.post(f"{args.env_url}/step", json=action, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    obs = data.get("observation", obs)
                    if data.get("done", False):
                        break
                else:
                    break
            except requests.RequestException:
                break

        success_count += 1
        if (ep + 1) % 10 == 0:
            print(f"  Generated {ep + 1}/{args.n_episodes} episodes ({len(all_examples)} examples)")

    # ── Save dataset ─────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\n[SFT-Gen] Done!")
    print(f"  Episodes: {success_count}/{args.n_episodes}")
    print(f"  Examples: {len(all_examples)}")
    print(f"  Saved to: {args.output}")
    print(f"\n[SFT-Gen] Next step: python -m training.train_sft --config configs/grpo_config.yaml")


def _fallback_format(messages: list[dict], action_json: str) -> str:
    """Fallback formatting when tokenizer chat template is unavailable."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{action_json}<|im_end|>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
