"""
scripts/evaluate.py

Evaluation script — runs N episodes and computes reward statistics.
Use this to generate the reward curves required by the judging criteria.

Supports two modes:
  1. Rule-based agent (default) — fast baseline evaluation
  2. Model-based agent (--model) — actual trained model inference using Unsloth

Outputs:
  - Per-difficulty reward breakdown (easy/medium/hard)
  - Per-component reward breakdown (format, correctness, efficiency, etc.)
  - CSV export for reward curve plotting
  - Statistical confidence intervals

Usage:
    # Baseline evaluation (rule-based):
    python scripts/evaluate.py --env-url http://localhost:8000 --n-episodes 50

    # Model evaluation (Unsloth):
    python scripts/evaluate.py --env-url http://localhost:8000 \\
        --model ./outputs/trace-final-16bit --n-episodes 50

    # Quick test:
    python scripts/evaluate.py --env-url http://localhost:8000 --n-episodes 5
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import statistics
from collections import defaultdict

import requests


# ── Task pool ────────────────────────────────────────────────────────────────

SAMPLE_TASKS = [
    {
        "instruction": "Find all travel and ride receipts from Gmail in the last 10 days.",
        "difficulty": "easy",
        "available_sources": ["gmail"],
        "ground_truth": {"answer": "Found ride receipts totaling", "expected_numeric_target": 120.50},
    },
    {
        "instruction": "Retrieve the latest 20 financial emails from Gmail.",
        "difficulty": "easy",
        "available_sources": ["gmail"],
        "ground_truth": {"answer": None, "expected_numeric_target": None},
    },
    {
        "instruction": "Audit all ride receipts from Gmail between 2022 and 2023, and calculate total spend.",
        "difficulty": "medium",
        "available_sources": ["gmail", "sheets"],
        "ground_truth": {"answer": "Total ride spend", "expected_numeric_target": 1500.00},
    },
    {
        "instruction": "Retrieve transactions from Gmail, then check Google Sheets to see if they are already logged.",
        "difficulty": "medium",
        "available_sources": ["gmail", "sheets"],
        "ground_truth": {"answer": None, "expected_numeric_target": None},
    },
    {
        "instruction": "Perform a full financial audit of travel and ride footprint from 2022-2024, flag missing receipts.",
        "difficulty": "hard",
        "available_sources": ["gmail", "sheets"],
        "ground_truth": {"answer": None},
    },
    {
        "instruction": "Audit the Google Sheets ledger against raw Gmail receipts to find discrepancies.",
        "difficulty": "hard",
        "available_sources": ["gmail", "sheets"],
        "ground_truth": {"answer": None},
    },
]


# ── Rule-based agent (baseline) ─────────────────────────────────────────────

def simulated_agent_step(obs: dict, step: int) -> dict:
    """A rule-based agent for evaluation baselines."""
    sources = obs.get("available_sources", ["gmail"])

    if step == 0:
        return {
            "action_type": "PLAN",
            "content": (
                f"Step 1: Retrieve relevant financial data from {sources[0]}. "
                f"Step 2: Analyze and categorize transactions. "
                f"Step 3: Verify totals. "
                f"Step 4: Produce summary answer."
            ),
        }
    elif step == 1:
        return {
            "action_type": "RETRIEVE",
            "content": obs["instruction"][:50],
            "source": sources[0],
        }
    elif step == 2 and len(sources) > 1:
        return {
            "action_type": "RETRIEVE",
            "content": "Read all existing transactions from the ledger.",
            "source": sources[1],
        }
    elif step == 3:
        return {
            "action_type": "MEMORIZE",
            "content": f"Retrieved data from {', '.join(sources)}. Ready to verify.",
        }
    elif step == 4:
        return {
            "action_type": "VERIFY",
            "content": f"Verifying findings for: {obs['instruction'][:60]}",
        }
    else:
        return {
            "action_type": "ANSWER",
            "content": (
                f"Based on analysis of {', '.join(sources)}, "
                f"I found relevant records matching the query: {obs['instruction'][:80]}. "
                f"Summary: Retrieved and verified data across all available sources. "
                f"Financial audit complete with categorized amounts."
            ),
        }


# ── Model-based agent ────────────────────────────────────────────────────────

class ModelAgent:
    """
    Evaluation agent that uses a trained model for inference.
    Uses Unsloth's FastLanguageModel.for_inference() for speed.
    """

    def __init__(self, model_path: str, max_seq_length: int = 4096):
        import torch
        from unsloth import FastLanguageModel

        print(f"[Evaluate] Loading model from {model_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

        # Enable Unsloth's fast inference mode (2x speedup)
        FastLanguageModel.for_inference(self.model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[Evaluate] Model loaded (fast inference mode enabled)")

    def step(self, obs: dict, step: int) -> dict:
        """Generate an action from the model."""
        import torch
        from training.dataset import format_prompt_for_chat

        messages = format_prompt_for_chat(
            instruction=obs.get("instruction", ""),
            sources=obs.get("available_sources", ["gmail"]),
            step=step,
            context=obs.get("context", "")[:300],
            memory=obs.get("memory_summary", ""),
        )

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt = f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n<|im_start|>user\n{messages[1]['content']}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,  # lower temp for evaluation
                do_sample=True,
                top_p=0.9,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated[len(prompt):].strip()

        # Parse JSON action
        try:
            # Handle markdown code blocks
            if "```json" in generated:
                generated = generated.split("```json")[1].split("```")[0].strip()
            elif "```" in generated:
                generated = generated.split("```")[1].split("```")[0].strip()

            action = json.loads(generated)
            return action
        except json.JSONDecodeError:
            # Fallback: try to extract action from text
            return {
                "action_type": "ANSWER",
                "content": generated[:200],
            }


# ── Episode runner ───────────────────────────────────────────────────────────

def run_episode(env_url: str, task: dict, agent=None) -> dict:
    """Run a single episode and return detailed reward statistics."""
    # Reset
    resp = requests.post(f"{env_url}/reset", json=task, timeout=15)
    if resp.status_code != 200:
        return {"error": resp.text}

    obs = resp.json()
    total_reward = 0.0
    step_rewards = []
    action_sequence = []
    done = False
    step = 0

    while not done and step < 10:
        if agent is not None:
            action = agent.step(obs, step)
        else:
            action = simulated_agent_step(obs, step)

        action_sequence.append(action.get("action_type", "UNKNOWN"))

        resp = requests.post(f"{env_url}/step", json=action, timeout=15)
        if resp.status_code != 200:
            break

        data = resp.json()
        reward = data["reward"]
        done = data["done"]
        obs = data["observation"]

        total_reward += reward
        step_rewards.append(reward)
        step += 1

    return {
        "difficulty": task["difficulty"],
        "total_reward": total_reward,
        "steps": step,
        "step_rewards": step_rewards,
        "action_sequence": action_sequence,
        "done": done,
    }


# ── Statistics helpers ───────────────────────────────────────────────────────

def confidence_interval(data: list[float], confidence: float = 0.95) -> tuple:
    """Compute mean and confidence interval."""
    n = len(data)
    if n < 2:
        return (sum(data) / max(n, 1), 0.0, 0.0)

    mean = statistics.mean(data)
    stderr = statistics.stdev(data) / math.sqrt(n)

    # t-value approximation for 95% CI
    t_val = 1.96 if n > 30 else 2.0
    margin = t_val * stderr

    return (mean, mean - margin, mean + margin)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trace Evaluation")
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--model", default=None,
                        help="Path to trained model for model-based evaluation")
    parser.add_argument("--output-csv", default="./outputs/eval_results.csv",
                        help="CSV output path for reward curves")
    args = parser.parse_args()

    # ── Load agent ───────────────────────────────────────────────────────
    agent = None
    if args.model:
        agent = ModelAgent(args.model)
        agent_type = "model"
    else:
        agent_type = "rule-based"

    print(f"\n{'='*60}")
    print(f"[Evaluate] Running {args.n_episodes} episodes ({agent_type} agent)")
    print(f"[Evaluate] Environment: {args.env_url}")
    print(f"{'='*60}\n")

    results_by_difficulty = defaultdict(list)
    all_results = []

    for i in range(args.n_episodes):
        task = random.choice(SAMPLE_TASKS).copy()
        start_time = time.time()
        result = run_episode(args.env_url, task, agent)
        elapsed = time.time() - start_time

        if "error" not in result:
            results_by_difficulty[result["difficulty"]].append(result["total_reward"])
            all_results.append(result)

        print(
            f"  Episode {i+1:3d} | {task['difficulty']:6s} | "
            f"reward={result.get('total_reward', 0):.3f} | "
            f"steps={result.get('steps', 0)} | "
            f"time={elapsed:.1f}s | "
            f"actions={' → '.join(result.get('action_sequence', []))}"
        )

    # ── Results ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS ({agent_type} agent, {len(all_results)} episodes)")
    print(f"{'='*60}")

    # Per-difficulty breakdown with confidence intervals
    for diff in ["easy", "medium", "hard"]:
        rewards = results_by_difficulty.get(diff, [])
        if rewards:
            mean, ci_lo, ci_hi = confidence_interval(rewards)
            mx = max(rewards)
            mn = min(rewards)
            print(
                f"  {diff:6s} | n={len(rewards):3d} | "
                f"avg={mean:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] | "
                f"max={mx:.3f} | min={mn:.3f}"
            )

    # Overall
    all_rewards = [r["total_reward"] for r in all_results]
    if all_rewards:
        mean, ci_lo, ci_hi = confidence_interval(all_rewards)
        print(f"\n  OVERALL | n={len(all_rewards):3d} | "
              f"avg={mean:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Action distribution
    action_counts = defaultdict(int)
    for r in all_results:
        for a in r.get("action_sequence", []):
            action_counts[a] += 1
    if action_counts:
        total_actions = sum(action_counts.values())
        print(f"\n  Action distribution:")
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            print(f"    {action:10s}: {count:4d} ({100*count/total_actions:.1f}%)")

    # ── Export CSV ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "difficulty", "total_reward", "steps", "done", "actions"])
        for i, r in enumerate(all_results):
            writer.writerow([
                i + 1,
                r["difficulty"],
                f"{r['total_reward']:.4f}",
                r["steps"],
                r["done"],
                " → ".join(r.get("action_sequence", [])),
            ])
    print(f"\n  CSV saved: {args.output_csv}")

    print(f"\n{'='*60}")
    print(f"[Evaluate] Done. Use the CSV for reward curve plots.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
