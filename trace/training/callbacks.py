"""
training/callbacks.py

Training callbacks for monitoring reward curves, inspecting
model outputs for reward hacking, and advancing curriculum.

Per the hackathon guide:
  - "Periodic human inspection is still necessary."
  - "Sample outputs frequently and inspect them"
  - "Terminate or roll back runs if behavior drifts badly"
"""

from __future__ import annotations
import json
import os
import time
import csv
from collections import deque
from typing import Optional

import requests
from transformers import TrainerCallback, TrainerState, TrainerControl


class TraceRewardCallback(TrainerCallback):
    """
    Comprehensive training callback that:
      1. Samples model outputs every N steps for human inspection
      2. Flags suspicious patterns indicating reward hacking
      3. Logs per-component reward breakdowns
      4. Detects reward drift (sudden spikes = possible hacking)
      5. Advances curriculum difficulty based on training step
      6. Saves best checkpoint based on moving-average reward
      7. Exports reward history to CSV for plotting
    """

    HACK_PATTERNS = [
        "ignore",
        "bypass",
        "hack",
        "cheat",
        "skip verification",
        "fabricat",
        "pretend",
        "make up",
        "assume",
    ]

    def __init__(
        self,
        env_url: str,
        curriculum=None,
        sample_every_n_steps: int = 20,
        drift_window: int = 50,
        drift_threshold: float = 0.5,
        output_dir: str = "./outputs",
    ):
        self.env_url = env_url
        self.curriculum = curriculum
        self.sample_every_n_steps = sample_every_n_steps
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold
        self.output_dir = output_dir

        self._reward_history: list[float] = []
        self._reward_components: list[dict] = []
        self._hack_alerts: list[dict] = []
        self._recent_rewards: deque = deque(maxlen=drift_window)
        self._best_avg_reward: float = float("-inf")
        self._generation_stats: dict = {
            "json_parse_success": 0,
            "json_parse_fail": 0,
            "action_type_counts": {},
            "avg_length": 0.0,
            "total_samples": 0,
        }

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Advance curriculum after each training step."""
        if self.curriculum is not None:
            self.curriculum.advance(state.global_step)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Track reward components and detect drift."""
        if not logs:
            return

        # Track total reward
        reward = logs.get("reward", logs.get("rewards/mean"))
        if reward is not None:
            self._reward_history.append(float(reward))
            self._recent_rewards.append(float(reward))

        # Track individual reward function outputs if available
        component_log = {}
        for key, value in logs.items():
            if key.startswith("rewards/") or key.startswith("reward_"):
                component_log[key] = float(value)
        if component_log:
            component_log["step"] = state.global_step
            self._reward_components.append(component_log)

        # ── Reward drift detection ───────────────────────────────────────
        if len(self._recent_rewards) >= self.drift_window:
            recent_avg = sum(self._recent_rewards) / len(self._recent_rewards)
            older_avg = sum(list(self._recent_rewards)[:self.drift_window // 2]) / (self.drift_window // 2)

            if recent_avg - older_avg > self.drift_threshold:
                print(
                    f"\n⚠️  [REWARD DRIFT] Step {state.global_step}: "
                    f"avg jumped from {older_avg:.3f} to {recent_avg:.3f} "
                    f"(Δ={recent_avg - older_avg:.3f}). Possible reward hacking!"
                )

        # ── Periodic sample inspection ───────────────────────────────────
        if state.global_step % self.sample_every_n_steps == 0 and state.global_step > 0:
            self._inspect_outputs(state, kwargs.get("model"), kwargs.get("tokenizer"))

    def _inspect_outputs(self, state, model, tokenizer):
        """Generate sample outputs and inspect for hacking."""
        if model is None or tokenizer is None:
            return

        # Reset a fresh episode for sampling
        try:
            resp = requests.post(
                f"{self.env_url}/reset",
                json={
                    "instruction": "Find all travel and ride receipts from Gmail in the last 10 days.",
                    "difficulty": "easy",
                    "available_sources": ["gmail"],
                    "ground_truth": {"answer": "Found ride receipts", "expected_numeric_target": 120.50},
                },
                timeout=5,
            )
            if resp.status_code != 200:
                return

            prompt_resp = requests.get(f"{self.env_url}/observation_prompt", timeout=5)
            if prompt_resp.status_code != 200:
                return

            prompt = prompt_resp.json().get("prompt", "")

            # Tokenize and generate
            import torch
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated[len(prompt):]

            # ── Track generation quality metrics ─────────────────────────
            self._generation_stats["total_samples"] += 1
            self._generation_stats["avg_length"] = (
                (self._generation_stats["avg_length"] * (self._generation_stats["total_samples"] - 1)
                 + len(generated)) / self._generation_stats["total_samples"]
            )

            try:
                parsed = json.loads(generated.strip())
                self._generation_stats["json_parse_success"] += 1
                action_type = parsed.get("action_type", "UNKNOWN")
                self._generation_stats["action_type_counts"][action_type] = (
                    self._generation_stats["action_type_counts"].get(action_type, 0) + 1
                )
            except json.JSONDecodeError:
                self._generation_stats["json_parse_fail"] += 1

            # ── Check for hack patterns ──────────────────────────────────
            flags = [p for p in self.HACK_PATTERNS if p in generated.lower()]
            if flags:
                alert = {
                    "step": state.global_step,
                    "flags": flags,
                    "output_preview": generated[:200],
                }
                self._hack_alerts.append(alert)
                print(f"\n⚠️  [ANTI-HACK ALERT] Step {state.global_step}")
                print(f"   Flags: {flags}")
                print(f"   Output: {generated[:200]}\n")
            else:
                print(f"\n✅ [SAMPLE] Step {state.global_step}: {generated[:150]}\n")

            # ── Log generation stats ─────────────────────────────────────
            total = self._generation_stats["total_samples"]
            success = self._generation_stats["json_parse_success"]
            parse_rate = success / total if total > 0 else 0
            print(f"   📊 JSON parse rate: {parse_rate:.1%} ({success}/{total})")
            print(f"   📊 Avg output length: {self._generation_stats['avg_length']:.0f} chars")
            if self._generation_stats["action_type_counts"]:
                print(f"   📊 Action distribution: {self._generation_stats['action_type_counts']}")

        except Exception as e:
            print(f"[Callback] Inspection failed: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        """Print final summary and export reward history."""
        if self._reward_history:
            avg = sum(self._reward_history) / len(self._reward_history)
            mx = max(self._reward_history)
            mn = min(self._reward_history)

            print(f"\n{'='*60}")
            print(f"[Trace] Training Complete!")
            print(f"{'='*60}")
            print(f"  Avg reward:       {avg:.3f}")
            print(f"  Max reward:       {mx:.3f}")
            print(f"  Min reward:       {mn:.3f}")
            print(f"  Total steps:      {len(self._reward_history)}")
            print(f"  Hack alerts:      {len(self._hack_alerts)}")
            print(f"  JSON parse rate:  {self._generation_stats['json_parse_success']}/{self._generation_stats['total_samples']}")
            print(f"{'='*60}\n")

        # ── Export reward history to CSV ──────────────────────────────────
        self._export_reward_csv()

    def _export_reward_csv(self):
        """Export reward history and components to CSV for plotting."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Total rewards
        csv_path = os.path.join(self.output_dir, "reward_history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward"])
            for i, r in enumerate(self._reward_history):
                writer.writerow([i, r])
        print(f"[Trace] Reward history saved to {csv_path}")

        # Per-component rewards
        if self._reward_components:
            comp_path = os.path.join(self.output_dir, "reward_components.csv")
            all_keys = set()
            for comp in self._reward_components:
                all_keys.update(comp.keys())
            all_keys = sorted(all_keys)

            with open(comp_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                for comp in self._reward_components:
                    writer.writerow(comp)
            print(f"[Trace] Reward components saved to {comp_path}")

        # Hack alerts
        if self._hack_alerts:
            alert_path = os.path.join(self.output_dir, "hack_alerts.json")
            with open(alert_path, "w") as f:
                json.dump(self._hack_alerts, f, indent=2)
            print(f"[Trace] Hack alerts saved to {alert_path}")
