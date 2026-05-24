# import json
# import wandb
# import os

# def analyze_sft_dataset(jsonl_path="data/sft_demos_v2.jsonl"):
#     if not os.path.exists(jsonl_path):
#         print(f"Error: Could not find {jsonl_path}")
#         print("Make sure you generated it first using generate_sft_data.py")
#         return

#     print(f"Analyzing SFT dataset: {jsonl_path}...")
    
#     # Initialize W&B run for dataset analysis
#     wandb.init(
#         project="trace-grpo",
#         name="sft-dataset-analysis",
#         job_type="dataset_eval",
#         config={
#             "dataset": jsonl_path,
#             "type": "Supervised Fine-Tuning Demonstrations",
#             "note": "Analysis of the generated expert trajectories including adversarial cases."
#         }
#     )

#     total_examples = 0
#     action_counts = {"PLAN": 0, "RETRIEVE": 0, "VERIFY": 0, "ANSWER": 0, "MEMORIZE": 0}
#     adversarial_flags = {"noise": 0, "refund": 0, "schema_drift": 0, "conflict": 0, "multi_item": 0}
    
#     # Read the dataset
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()
        
#     for step, line in enumerate(lines):
#         if not line.strip():
#             continue
            
#         total_examples += 1
#         data = json.loads(line)
#         text = data.get("text", "")
        
#         # 1. Extract Action Type
#         action_type = "UNKNOWN"
#         if '{"action_type"' in text:
#             try:
#                 # Naive extraction for analysis
#                 json_str = text[text.find('{"action_type"'):text.rfind('}')+1]
#                 action_data = json.loads(json_str)
#                 action_type = action_data.get("action_type", "UNKNOWN")
#                 if action_type in action_counts:
#                     action_counts[action_type] += 1
#             except:
#                 pass

#         # 2. Detect Adversarial Content in the Prompt
#         # We check the prompt part of the text for our injected adversarial keywords
#         is_adv = 0
#         if "swiggy" in text.lower() and "offers" in text.lower():
#             adversarial_flags["noise"] += 1
#             is_adv = 1
#         elif "refund" in text.lower() or "cancelled" in text.lower():
#             adversarial_flags["refund"] += 1
#             is_adv = 1
#         elif "cost" in text.lower() and "merchant" in text.lower():
#             adversarial_flags["schema_drift"] += 1
#             is_adv = 1
#         elif "uber eats" in text.lower():
#             adversarial_flags["conflict"] += 1
#             is_adv = 1
#         elif "grand total" in text.lower():
#             adversarial_flags["multi_item"] += 1
#             is_adv = 1

#         # 3. Log step-by-step to build a timeseries-like dashboard
#         wandb.log({
#             "dataset/example_length": len(text),
#             "dataset/is_adversarial_case": is_adv,
#             "distribution/plans": action_counts["PLAN"],
#             "distribution/retrieves": action_counts["RETRIEVE"],
#             "distribution/verifies": action_counts["VERIFY"],
#             "distribution/answers": action_counts["ANSWER"]
#         }, step=step)

#     # 4. Log final summary metrics
#     wandb.summary["total_examples"] = total_examples
#     wandb.summary["action_distribution"] = action_counts
#     wandb.summary["adversarial_cases"] = adversarial_flags
    
#     total_adv = sum(adversarial_flags.values())
#     wandb.summary["adversarial_percentage"] = (total_adv / total_examples) * 100 if total_examples > 0 else 0

#     print("\n✅ Dataset Analysis Complete!")
#     print(f"Total Examples: {total_examples}")
#     print(f"Action Distribution: {action_counts}")
#     print(f"Adversarial Cases Found: {total_adv} ({wandb.summary['adversarial_percentage']:.1f}%)")
#     print(f"Adversarial Breakdown: {adversarial_flags}")
    
#     wandb.finish()

# if __name__ == "__main__":
#     analyze_sft_dataset()


import json
import os
import re
from collections import Counter

import wandb


# Match ONLY the assistant turn's JSON. Tolerates whitespace/newlines.
# We intentionally use a non-greedy match and re.DOTALL so the JSON body
# can span lines without leaking past <|im_end|>.
_ASSISTANT_RE = re.compile(
    r'<\|im_start\|>assistant\s*\n(\{.?\})\s<\|im_end\|>',
    re.DOTALL,
)

_GOAL_RE    = re.compile(r'GOAL:\s*([^\n]+)')
_SOURCES_RE = re.compile(r'SOURCES:\s*([^\n]+)')

# Adversarial keyword groups. Broad enough to fire on real data variants.
_ADVERSARIAL_PATTERNS = {
    "noise":        [r'\bswiggy\b.*\boffers?\b', r'\bpromotional\b', r'\bspam\b'],
    "refund":       [r'\brefund\b', r'\bcancell?ed\b', r'\breversed\b', r'\bchargeback\b'],
    "schema_drift": [r'\bmerchant\b.\bcost\b', r'\bcost\b.\bmerchant\b',
                     r'\bamount_inr\b', r'\bamount_usd\b', r'\bschema\b'],
    "conflict":     [r'\buber\s*eats\b', r'\bubereats\b', r'\bduplicate\b'],
    "multi_item":   [r'\bgrand\s*total\b', r'\bsubtotal\b', r'\bline\s*items?\b'],
}
_ADVERSARIAL_PATTERNS = {
    k: [re.compile(p, re.IGNORECASE) for p in v]
    for k, v in _ADVERSARIAL_PATTERNS.items()
}

# Difficulty inference from goal phrasing. Mirrors the easy/medium/hard split
# used in training/dataset.py so the dashboard reflects the curriculum.
_HARD_KEYWORDS   = ('calculate', 'total', 'aggregate', 'flag', 'anomal',
                    'recurring', 'audit', 'compare')
_MEDIUM_KEYWORDS = ('cross-reference', 'sheet', 'combined', ' vs ', 'reconcile')


def _extract_action(text: str) -> dict | None:
    """Pull the assistant's action JSON out of one chat-formatted example."""
    m = _ASSISTANT_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _infer_difficulty(goal: str) -> str:
    g = goal.lower()
    if any(w in g for w in _HARD_KEYWORDS):
        return "hard"
    if any(w in g for w in _MEDIUM_KEYWORDS):
        return "medium"
    return "easy"


def _detect_adversarial(text: str) -> str | None:
    """Return the first adversarial category that fires, or None."""
    for category, patterns in _ADVERSARIAL_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            return category
    return None


def analyze_sft_dataset(jsonl_path: str = "../data/sft_demos_v2.jsonl") -> None:
    if not os.path.exists(jsonl_path):
        print(f"Error: Could not find {jsonl_path}")
        print("Make sure you generated it first using generate_sft_data.py")
        return

    print(f"Analyzing SFT dataset: {jsonl_path}...")

    wandb.init(
        project="trace-grpo",
        name="sft-dataset-analysis",
        job_type="dataset_eval",
        config={
            "dataset": jsonl_path,
            "type": "Supervised Fine-Tuning Demonstrations",
            "note": "Analysis of the generated expert trajectories including adversarial cases.",
        },
    )

    # Counters
    action_counts     = Counter({"PLAN": 0, "RETRIEVE": 0, "MEMORIZE": 0,
                                 "VERIFY": 0, "ANSWER": 0, "UNKNOWN": 0})
    source_counts     = Counter({"gmail": 0, "sheets": 0, "null": 0})
    difficulty_counts = Counter({"easy": 0, "medium": 0, "hard": 0})
    adversarial_flags = Counter({k: 0 for k in _ADVERSARIAL_PATTERNS})

    total_examples   = 0
    multi_source_n   = 0
    total_text_chars = 0

    # Read all lines first so we know episode boundaries and totals.
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]

    # Build a W&B Table for the final summary view.
    table = wandb.Table(columns=[
        "step", "goal", "action_type", "source", "difficulty",
        "is_adversarial", "adv_category", "length_chars",
    ])

    for step, raw in enumerate(lines):
        total_examples += 1
        data = json.loads(raw)
        text = data.get("text", "")
        total_text_chars += len(text)

        # 1. Action type — fixed extraction.
        action = _extract_action(text)
        if action is not None:
            atype = action.get("action_type", "UNKNOWN")
            src   = action.get("source") or "null"
        else:
            atype, src = "UNKNOWN", "null"

        if atype not in action_counts:
            atype = "UNKNOWN"
        action_counts[atype] += 1

        if src not in source_counts:
            source_counts[src] = 0
        source_counts[src] += 1

        # 2. Goal & difficulty inference.
        gm = _GOAL_RE.search(text)
        goal = gm.group(1).strip() if gm else ""
        difficulty = _infer_difficulty(goal) if goal else "easy"
        difficulty_counts[difficulty] += 1

        # 3. Multi-source flag (federated-retrieval evidence).
        sm = _SOURCES_RE.search(text)
        is_multi_source = 1 if (sm and "," in sm.group(1)) else 0
        multi_source_n += is_multi_source

        # 4. Adversarial detection.
        adv_category = _detect_adversarial(text)
        is_adv = 1 if adv_category else 0
        if adv_category:
            adversarial_flags[adv_category] += 1

        # 5. Per-step W&B log. We log BOTH instantaneous one-hots AND
        # running cumulative counts. The cumulative versions are what
        # used to be flat-zero — they will now show real curves.
        wandb.log({
            # Instantaneous: 0/1 spike at this row's action type.
            "action_step/plan":     int(atype == "PLAN"),
            "action_step/retrieve": int(atype == "RETRIEVE"),
            "action_step/memorize": int(atype == "MEMORIZE"),
            "action_step/verify":   int(atype == "VERIFY"),
            "action_step/answer":   int(atype == "ANSWER"),

            # Cumulative running counts (these are the panels that were broken).
            "distribution/plans":     action_counts["PLAN"],
            "distribution/retrieves": action_counts["RETRIEVE"],
            "distribution/memorizes": action_counts["MEMORIZE"],
            "distribution/verifies":  action_counts["VERIFY"],
            "distribution/answers":   action_counts["ANSWER"],

            # Source usage (instantaneous one-hot).
            "source_step/gmail":  int(src == "gmail"),
            "source_step/sheets": int(src == "sheets"),
            "source_step/null":   int(src == "null"),

            # Difficulty (instantaneous one-hot).
            "difficulty/easy":   int(difficulty == "easy"),
            "difficulty/medium": int(difficulty == "medium"),
            "difficulty/hard":   int(difficulty == "hard"),

            # Per-example properties.
            "dataset/example_length":    len(text),
            "dataset/multi_source":      is_multi_source,
            "dataset/is_adversarial_case": is_adv,
        }, step=step)

        # Add a row to the table (cap to keep table light).
        if step < 250:
            table.add_data(
                step,
                goal[:80],
                atype,
                src,
                difficulty,
                is_adv,
                adv_category or "",
                len(text),
            )

    # ── Final summary panels ────────────────────────────────────────────
    # Bar charts for the W&B run page (these render as proper bars,
    # not as misleading time-series).
    wandb.log({
        "summary/action_dist": wandb.plot.bar(
            wandb.Table(
                data=[[k, v] for k, v in action_counts.items() if v > 0],
                columns=["action", "count"],
            ),
            "action", "count", title="Action distribution (final)",
        ),
        "summary/source_dist": wandb.plot.bar(
            wandb.Table(
                data=[[k, v] for k, v in source_counts.items() if v > 0],
                columns=["source", "count"],
            ),
            "source", "count", title="Source distribution (final)",
        ),
        "summary/difficulty_dist": wandb.plot.bar(
            wandb.Table(
                data=[[k, v] for k, v in difficulty_counts.items() if v > 0],
                columns=["difficulty", "count"],
            ),
            "difficulty", "count", title="Curriculum mix (inferred from goals)",
        ),
        "summary/dataset_table": table,
    })

    # ── wandb.summary scalars (shown prominently on run page) ───────────
    total_adv = sum(adversarial_flags.values())
    avg_len = total_text_chars / total_examples if total_examples else 0

    wandb.summary["total_examples"]            = total_examples
    wandb.summary["avg_example_length_chars"]  = round(avg_len, 1)
    wandb.summary["multi_source_examples"]     = multi_source_n
    wandb.summary["multi_source_pct"]          = round(
        100 * multi_source_n / total_examples, 1) if total_examples else 0.0

    # Action distribution scalars
    for k, v in action_counts.items():
        wandb.summary[f"actions/{k.lower()}"] = v
    wandb.summary["actions/unknown_pct"] = round(
        100 * action_counts["UNKNOWN"] / total_examples, 2) if total_examples else 0.0

    # Source distribution scalars
    for k, v in source_counts.items():
        wandb.summary[f"sources/{k}"] = v

    # Difficulty distribution scalars
    for k, v in difficulty_counts.items():
        wandb.summary[f"difficulty_total/{k}"] = v

    # Adversarial scalars
    wandb.summary["adversarial_total"] = total_adv
    wandb.summary["adversarial_percentage"] = round(
        100 * total_adv / total_examples, 2) if total_examples else 0.0
    for k, v in adversarial_flags.items():
        wandb.summary[f"adversarial/{k}"] = v

    # ── Console summary ────────────────────────────────────────────────
    print("\n✅ Dataset Analysis Complete!")
    print(f"  Total Examples:          {total_examples}")
    print(f"  Avg length (chars):      {avg_len:.0f}")
    print(f"  Multi-source examples:   {multi_source_n} "
          f"({100 * multi_source_n / max(total_examples, 1):.1f}%)")
    print(f"  Action Distribution:     {dict(action_counts)}")
    print(f"  Source Distribution:     {dict(source_counts)}")
    print(f"  Difficulty Distribution: {dict(difficulty_counts)}")
    print(f"  Adversarial Cases:       {total_adv} "
          f"({100 * total_adv / max(total_examples, 1):.1f}%)")
    print(f"  Adversarial Breakdown:   {dict(adversarial_flags)}")

    if action_counts["UNKNOWN"] > total_examples * 0.05:
        print("\n  ⚠️  >5% UNKNOWN actions — assistant JSON may not be parseable. "
              "Check chat-template formatting in generate_sft_data.py.")

    wandb.finish()


if __name__ == "__main__":
    analyze_sft_dataset()