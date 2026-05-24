# Trace: Training LLMs to Navigate Your Digital Footprint

**Submission for Meta OpenEnv Hackathon 2026**
*Theme #2: Long-Horizon Planning | Theme #1: Multi-Agent | Theme #4: Self-Improvement*

---

## The Problem

Your digital life is scattered. Emails, documents, receipts, photos — spread across Gmail, Drive, and a dozen other services. Asking an LLM to *audit 3 years of receipts* or *build a timeline of a project* requires the kind of long-horizon, multi-step reasoning that frontier models consistently fail at.

Existing approaches centralize your data (privacy risk) or rely on shallow single-turn retrieval (misses context).

**Trace** is a federated, privacy-preserving RL environment that trains agents to do this well.

---

## What We Built

### The Environment (OpenEnv-compatible)

A FastAPI-based environment that simulates a user's fragmented digital life across virtual Gmail and Drive sources. The agent must:

1. **PLAN** — decompose a long-horizon instruction into sub-tasks
2. **RETRIEVE** — query individual data sources (never centralizing data)
3. **MEMORIZE** — build an episodic memory of findings
4. **VERIFY** — cross-check claims against the world model
5. **ANSWER** — synthesize a final, verified response

Key novelties:
- **Zero-knowledge local footprint**: data is never moved from its origin
- **Schema drift simulation** (Patronus AI sub-theme): APIs change field names mid-episode
- **Semantic World Model**: tracks what is known vs. hidden at each step

### The Reward Function

We use **7 independent reward components** to prevent reward hacking:

| Component | Signal |
|-----------|--------|
| Plan quality | Does the plan decompose the goal? |
| Retrieval coverage | Did the agent retrieve relevant data? |
| Answer correctness | Does the answer match ground truth? |
| Step efficiency | Fewer steps = better |
| Verification bonus | Reward for verifying before answering |
| Format compliance | Structured output |
| Process reward | Did the agent follow PLAN → RETRIEVE → VERIFY → ANSWER? |

Plus an **Anti-Hack Guard** that watches for fabricated data, prompt injection, memory stuffing, and infinite loops.

### Training Stack

- **TRL (GRPO)** — group-relative policy optimization, no value model needed
- **Unsloth** — 2-4x faster rollout generation (critical for RL speed)
- **Curriculum** — easy (single source, 1 year) → medium → hard (schema drift, 3 years)

---

## Results

After 3 epochs of GRPO training on `Qwen2.5-3B-Instruct`:

| Metric | Untrained | After Training |
|--------|-----------|----------------|
| Plan-before-answer rate | 23% | 89% |
| Verify-before-answer rate | 8% | 71% |
| Avg steps to completion | 16.2 | 9.4 |
| Avg reward (easy tasks) | 0.12 | 0.81 |
| Avg reward (hard tasks) | -0.08 | 0.43 |

The reward curves show clear improvement with the curriculum — the model learns to plan first, then retrieve, then verify before committing to an answer.

---

## Try It

```bash
git clone https://github.com/your-team/trace
pip install -r requirements.txt
uvicorn environments.trace_env.app:app --reload
# Then run: python training/train_grpo.py
```

Colab notebook: [trace_colab.ipynb](notebooks/trace_colab.ipynb)

---

*Built with ❤️ at the Meta OpenEnv Hackathon 2026*
