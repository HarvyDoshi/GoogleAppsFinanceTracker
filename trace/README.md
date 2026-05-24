# 🕵️ Trace — "Your Digital Footprint"
> Meta OpenEnv Hackathon 2026 | Team Submission

**Trace** is a privacy-centric, multi-agent RL environment that builds a dynamic *Semantic World Model* of a user's fragmented digital life — without centralizing data.

---

## Themes Addressed
| Theme | Coverage |
|-------|----------|
| **Theme #2** – Long-Horizon Planning & Instruction Following | Primary — federated multi-step retrieval across years of data |
| **Theme #1** – Multi-Agent Interactions | Secondary — planner, retriever, verifier, memory agents |
| **Theme #4** – Self-Improvement | Tertiary — agents learn from past executions, refine strategies |
| **Sub-theme: Scale AI** | Non-code long-horizon business/personal workflows |
| **Sub-theme: Patronus AI** | Consumer workflows with schema drift (Gmail/Drive APIs change) |

---

## Architecture Overview

```
User Query (e.g., "Audit all receipts from 2022-2024 and flag anomalies")
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                   PLANNER AGENT                          │
│  plan-act-verify framework | goal decomposition          │
└────────────────────────┬────────────────────────────────┘
                         │ sub-tasks
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   RETRIEVER AGENT   MEMORY AGENT  VERIFIER AGENT
   (federated fetch) (episodic KV) (reward scorer)
          │              │              │
          └──────────────┴──────────────┘
                         │ observations
                         ▼
              OpenEnv Environment Loop
              (reset / step / reward)
                         │
                         ▼
              TRL + Unsloth RL Training
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Bootstrap OpenEnv environment
cd environments/trace_env
openenv init   # or run: uvicorn app:app --reload

# 3. Run training script (Google Colab friendly)
cd training
python train_grpo.py --config ../configs/grpo_config.yaml

# 4. Evaluate
python scripts/evaluate.py --env-url http://localhost:8000
```

---

## Project Structure

```
trace/
├── environments/
│   └── trace_env/          # OpenEnv-compatible RL environment
│       ├── app.py           # FastAPI server (OpenEnv interface)
│       ├── core/
│       │   ├── env.py       # TraceEnv: reset(), step(), state()
│       │   ├── world_model.py  # Semantic World Model (SWM)
│       │   └── schemas.py   # Action / Observation dataclasses
│       ├── agents/
│       │   ├── planner.py   # Long-horizon goal decomposer
│       │   ├── retriever.py # Federated data fetcher (Gmail, Drive)
│       │   ├── memory.py    # Episodic + semantic memory store
│       │   └── verifier.py  # Plan verification agent
│       ├── tools/
│       │   ├── gmail_tool.py
│       │   ├── drive_tool.py
│       │   └── timeline_tool.py
│       └── rewards/
│           ├── reward_fn.py # Multi-component reward functions
│           └── anti_hack.py # Anti-reward-hacking guards
├── training/
│   ├── train_grpo.py        # Main RL training script (TRL + Unsloth)
│   ├── dataset.py           # Task curriculum generator
│   └── callbacks.py         # Training monitors
├── configs/
│   ├── grpo_config.yaml     # GRPO hyperparameters
│   └── env_config.yaml      # Environment settings
├── scripts/
│   ├── evaluate.py          # Reward curve evaluation
│   └── sample_outputs.py    # Anti-hacking output inspector
├── notebooks/
│   └── trace_colab.ipynb    # Colab-ready training notebook
├── docs/
│   └── blog_post.md         # HuggingFace mini-blog
├── requirements.txt
└── README.md
```

---

## Judging Criteria Alignment

| Criterion | Implementation |
|-----------|---------------|
| **Environment Innovation (40%)** | Federated multi-source retrieval + zero-knowledge SWM; novel schema-drift curriculum |
| **Storytelling (30%)** | Privacy narrative + before/after timeline demo |
| **Showing Reward Improvement (20%)** | Reward curves across 3 difficulty tiers; plan-quality scoring |
| **Training Script Setup (10%)** | OpenEnv + TRL GRPO + Unsloth Colab notebook |
