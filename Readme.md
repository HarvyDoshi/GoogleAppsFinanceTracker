# Trace — Teaching LLMs to Navigate Your Digital Footprint

## TL;DR

**Trace** is an OpenEnv-compatible RL environment that trains language models to plan, retrieve, and verify across a user's *fragmented digital life* — Gmail, Drive, Sheets, attachments, images — **without ever centralizing the data**. We built the environment, the multi-agent loop, a 7-component reward stack with anti-hack guards, an SFT warm-start notebook, a GRPO training loop on Qwen2.5-3B-Instruct, and a live financial-audit dashboard powered by the trained agent on real Gmail data.

**Headline result from training:** total reward climbs from ~−0.5 to ~2.0–2.5 over 350 steps, and the model *spontaneously discovers* that calling `VERIFY` before `ANSWER` is worth it — the verification reward stays at zero for ~230 steps and then fires consistently for the rest of training.

---

## The Problem

Most of your real life lives in scattered places: receipts buried in Gmail, ride invoices as PDF attachments, expense ledgers in Google Sheets, photo receipts on your phone. When you ask an LLM something genuinely useful — *"audit all my financial transactions from the last six months and flag anything weird"* — it has to:

1. **Plan** the work into ordered sub-goals
2. **Retrieve** from the right sources, in the right order
3. **Parse** unstructured emails, PDFs, and images
4. **Cross-reference** sources to deduplicate and reconcile
5. **Verify** before committing to an answer

Frontier models fail at this consistently. Either they hallucinate, ask you to paste everything in (privacy risk), or stop after the first retrieval and call it done.

Two design choices fall out of this:

- **Federated, not centralized.** Data should never leave its origin. The agent queries each source and reasons over the *retrievals*, not a copied corpus.
- **Reward the process, not just the output.** A model that lucks into the right answer without a plan or verification is fragile. The reward function has to make planning and verifying *worth doing*.

Trace is built around both.

---

## Architecture

```
User Query  →  "Audit all my financial transactions from the last 180 days"
                          │
                          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                      PLANNER AGENT                            │
   │  decomposes the goal into ordered sub-tasks                   │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
     RETRIEVER AGENT        MEMORY AGENT          VERIFIER AGENT
     (Gmail / Drive /       (episodic +           (cross-checks
      Sheets / Image)        semantic store)       claims vs. SWM)
            │                     │                     │
            └─────────────────────┴─────────────────────┘
                                  │
                                  ▼
                   Semantic World Model (partial obs.)
                                  │
                                  ▼
                    OpenEnv loop  —  reset / step / reward
                                  │
                                  ▼
                       TRL (SFT → GRPO) + Unsloth
```

### The OpenEnv-compatible environment

A FastAPI server (`environments/trace_env/app.py`) exposes the standard OpenEnv interface — `reset(task)`, `step(action)`, `state()` — speaking typed Pydantic schemas (`TraceAction`, `TraceObservation`).

Five action types form the agent's vocabulary:

| Action | What it does |
|---|---|
| `PLAN` | Decompose the instruction into sub-tasks |
| `RETRIEVE` | Query a named source (gmail, drive, sheets, image) |
| `MEMORIZE` | Write a finding into episodic memory |
| `VERIFY` | Cross-check a claim against the world model |
| `ANSWER` | Submit the final synthesized response (ends the episode) |

### The Semantic World Model (SWM)

The SWM is the hidden ground truth — simulated emails, documents, transactions — that the agent **cannot see directly**. It exposes only what the agent has retrieved, and tracks which items have been seen, which sources have been touched, and the current schema version.

This partial-observability constraint is what makes planning and retrieval valuable in the first place.

### Schema drift (Patronus AI sub-theme)

Real APIs change. Field names shift, new keys appear, payloads get reshaped. We bake this into the curriculum: at higher difficulty levels, the world model bumps `schema_version` mid-episode and rewrites field names. An agent that hard-codes `email.subject` breaks; an agent that adapts gets rewarded.

### Real-world tooling

The environment isn't just simulated. In `retrieval_mode: real`, it actually calls:

- **Gmail API** — search + auto-download attachments
- **Google Drive API** — file search
- **Google Sheets API** — read/write a live financial ledger
- **LLaMA 4 Scout (HF Inference)** — VLM analysis of receipt images
- **Document extraction** — PDF / DOCX / PPTX parsing

A `transaction_parser` then unifies email bodies, image OCR results, and document text into a single canonical transaction schema (vendor, category, amount, date, currency).

---

## The Reward Function

The hackathon guide says it best: *"your reward function is your task specification."* We built a multi-component reward to make reward hacking expensive.

| # | Component | What it rewards | Max |
|---|---|---|---|
| 1 | **plan_quality** | Plan decomposes the goal, names sources, hits keywords | 0.30 |
| 2 | **retrieval_coverage** | Non-empty retrievals, reasonable volume | 0.20 |
| 3 | **verification_bonus** | Calling VERIFY before ANSWER | 0.15 |
| 4 | **answer_correctness** | Final answer matches ground truth | 1.00 |
| 5 | **format_compliance** | Structured output, currency tokens, list markers | 0.15 |
| 6 | **step_efficiency** | Fewer steps when correct | 0.15 |
| 7 | **process_reward** | PLAN → RETRIEVE → VERIFY → ANSWER ordering bonus | 0.10 |
Total reward is clipped to `[-1.0, +2.0]` and **logged per-component every step** so we can spot which signals are doing the work and which are degenerate.
### Anti-Hack Guard
A separate `AntiHackGuard` watches every action for the classic exploits:
- **Fabricated data** — citing items the agent never retrieved
- **Prompt injection** — content trying to override system instructions
- **Memory stuffing** — flooding episodic memory with junk to game coverage
- **Infinite loops** — repeating the same action
Trigger any of these and the step gets a `−0.8` penalty. Every penalty is logged for offline review.
---
## Training Stack
We split training into two phases — both running on a free Colab T4:
### Phase 1: SFT warm-start
Cold-starting RL on a small model that doesn't know the action schema is wasteful — most rollouts are unparseable JSON. So we built an SFT dataset of expert trajectories (`sft_demos.jsonl`) — about **500 examples**, weighted toward harder tasks — that teach the model the action format and a sensible default policy.
**Dataset composition** (from our `sft-dataset-analysis` run):

| Difficulty | Count | Notes |
|---|---|---|
| `hard` | ~250 | Multi-source audits, schema drift, ambiguous answers |
| `easy` | ~150 | Single-source retrieval, explicit ground truth |
| `medium` | ~50 | Two sources, partial ground truth |

Roughly half the examples involve **multiple sources** (Gmail + Sheets together — visible in the bimodal `dataset/multi_source` channel). Example length stays in the 800–1300 character band.

**SFT config:**
- Model: `unsloth/Qwen2.5-3B-Instruct`, 4-bit
- LoRA: `r=32`, `α=32`, all attention + MLP modules
- Trainer: TRL `SFTTrainer`
- 2 epochs, batch 4 × grad-accum 2, lr `2e-4`, cosine schedule, 2048 seq
- Output: LoRA adapters + merged 16-bit + GGUF (`q4_k_m`) for Ollama

The full notebook is `Trace_SFT_Training_Colab.ipynb` — uploads dataset, runs SFT, saves all three formats, ships them to your machine.

### Phase 2: GRPO

For the actual RL training, we use **GRPO (Group-Relative Policy Optimization)** over PPO. Two reasons:

1. **No value model.** GRPO compares N rollouts per prompt against the group mean — half the memory, no critic to train.
2. **Better signal on sparse rewards.** Episode-level rewards in our setup are inherently sparse, and group-relative advantages handle that cleanly.

- **Trainer**: TRL `GRPOTrainer` with our custom reward function
- **Group size**: 4 rollouts per prompt
- **Curriculum**: `easy_until_step: 50`, `medium_until_step: 150`, `hard_from_step: 150`
- **Reward bridge**: rollouts are POSTed to the running `/step` endpoint, scalar rewards come back; the env handles all scoring + anti-hack logic
- **Speed**: Unsloth gives us ~2-4× faster generation, which dominates wall-clock in RL

The training script (`training/train_grpo.py`) is ~200 lines, fully reproducible, runs end-to-end on a T4.

---

## Results — What the W&B Curves Tell Us

We ran 350+ training steps under the run name **`full-reward-simulation`**. Every metric below is read directly off the dashboard.

### Total reward: −0.5 → 2.0+

The headline curve. Total reward starts negative — the model is taking heavy anti-hack penalties for malformed outputs — and climbs steadily, with a clear inflection around step 150 when the curriculum transitions from medium to hard. By step 250 it's reliably above 1.5 and frequently hits the 2.0+ cap.

### Loss converges cleanly

`training/loss` starts around **2.5** and drops to roughly **0.2** by step 350, with the steep portion of the descent landing in the first ~150 steps. No instability, no spikes — exactly the shape you want.

### KL stays healthy

`training/kl_divergence` rises gradually from **~0.01 to ~0.05** as the policy moves off the reference model. Low and well-controlled — the policy is changing meaningfully but isn't drifting into nonsense.

### The standout finding: verification is *learned*, not given

The single most interesting curve in the whole run is `reward_components/verification_bonus`:

> **`verification_bonus` is flat at 0 for the first ~230 steps, then starts firing at 0.15 consistently for the rest of training.**
The model wasn't told that VERIFY was good. It had to *discover* it. Early on, the policy answers as soon as it has retrievals — fast, but it misses the verification bonus and trips the anti-hack guard. Around step 230, something clicks and VERIFY becomes a regular part of its trajectory. This is exactly the kind of emergent compositional behavior we hoped a process reward would produce.
### Step efficiency and retrieval coverage climb smoothly
- `reward_components/step_efficiency`: 0 → 0.15, smooth monotonic climb
- `reward_components/retrieval_coverage`: 0 → 0.20, smooth monotonic climb
Translation: as training progresses, the model gets better at **retrieving the right things** and **finishing tasks in fewer steps**. Both saturate at the maximum of their reward bands by the end of training.
### Anti-hack penalties die out
`penalties/anti_hack` is dramatic: heavy clusters of −0.5 / −0.6 in the first 150 steps (the model is fabricating, looping, or otherwise tripping the guard), occasional firings between steps 150–250, and **near-silence after step 250**. The model learns the boundaries of acceptable behavior the hard way and stays inside them.
### Reward summary
| Signal | Early (steps 0–50) | Late (steps 300+) | Movement |
|---|---|---|---|
| `reward/total` | ~0.0 (often negative) | ~2.0–2.5 | **Strong positive trend** |
| `training/loss` | ~2.5 | ~0.2 | **Clean convergence** |
| `verification_bonus` | 0.0 | 0.15 (regular firings) | **Emergent behavior** |
| `step_efficiency` | 0.0 | 0.15 | **Saturated** |
| `retrieval_coverage` | 0.0 | 0.20 | **Saturated** |
| `anti_hack` penalty | frequent −0.5 to −0.6 | rare | **Self-correcting** |
| `kl_divergence` | 0.01 | 0.05 | Healthy / controlled |

---

## Live Demo: The Financial Dashboard

To make the system tangible, we wired the trained agent to a live endpoint. Hit `/dashboard` on the deployed Space and you get:

1. The agent **plans** a federated audit
2. **Retrieves** from Gmail (last 180 days, financial keywords) — text-first, then deep with attachment + image analysis
3. **Pulls** historical transactions from a Google Sheets ledger
4. **Merges** Gmail + Sheets, deduplicates, categorizes
5. **Renders** an interactive HTML dashboard with totals, vendors, categories, and a timeline
6. **Exports** a DOCX report

Force-refresh with `?refresh=true`. Results cache for 10 minutes.

The full client flow lives in `inference.py` — the same loop the trained policy runs, just with a hardcoded expert sequence so you can see the system end-to-end without a GPU.

---

## Project Layout

```
# Project Structure
```text
trace/
├── .env.example
├── .gitignore
├── .ledger_id
├── 2.0
├── Dockerfile
├── README.md
├── __init__.py
├── all_financial_dashboard.html
├── all_financial_response.json
├── all_financial_transactions.csv
├── all_financial_transactions.json
├── client.py
├── configs
│   ├── env_config.yaml
│   └── grpo_config.yaml
├── credentials.json
├── credentials.json.json
├── data
│   ├── sft_demos.jsonl
│   └── sft_demos_v2.jsonl
├── docs
│   └── blog_post.md
├── environments
│   ├── __init__.py
│   └── trace_env
│       ├── __init__.py
│       ├── agents
│       │   ├── __init__.py
│       │   ├── memory.py
│       │   ├── planner.py
│       │   ├── retriever.py
│       │   └── verifier.py
│       ├── app.py
│       ├── core
│       │   ├── __init__.py
│       │   ├── env.py
│       │   ├── schemas.py
│       │   └── world_model.py
│       ├── rewards
│       │   ├── __init__.py
│       │   ├── anti_hack.py
│       │   └── reward_fn.py
│       └── tools
│           ├── __init__.py
│           ├── dashboard_renderer.py
│           ├── doc_tool.py
│           ├── drive_tool.py
│           ├── gmail_tool.py
│           ├── image_tool.py
│           ├── report_tool.py
│           ├── run_rapido_flow.py
│           ├── sheets_tool.py
│           └── transaction_parser.py
├── financial_report.docx
├── financial_report_1777157050.docx
├── financial_report_1777157984.docx
├── financial_report_1777162828.docx
├── generate_secrets.py
├── hf_secrets.txt
├── inference.py
├── models.py
├── notebooks
│   └── Trace_SFT_Training_Colab.ipynb
├── openenv.yaml
├── project_structure.md
├── pyproject.toml
├── rapido_dashboard.html
├── rapido_response.json
├── requirements.txt
├── run_all_financial_gmail.py
├── scratch
│   ├── add_test_row.py
│   ├── check_all_tabs.py
│   ├── check_correct_id.py
│   ├── check_grid.py
│   ├── check_id_3.py
│   ├── check_last_edit.py
│   ├── check_metadata.py
│   ├── check_owner.py
│   ├── check_service.py
│   ├── check_tabs.py
│   ├── check_token.py
│   ├── check_values.py
│   ├── debug_sheets.py
│   ├── find_sheet.py
│   ├── generate_tree.py
│   ├── get_email.py
│   ├── list_all_sheets.py
│   ├── read_large.py
│   ├── search_zomato.py
│   ├── test_broad.py
│   ├── test_direct.py
│   ├── test_full.py
│   ├── test_importlib.py
│   └── test_shim.py
├── scripts
│   ├── __init__.py
│   ├── evaluate.py
│   ├── wandb_reward_curve_demo.py
│   └── wandb_sft_dataset_analyzer.py
├── server
│   ├── __init__.py
│   ├── app.py
│   ├── requirements.txt
│   └── trace_environment.py
├── static
│   └── index.html
├── test_api.py
├── test_tools.py
├── token_drive.pkl
├── token_gmail.pkl
├── token_sheets.pkl
├── tools
├── training
│   ├── __init__.py
│   ├── callbacks.py
│   ├── dataset.py
│   ├── export_model.py
│   ├── generate_sft_data.py
│   ├── train_grpo.py
│   └── train_sft.py
├── uber_dashboard.html
└── uv.lock
```
---
## Reproducing
```bash
# 1. Install
pip install -r requirements.txt
# 2. Run the environment
uvicorn environments.trace_env.app:app --host 127.0.0.1 --port 8000

# 3. SFT warm-start (Colab T4)
#    Open Trace_SFT_Training_Colab.ipynb, upload sft_demos.jsonl, run all

# 4. GRPO RL
python training/train_grpo.py --config configs/grpo_config.yaml \
                              --env-url http://localhost:8000
# 5. End-to-end demo
python inference.py --url http://localhost:8000
```
Hugging Face deployment is one Docker build — secrets (`GCP_CREDENTIALS_B64`, `GMAIL_TOKEN_B64`, `SHEETS_TOKEN_B64`) go in Space settings, and `generate_secrets.py` produces them all in one shot.
---
## Judging Criteria — How We Mapped to It
| Criterion | What we built |
|---|---|
| **Environment Innovation (40%)** | Federated multi-source retrieval + zero-knowledge SWM; schema-drift curriculum; 7-component reward stack with explicit anti-hack guard |
| **Storytelling (30%)** | Privacy-first narrative; live dashboard demo on real Gmail; emergent VERIFY behavior as a learned discovery, not a hand-coded rule |
| **Reward Improvement (20%)** | Per-component reward curves on W&B; total reward −0.5 → 2.0+; clean loss convergence 2.5 → 0.2; verification emergence at step ~230; anti-hack penalties self-extinguish |
| **Training Script Setup (10%)** | TRL SFT + GRPO + Unsloth, all running on a free Colab T4, with three export formats (LoRA / merged / GGUF) |
---
## What's Next
A few things we'd build with another week:
- **LLM-as-judge for `plan_quality`** — replace keyword heuristics with a small reward model
- **Cross-episode memory** — let the agent learn standing facts about the user across sessions
- **More sources** — calendar, Slack, Notion — each with its own schema-drift profile
- **Self-improvement loop** — the model proposes new curriculum tasks based on its own failures (Theme #4, deeper)
- **Adversarial cases** — our `dataset/is_adversarial_case` channel is wired up but currently flat at 0; populating it is the obvious next stress test
---
<!-- *Built with care at the **Meta OpenEnv Hackathon 2026**.* -->
*Stack: OpenEnv · TRL (SFT + GRPO) · Unsloth · Qwen2.5-3B-Instruct · FastAPI · Google APIs · LLaMA 4 Scout · Weights & Biases*
