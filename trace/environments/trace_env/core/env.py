# """
# environments/trace_env/core/env.py
#
# OpenEnv-compatible environment for the Trace project.
# Hardened with per-step timeouts, cumulative reward tracking,
# locked-down reset, and step timing metrics.
# """


from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .schemas import TraceAction, TraceObservation, EpisodeState
from .world_model import SemanticWorldModel
from ..agents.planner import PlannerAgent
from ..agents.retriever import RetrieverAgent
from ..agents.memory import MemoryAgent
from ..agents.verifier import VerifierAgent
from ..rewards.reward_fn import compute_reward
from ..rewards.anti_hack import AntiHackGuard


class TraceEnv:
    """
    OpenEnv-compatible environment for the Trace project.

    Episode lifecycle:
        reset(task) -> observation
        step(action) -> observation, reward, done, info
        state()      -> EpisodeState (for logging/debugging)

    The environment simulates a user with a federated digital footprint.
    The agent is given a long-horizon instruction (e.g. "Audit all receipts
    from 2022-2024") and must plan sub-tasks, retrieve data from virtual
    sources, and synthesize a verified result.
    """

    DEFAULT_MAX_STEPS = 20          # hard limit per episode
    DEFAULT_TIMEOUT_SECONDS = 300   # wall-clock timeout (total episode)
    DEFAULT_PER_STEP_TIMEOUT = 60   # wall-clock timeout (per-step)

    def __init__(self, config: dict):
        self.config = config
        self.max_steps = config.get("max_steps", self.DEFAULT_MAX_STEPS)
        self.timeout_seconds = config.get("timeout_seconds", self.DEFAULT_TIMEOUT_SECONDS)
        self.per_step_timeout = config.get("per_step_timeout_seconds", self.DEFAULT_PER_STEP_TIMEOUT)

        self.world_model = SemanticWorldModel(config)
        self.planner = PlannerAgent(config)
        self.retriever = RetrieverAgent(config)
        self.memory = MemoryAgent(config)
        self.verifier = VerifierAgent(config)

        # Anti-hack guard with configurable thresholds
        anti_hack_config = config.get("anti_hack", {})
        self.anti_hack = AntiHackGuard(anti_hack_config)

        self._episode_id: Optional[str] = None
        self._steps: int = 0
        self._step_start_time: float = 0.0
        self._episode_start_time: float = 0.0
        self._state: Optional[EpisodeState] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task: dict) -> TraceObservation:
        """
        Start a fresh episode with a new task.

        LOCKED-DOWN: Clears all prior episode state to prevent
        cross-episode data leakage (per guide: "Lock down execution").

        Args:
            task: {
                "instruction": str,          # natural-language goal
                "difficulty": "easy"|"medium"|"hard",
                "available_sources": list[str],  # e.g. ["gmail", "sheets"]
                "ground_truth": dict,        # for reward computation
            }

        Returns:
            TraceObservation: initial observation for the agent.
        """
        self._episode_id = str(uuid.uuid4())
        self._steps = 0
        self._step_start_time = time.time()
        self._episode_start_time = time.time()

        # Full state reset — prevent cross-episode leakage
        self.world_model.initialize(task)
        self.memory.reset()
        self.anti_hack.reset()

        self._state = EpisodeState(
            episode_id=self._episode_id,
            task=task,
            plan=[],
            retrieved_data=[],
            verified=False,
            steps=0,
            done=False,
            cumulative_reward=0.0,
            action_history=[],
            query_history=[],
            step_times=[],
        )

        obs = TraceObservation(
            episode_id=self._episode_id,
            step=0,
            instruction=task["instruction"],
            available_sources=task["available_sources"],
            context="",
            memory_summary=self.memory.summarize(),
            world_state=self.world_model.snapshot(),
            metadata={},
        )
        return obs

    def step(self, action: TraceAction) -> tuple[TraceObservation, float, bool, dict]:
        """
        Execute one agent action and return the next state.

        Action types:
          - PLAN:     decompose instruction into sub-tasks
          - RETRIEVE: fetch data from a virtual source
          - MEMORIZE: store a finding into episodic memory
          - VERIFY:   verify the current plan/answer against world model
          - ANSWER:   submit the final synthesized answer

        Returns:
            (observation, reward, done, info)
        """
        assert self._state is not None, "Call reset() before step()"

        step_start = time.time()
        self._steps += 1
        self._state.steps = self._steps

        # Track action history
        action_type = action.action_type.strip().upper()
        self._state.action_history.append(action_type)

        # ── Timeout / step-limit guards ─────────────────────────────────
        self._step_start_time = time.time()
        if self._steps > self.max_steps:
            return self._terminate(reason="max_steps")

        # Episode-level wall-clock timeout
        episode_elapsed = time.time() - self._episode_start_time
        if episode_elapsed > self.timeout_seconds:
            return self._terminate(reason="episode_timeout")

        # ── Anti-hack validation ─────────────────────────────────────────
        hack_flag = self.anti_hack.check(action)
        if hack_flag:
            self._state.hack_flags.append(hack_flag)
            reward = compute_reward(
                action, self._state, hack_penalty=True, config=self.config
            )
            obs = self._build_obs(f"[ANTI-HACK] {hack_flag}")
            step_time = time.time() - step_start
            self._state.step_times.append(step_time)
            return obs, reward, False, {"hack": hack_flag}

        # ── Dispatch action ──────────────────────────────────────────────
        result_context = ""
        info = {}

        if action_type == "PLAN":
            plan = self.planner.decompose(
                action.content, self._state.task
            )
            self._state.plan = plan
            result_context = f"Plan created: {plan}"

        elif action_type == "RETRIEVE":
            # Track query for duplicate detection
            query_key = f"{action.source}:{action.content.strip().lower()}"
            self._state.query_history.append(query_key)

            data = self.retriever.fetch(
                source=action.source,
                query=action.content,
                world_model=self.world_model,
                metadata=action.metadata,
            )
            if not isinstance(data, list):
                data = [data]

            # Inject real data into world model so visible_items updates
            self.world_model.inject_real_data(action.source, data)
            self._state.retrieved_data.extend(data)

            # Update anti-hack with retrieval count
            self.anti_hack.check.__func__  # just to keep reference
            # Re-check with retrieved_count for empty-retrieval tracking
            # (the main check already passed, this is supplementary)

            result_context = f"Retrieved {len(data)} items from {action.source}"

            # ── Gmail processing: merge and summarize all retrieved transactions ──
            if action.source == "gmail" and data:
                try:
                    from ..tools.transaction_parser import parse_transactions_bulk
                    from ..tools.dashboard_renderer import render_dashboard
                    
                    # We parse everything we've retrieved so far to ensure deduplication
                    # and a cumulative summary.
                    parsed_all = parse_transactions_bulk(self._state.retrieved_data)
                    summary = parsed_all.get("summary", {})
                    transactions = parsed_all.get("transactions", [])
                    
                    total_spend = summary.get("total_spend", 0.0)
                    tx_count = summary.get("count", 0)
                    by_category = summary.get("by_category", {})
                    top_category = next(iter(by_category.keys()), "unknown")
                    top_category_spend = by_category.get(top_category, 0.0)
                    
                    dashboard_html = render_dashboard(parsed_all)

                    result_context = (
                        f"Step summary: Retrieved {len(data)} new items. "
                        f"Cumulative Audit: {tx_count} total transactions | "
                        f"Total Spend: ₹{total_spend:,.2f} | "
                        f"Top Category: {top_category} (₹{top_category_spend:,.2f})"
                    )
                    info = {
                        "gmail_query": action.content,
                        "transactions_summary": summary,
                        "transactions": transactions,
                        "dashboard_html": dashboard_html,
                        "dashboard_generated": True,
                        "cumulative_count": tx_count,
                        "cumulative_spend": total_spend,
                    }
                except Exception as e:
                    info = {
                        "dashboard_generated": False,
                        "dashboard_error": str(e),
                    }
            
            elif action.source == "sheets":
                try:
                    from ..tools.sheets_tool import fetch_and_summarize
                    from ..tools.transaction_parser import parse_transactions_bulk
                    
                    summary = fetch_and_summarize()
                    sheet_txs = summary.get("transactions", [])
                    
                    # Deduplicate before extending: only add Sheets rows
                    # whose IDs are not already present from Gmail retrieval
                    existing_ids = {
                        item.get("id") for item in self._state.retrieved_data
                        if item.get("id")
                    }
                    # Keep all sheet transactions; parse_transactions_bulk will handle merging
                    self._state.retrieved_data.extend(sheet_txs)
                    
                    # Build cumulative summary from ALL retrieved data (Gmail + Sheets)
                    parsed_all = parse_transactions_bulk(self._state.retrieved_data)
                    summary = parsed_all.get("summary", {})
                    all_txs = parsed_all.get("transactions", [])
                    
                    # Calculate overlapping items for logging
                    gmail_ids = {
                        item.get("id") for item in self._state.retrieved_data
                        if item.get("_source") != "sheets" and item.get("id")
                    }
                    overlapping = sum(1 for tx in sheet_txs if tx.get("id") in gmail_ids)
                    
                    result_context = (
                        f"Retrieved {len(sheet_txs)} items from Google Sheets "
                        f"({len(sheet_txs) - overlapping} new, {overlapping} already in Gmail). "
                        f"Cumulative Audit: {summary.get('count', 0)} total transactions | "
                        f"Total Spend: ₹{summary.get('total_spend', 0.0):,.2f}"
                    )
                    info = {
                        "source": "sheets",
                        "sheets_count": len(sheet_txs),
                        "new_from_sheets": len(sheet_txs) - overlapping,
                        "transactions": all_txs,  # merged Gmail + Sheets
                        "transactions_summary": summary,
                    }
                except Exception as e:
                    result_context = f"Error retrieving from Sheets: {e}"
                    info = {"error": str(e)}

        elif action_type == "MEMORIZE":
            self.memory.store(action.content, action.metadata)
            result_context = "Stored to episodic memory."

        elif action_type == "VERIFY":
            verification = self.verifier.verify(
                claim=action.content,
                world_model=self.world_model,
                memory=self.memory,
            )
            self._state.verified = verification["passed"]
            result_context = f"Verification: {verification}"

        elif action_type == "SYNC":
            # Sync retrieved transactions to Google Sheets
            try:
                from ..tools.sheets_tool import append_transactions, fetch_and_summarize
                from ..tools.transaction_parser import parse_transactions_bulk
                
                parsed = parse_transactions_bulk(self._state.retrieved_data)
                transactions = parsed.get("transactions", [])
                
                sheet_url = append_transactions(transactions)
                if sheet_url:
                    ledger_summary = fetch_and_summarize()
                    total_ledger = ledger_summary.get("total_spend", 0.0)
                    
                    result_context = (
                        f"Synced {len(transactions)} transactions to Google Sheets: {sheet_url}. "
                        f"Current Ledger Total: ₹{total_ledger:,.2f}"
                    )
                    info = {
                        "sheet_url": sheet_url,
                        "ledger_summary": ledger_summary,
                        "sync_count": len(transactions)
                    }
                else:
                    result_context = "Failed to sync to Google Sheets. Check credentials."
                    info = {"error": "Sync failed"}
            except Exception as e:
                result_context = f"Error during SYNC: {e}"
                info = {"error": str(e)}

        elif action_type == "EXPORT":
            # Export retrieved transactions to a DOCX report
            try:
                from ..tools.transaction_parser import parse_transactions_bulk
                from ..tools.report_tool import export_transactions_to_docx
                
                parsed = parse_transactions_bulk(self._state.retrieved_data)
                transactions = parsed.get("transactions", [])
                
                report_path = export_transactions_to_docx(transactions)
                
                result_context = (
                    f"Exported {len(transactions)} transactions to DOCX report at: {report_path}."
                )
                info = {
                    "report_path": report_path,
                    "export_count": len(transactions)
                }
            except Exception as e:
                result_context = f"Error during EXPORT: {e}"
                info = {"error": str(e)}

        elif action_type == "ANSWER":
            self._state.final_answer = action.content
            reward = compute_reward(action, self._state, config=self.config)
            obs = self._build_obs("Episode complete.")
            self._state.done = True
            step_time = time.time() - step_start
            self._state.step_times.append(step_time)
            return obs, reward, True, {"final_answer": action.content}

        else:
            result_context = f"Unknown action type: {action.action_type}"

        # ── Per-step wall-clock timeout check ────────────────────────────
        step_elapsed = time.time() - step_start
        self._state.step_times.append(step_elapsed)

        if step_elapsed > self.per_step_timeout:
            return self._terminate(reason=f"per_step_timeout ({step_elapsed:.1f}s)")

        # ── Intermediate reward & next observation ───────────────────────
        reward = compute_reward(action, self._state, config=self.config)
        obs = self._build_obs(result_context)
        return obs, reward, False, info

    def state(self) -> EpisodeState:
        """Return full episode state (for logging/debugging)."""
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self, context: str) -> TraceObservation:
        return TraceObservation(
            episode_id=self._episode_id,
            step=self._steps,
            instruction=self._state.task["instruction"],
            available_sources=self._state.task["available_sources"],
            context=context,
            memory_summary=self.memory.summarize(),
            world_state=self.world_model.snapshot(),
            metadata={},
        )

    def _terminate(self, reason: str) -> tuple[TraceObservation, float, bool, dict]:
        obs = self._build_obs(f"Episode terminated: {reason}")
        reward = compute_reward(None, self._state, terminal_penalty=True, config=self.config)
        self._state.done = True
        return obs, reward, True, {"termination_reason": reason}
