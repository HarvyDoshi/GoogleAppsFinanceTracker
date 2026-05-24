"""
Trace Environment Inference Script.

Runs a simple hardcoded policy against the Trace environment server
to demonstrate the full episode lifecycle.

Usage:
    # Make sure server is running first:
    #   cd trace && uvicorn server.app:app --host 127.0.0.1 --port 8000
    
    python inference.py --url http://127.0.0.1:8000
"""

import logging
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(__file__))

from client import TraceEnvClient
from models import TraceAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_inference(base_url: str = "http://localhost:8000"):
    """
    Run a full inference episode against the Trace environment.
    """
    logger.info("Connecting to Trace environment...")

    with TraceEnvClient(base_url=base_url).sync() as client:
        # 1. Reset
        logger.info("Step 0: Resetting environment...")
        result = client.reset()
        obs = result.observation
        logger.info(f"  Instruction: {obs.instruction}")
        logger.info(f"  Sources: {obs.available_sources}")

        # 2. Plan
        logger.info("Step 1: Planning...")
        result = client.step(TraceAction(
            action_type="PLAN",
            content=(
                "Step 1: Fast text-first retrieval for transactional emails. "
                "Step 2: Deep retrieval for attachment-heavy mails (PDF/image/doc). "
                "Step 3: Retrieve historical transactions from Google Sheets ledger. "
                "Step 4: Export merged data into a local DOCX report."
            )
        ))
        logger.info(f"  Context: {result.observation.context}")

        # 3. Quick Retrieval (Text-first)
        logger.info("Step 2: Quick retrieval (Text-first)...")
        result = client.step(TraceAction(
            action_type="RETRIEVE",
            content="newer_than:180d (receipt OR invoice OR payment OR transaction OR order OR bill OR booking OR ride OR food)",
            source="gmail",
            metadata={"pass": "quick_text", "max_results": 50}
        ))
        logger.info(f"  Context: {result.observation.context}")

        # 4. Deep Retrieval (Attachments)
        logger.info("Step 3: Deep retrieval (Attachments)...")
        result = client.step(TraceAction(
            action_type="RETRIEVE",
            content="newer_than:180d (receipt OR invoice OR payment OR transaction OR order OR bill OR booking OR ride OR food)",
            source="gmail",
            metadata={"pass": "deep_attachments", "max_results": 50, "analyse_images": True}
        ))
        logger.info(f"  Context: {result.observation.context}")

        # Track Gmail-only summary for provenance
        gmail_summary = result.observation.metadata.get("transactions_summary", {})
        gmail_count = gmail_summary.get("count", 0)
        gmail_spend = gmail_summary.get("total_spend", 0.0)
        gmail_by_cat = gmail_summary.get("by_category", {})
        logger.info(f"  Gmail data: {gmail_count} transactions, ₹{gmail_spend:,.2f}")
        logger.info(f"  Gmail categories: {gmail_by_cat}")

        # 5. RETRIEVE from Google Sheets (gets historical data)
        #    The env merges Gmail retrieved_data with Sheets data automatically
        logger.info("Step 4: Retrieving historical data from Google Sheets...")
        result = client.step(TraceAction(
            action_type="RETRIEVE",
            content="Read all existing transactions from the ledger.",
            source="sheets"
        ))
        logger.info(f"  Context: {result.observation.context}")

        # Extract MERGED summary (Gmail + Sheets combined by the env)
        summary = result.observation.metadata.get("transactions_summary", {})
        total_spend = summary.get("total_spend", 0.0)
        tx_count = summary.get("count", 0)
        by_category = summary.get("by_category", {})
        top_category = next(iter(by_category.keys()), "unknown")
        top_category_spend = by_category.get(top_category, 0.0)
        sheets_new = result.observation.metadata.get("new_from_sheets", 0)
        logger.info(f"  Merged: {tx_count} total ({gmail_count} Gmail + {sheets_new} Sheets-only)")
        logger.info(f"  Combined categories: {by_category}")

        # 6. EXPORT all transactions to local DOCX
        logger.info("Step 5: Exporting transactions to DOCX report...")
        result = client.step(TraceAction(
            action_type="EXPORT",
            content="Export the combined financial transactions into a DOCX report."
        ))
        logger.info(f"  Context: {result.observation.context}")


        # 7. Memorize findings
        logger.info("Step 6: Memorizing...")
        result = client.step(TraceAction(
            action_type="MEMORIZE",
            content=(
                f"Consolidated {tx_count} transactions (Gmail: {gmail_count}, "
                f"Sheets-only: {sheets_new}). Total: ₹{total_spend:,.2f}. "
                f"Categories: {', '.join(f'{k}: ₹{v:,.2f}' for k, v in by_category.items())}. "
                f"Report generated locally."
            )
        ))

        # 8. Submit answer — includes both data sources
        logger.info("Step 7: Submitting answer...")
        cat_breakdown = ", ".join(f"{k}: ₹{v:,.2f}" for k, v in list(by_category.items())[:5])
        result = client.step(TraceAction(
            action_type="ANSWER",
            content=(
                f"Total spend: ₹{total_spend:,.2f} across {tx_count} transactions "
                f"(Gmail: {gmail_count}, Sheets historical: {sheets_new}). "
                f"Top category: {top_category} (₹{top_category_spend:,.2f}). "
                f"Breakdown: {cat_breakdown}. "
                "Summary: Financial audit complete — Gmail + Sheets data merged and local DOCX report generated."
            )
        ))

        logger.info(f"  Final Reward: {result.reward}")
        logger.info(f"  Done: {result.done}")

        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trace Environment Inference")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    run_inference(base_url=args.url)
