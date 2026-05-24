import json
import re
import argparse
import csv
from pathlib import Path
import requests
from requests.exceptions import RequestException
from datetime import datetime

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
QUERY_PRESETS = {
    "latest": "",  # Unfiltered: just get the latest emails
    "financial": (
        "(receipt OR invoice OR payment OR transaction OR booking OR order OR trip OR "
        "ride OR bill OR statement OR subscription OR recharge OR GST OR tax invoice)"
    ),
    "rides": "(uber OR rapido OR ola OR trip OR ride OR fare OR receipt OR invoice)",
    "shopping": "(amazon OR flipkart OR myntra OR order OR invoice OR payment OR receipt)",
}

from client import TraceEnvClient
from models import TraceAction

def post(base_url: str, path: str, payload: dict) -> dict:
    # Use separate connect/read timeout so hangs fail faster.
    r = requests.post(f"{base_url}{path}", json=payload, timeout=(5, 600))
    r.raise_for_status()
    return r.json()


def ensure_api_ready(base_url: str) -> None:
    """Fail fast if FastAPI is down or hung. Increased timeout for slow laptops."""
    try:
        # Increase timeout to 30s to allow for cold starts or heavy background tasks
        r = requests.get(f"{base_url}/health", timeout=(5, 30))
        r.raise_for_status()
    except RequestException as e:
        raise RuntimeError(
            f"FastAPI is not ready at {base_url}. "
            "Start/restart server first: "
            "uvicorn environments.trace_env.app:app --host 127.0.0.1 --port 8000 --reload"
        ) from e


def _amount_to_float(amount_str: str | None) -> float:
    """Convert currency-like strings to float while preserving decimals."""
    if not amount_str:
        return 0.0
    cleaned = re.sub(r"[^\d.]", "", amount_str)
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def _build_summary(transactions: list[dict]) -> dict:
    total_spend = 0.0
    by_category = {}
    by_vendor = {}

    for tx in transactions:
        amount = _amount_to_float(tx.get("total"))
        if amount <= 0:
            continue
        total_spend += amount
        category = tx.get("category", "unknown")
        vendor = tx.get("vendor", "Unknown")
        by_category[category] = by_category.get(category, 0.0) + amount
        by_vendor[vendor] = by_vendor.get(vendor, 0.0) + amount

    return {
        "total_spend": round(total_spend, 2),
        "count": len(transactions),
        "by_category": {k: round(v, 2) for k, v in sorted(by_category.items(), key=lambda x: -x[1])},
        "by_vendor": {k: round(v, 2) for k, v in sorted(by_vendor.items(), key=lambda x: -x[1])},
    }


def _canonical_date(date_value: str | None) -> str | None:
    if not date_value:
        return None
    date_value = date_value.strip()
    known_formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%d %b %Y",
    ]
    for fmt in known_formats:
        try:
            dt = datetime.strptime(date_value, fmt)
            return dt.date().isoformat()
        except ValueError:
            continue
    return None


def _extract_evidence_flags(tx: dict) -> dict:
    doc_analyses = tx.get("doc_analyses", []) or []
    img_analyses = tx.get("image_analyses", []) or []
    attachment_count = tx.get("attachment_count", 0) or 0
    return {
        "has_body_text": bool(tx.get("body_preview")),
        "has_doc_analysis": len(doc_analyses) > 0,
        "has_image_analysis": len(img_analyses) > 0,
        "has_attachments": attachment_count > 0 or len(doc_analyses) > 0 or len(img_analyses) > 0,
    }


def _normalize_transaction(tx: dict) -> dict:
    normalized_amount = _amount_to_float(tx.get("total"))
    normalized_date = _canonical_date(tx.get("date"))
    evidence = _extract_evidence_flags(tx)
    confidence = 0.3
    if normalized_amount > 0:
        confidence += 0.3
    if normalized_date:
        confidence += 0.2
    if evidence["has_doc_analysis"] or evidence["has_image_analysis"]:
        confidence += 0.2
    tx["normalized_amount"] = round(normalized_amount, 2)
    tx["normalized_date"] = normalized_date
    tx["evidence"] = evidence
    tx["confidence"] = round(min(confidence, 1.0), 2)
    return tx


def _merge_transactions(*transaction_lists: list[dict]) -> list[dict]:
    merged = {}
    for tx_list in transaction_lists:
        for tx in tx_list:
            tx = _normalize_transaction(dict(tx))
            # Dedupe by stable key to avoid counting same email twice across passes.
            dedupe_key = tx.get("id") or (
                f"{tx.get('vendor', 'unknown').lower()}|"
                f"{tx.get('normalized_date') or tx.get('date', '')}|"
                f"{tx.get('normalized_amount', 0.0):.2f}|"
                f"{tx.get('subject', '').strip().lower()[:80]}"
            )
            tx["dedupe_key"] = dedupe_key
            merged[dedupe_key] = tx
    return list(merged.values())


def _filter_by_vendor(transactions: list[dict], vendor_filter: str) -> list[dict]:
    if not vendor_filter:
        return transactions
    needle = vendor_filter.strip().lower()
    return [
        tx for tx in transactions
        if needle in (tx.get("vendor", "").lower())
        or needle in (tx.get("from_email", "").lower())
        or needle in (tx.get("subject", "").lower())
    ]


def _filter_by_category(transactions: list[dict], category_filter: list[str] | str) -> list[dict]:
    if not category_filter:
        return transactions
    if isinstance(category_filter, str):
        categories = [category_filter.strip().lower()]
    else:
        categories = [c.strip().lower() for c in category_filter]
    return [tx for tx in transactions if (tx.get("category", "").lower() in categories)]


def _filter_by_min_amount(transactions: list[dict], min_amount: float) -> list[dict]:
    if min_amount <= 0:
        return transactions
    return [tx for tx in transactions if float(tx.get("normalized_amount", 0.0) or 0.0) >= min_amount]


def _write_transactions_csv(path: str, transactions: list[dict]) -> None:
    fieldnames = [
        "id",
        "vendor",
        "category",
        "date",
        "normalized_date",
        "total",
        "normalized_amount",
        "payment_method",
        "reimbursable",
        "confidence",
        "has_body_text",
        "has_doc_analysis",
        "has_image_analysis",
        "has_attachments",
        "subject",
        "from_email",
        "dedupe_key",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tx in transactions:
            evidence = tx.get("evidence", {}) or {}
            writer.writerow({
                "id": tx.get("id", ""),
                "vendor": tx.get("vendor", ""),
                "category": tx.get("category", ""),
                "date": tx.get("date", ""),
                "normalized_date": tx.get("normalized_date", ""),
                "total": tx.get("total", ""),
                "normalized_amount": tx.get("normalized_amount", ""),
                "payment_method": tx.get("payment_method", ""),
                "reimbursable": tx.get("reimbursable", False),
                "confidence": tx.get("confidence", ""),
                "has_body_text": evidence.get("has_body_text", False),
                "has_doc_analysis": evidence.get("has_doc_analysis", False),
                "has_image_analysis": evidence.get("has_image_analysis", False),
                "has_attachments": evidence.get("has_attachments", False),
                "subject": tx.get("subject", ""),
                "from_email": tx.get("from_email", ""),
                "dedupe_key": tx.get("dedupe_key", ""),
            })


def main():
    parser = argparse.ArgumentParser(
        description="Fast + deep Gmail financial transaction summarizer."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FastAPI base URL")
    parser.add_argument("--days", type=int, default=90, help="Lookback window in days")
    parser.add_argument("--limit", type=int, default=10, help="Max results per retrieval pass")
    parser.add_argument(
        "--mode",
        choices=["fast", "deep"],
        default="fast",
        help="fast=text first + optional deep; deep=always run attachment-heavy pass",
    )
    parser.add_argument(
        "--preset",
        choices=["latest", "financial", "rides", "shopping"],
        default="financial",
        help="Query preset for retrieval focus",
    )
    parser.add_argument(
        "--vendor",
        default="",
        help="Optional vendor filter (e.g. uber, rapido, amazon)",
    )
    parser.add_argument(
        "--category",
        nargs="+",
        default=[],
        help="Optional exact category filter (e.g. ride food shopping travel)",
    )
    parser.add_argument(
        "--min-amount",
        type=float,
        default=0.0,
        help="Optional minimum normalized amount filter",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip AI image analysis inside PDFs for 10x speedup",
    )
    parser.add_argument(
        "--sync-sheets",
        action="store_true",
        help="Append new transactions to a Google Sheet ledger and summarize from it",
    )
    args = parser.parse_args()
    ensure_api_ready(args.base_url)

    with TraceEnvClient(base_url=args.base_url).sync() as client:
        # 1) RESET
        reset_res = client.reset()
        episode_id = reset_res.observation.episode_id
        print("[RESET] episode_id:", episode_id)

        # 2) PLAN
        plan_payload = TraceAction(
            action_type="PLAN",
            content=(
                "Step 1: Fast text-first retrieval for transactional emails. "
                "Step 2: Deep retrieval for attachment-heavy mails (PDF/image/doc). "
                "Step 3: Parse and merge all transactions. "
                "Step 4: Aggregate by category/vendor and generate dashboard."
            )
        )
        plan_res = client.step(plan_payload)
        print("[PLAN] ", plan_res.observation.context)

        preset_query = QUERY_PRESETS[args.preset]

        # 3A) RETRIEVE (fast text-first pass)
        quick_query = (
            f"newer_than:{args.days}d "
            f"{preset_query} "
            "-category:promotions -in:chats"
        )
        retrieve_payload_quick = TraceAction(
            action_type="RETRIEVE",
            content=quick_query,
            source="gmail",
            metadata={"max_results": args.limit, "pass": "quick_text"}
        )
        retrieve_quick_res = client.step(retrieve_payload_quick)
        quick_info = retrieve_quick_res.observation.metadata
        quick_tx = quick_info.get("transactions", [])
        print(f"[RETRIEVE-QUICK] Found {len(quick_tx)} emails. (Text-only pass complete)")

        # 3B) RETRIEVE (deep attachment pass: PDF/image/doc)
        retrieve_deep_res = None
        run_deep = (args.mode == "deep")
        if run_deep:
            deep_query = quick_query
            retrieve_payload_deep = TraceAction(
                action_type="RETRIEVE",
                content=deep_query,
                source="gmail",
                metadata={
                    "max_results": args.limit, 
                    "pass": "deep_attachments",
                    "analyse_images": not args.skip_images
                },
            )
            print(f"[RETRIEVE-DEEP]  Processing {len(quick_tx)} emails with attachments in parallel. Please wait...")
            retrieve_deep_res = client.step(retrieve_payload_deep)
            print("[RETRIEVE-DEEP] ", retrieve_deep_res.observation.context)

        # 4) MERGE + BUILD FINAL SUMMARY/DASHBOARD
        quick_info = retrieve_quick_res.observation.metadata
        deep_info = retrieve_deep_res.observation.metadata if retrieve_deep_res else {}
    quick_transactions = quick_info.get("transactions", [])
    deep_transactions = deep_info.get("transactions", [])

    # DEBUG: Show what the deep pass actually returned
    print(f"\n[DEBUG] Quick pass: {len(quick_transactions)} tx, Deep pass: {len(deep_transactions)} tx")
    for i, tx in enumerate(deep_transactions):
        doc_count = len(tx.get("doc_analyses", []) or [])
        img_count = len(tx.get("image_analyses", []) or [])
        att_count = len(tx.get("attachments", []) or [])
        total = tx.get("total", "NONE")
        vendor = tx.get("vendor", "?")
        subj = tx.get("subject", "")[:50]
        has_body = "yes" if tx.get("body_preview", "") else "no"
        print(f"  [{i}] {vendor:15s} | total={str(total):10s} | docs={doc_count} imgs={img_count} atts={att_count} body={has_body} | {subj}")
    print()

    all_transactions = _merge_transactions(quick_transactions, deep_transactions)
    all_transactions = _filter_by_vendor(all_transactions, args.vendor)
    all_transactions = _filter_by_category(all_transactions, args.category)
    all_transactions = _filter_by_min_amount(all_transactions, args.min_amount)
    
    # Optional Google Sheets Integration
    sheet_url = None
    gmail_summary = _build_summary(all_transactions)
    final_summary = gmail_summary  # start with Gmail data
    
    if args.sync_sheets:
        try:
            from environments.trace_env.tools.sheets_tool import append_transactions, fetch_and_summarize
            print("\n[SHEETS] Syncing to Google Sheets...")
            sheet_url = append_transactions(all_transactions)
            if sheet_url:
                print(f"[SHEETS] Appended transactions. Ledger: {sheet_url}")
            
            print("[SHEETS] Fetching historical data to generate Global Summary...")
            global_summary = fetch_and_summarize()
            if global_summary and global_summary.get("count", 0) > 0:
                # ── MERGE both summaries instead of replacing ──────────
                # Collect Sheets-only transactions (not already in Gmail set)
                sheets_txs = global_summary.get("transactions", [])
                gmail_ids = {tx.get("id") for tx in all_transactions if tx.get("id")}
                sheets_only_txs = [
                    tx for tx in sheets_txs
                    if not tx.get("id") or tx["id"] not in gmail_ids
                ]
                if sheets_only_txs:
                    print(f"[SHEETS] Found {len(sheets_only_txs)} historical transactions not in Gmail.")
                    normalized_sheets = [_normalize_transaction(dict(tx)) for tx in sheets_only_txs]
                    all_transactions.extend(normalized_sheets)
                
                # Rebuild summary from the combined transaction list
                final_summary = _build_summary(all_transactions)
                final_summary["gmail_count"] = gmail_summary["count"]
                final_summary["gmail_spend"] = gmail_summary["total_spend"]
                final_summary["sheets_historical_count"] = len(sheets_only_txs)
                final_summary["sheet_url"] = global_summary.get("sheet_url", "")
                print(f"[SHEETS] Merged summary: {gmail_summary['count']} Gmail + "
                      f"{len(sheets_only_txs)} Sheets-only = {final_summary['count']} total")
        except Exception as e:
            print(f"[WARN] Failed to sync with Google Sheets: {e}")

    parsed_final = {"transactions": all_transactions, "summary": final_summary}
    if sheet_url:
        parsed_final["sheet_url"] = sheet_url
        
    try:
        from environments.trace_env.tools.dashboard_renderer import render_dashboard
        dashboard_html = render_dashboard(parsed_final)
    except Exception as e:
        dashboard_html = ""
        print("[WARN] dashboard render failed:", e)

    final_payload = {
        "mode": args.mode,
        "preset": args.preset,
        "vendor_filter": args.vendor or None,
        "category_filter": args.category or None,
        "min_amount_filter": args.min_amount if args.min_amount > 0 else None,
        "quick_query": quick_query,
        "deep_query": deep_query if run_deep else None,
        "quick_observation": retrieve_quick_res.observation.context,
        "deep_observation": retrieve_deep_res.observation.context if retrieve_deep_res else "skipped",
        "transactions_summary": final_summary,
        "transactions": all_transactions,
    }

    # 5) SAVE OUTPUTS
    Path("all_financial_response.json").write_text(
        json.dumps(
            {
                "quick_pass": {
                    "observation": retrieve_quick_res.observation.model_dump(),
                    "reward": retrieve_quick_res.reward,
                    "done": retrieve_quick_res.done,
                },
                "deep_pass": {
                    "observation": retrieve_deep_res.observation.model_dump() if retrieve_deep_res else {"context": "skipped"},
                    "reward": retrieve_deep_res.reward if retrieve_deep_res else 0.0,
                    "done": retrieve_deep_res.done if retrieve_deep_res else False,
                },
                "final": final_payload,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8"
    )

    Path("all_financial_transactions.json").write_text(
        json.dumps(final_payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    _write_transactions_csv("all_financial_transactions.csv", all_transactions)

    if dashboard_html:
        Path("all_financial_dashboard.html").write_text(dashboard_html, encoding="utf-8")
        print("[OK] all_financial_dashboard.html created")
    else:
        print("[WARN] dashboard_html missing in response")

    print("[OK] all_financial_response.json created")
    print("[OK] all_financial_transactions.json created")
    print("[OK] all_financial_transactions.csv created")
    print("[SUMMARY]", final_summary)
    print("[INFO] total unique transactions:", len(all_transactions))


if __name__ == "__main__":
    main()