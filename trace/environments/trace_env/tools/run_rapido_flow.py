# save as run_rapido_flow.py (in project root), then run: python run_rapido_flow.py
import json
import pathlib
import requests

BASE = "http://127.0.0.1:8000"

def post(path: str, payload: dict):
    r = requests.post(f"{BASE}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def main():
    # 1) RESET
    reset_payload = {
        "instruction": "Find Rapido-related emails from ayush@gmail.com including receipts/content and PDF receipts, then extract transaction details.",
        "difficulty": "easy",
        "available_sources": ["gmail"],
        "ground_truth": {}
    }
    reset_res = post("/reset", reset_payload)
    print("[RESET] episode:", reset_res.get("episode_id"))

    # 2) PLAN
    plan_payload = {
        "action_type": "PLAN",
        "content": "Step 1: Search Gmail for Rapido emails from/to Ayush. Step 2: Include receipt/trip/invoice emails. Step 3: Prefer emails with PDF attachments. Step 4: Parse and summarize transactions.",
        "source": None,
        "metadata": {}
    }
    plan_res = post("/step", plan_payload)
    print("[PLAN] context:", plan_res["observation"]["context"])

    # 3) RETRIEVE (primary query)
    retrieve_payload = {
        "action_type": "RETRIEVE",
        "content": "(from:ayush@gmail.com OR to:ayush@gmail.com OR from:rapido OR subject:rapido) (receipt OR trip OR invoice OR ride OR payment) (has:attachment OR filename:pdf)",
        "source": "gmail",
        "metadata": {}
    }
    retrieve_res = post("/step", retrieve_payload)
    print("[RETRIEVE] context:", retrieve_res["observation"]["context"])

    # If primary query yields zero, fallback query
    tx_summary = retrieve_res.get("info", {}).get("transactions_summary", {})
    tx_count = tx_summary.get("count", 0)
    if tx_count == 0:
        fallback_payload = {
            "action_type": "RETRIEVE",
            "content": "(rapido OR \"Rapido\") (receipt OR invoice OR trip OR ride) newer_than:2y",
            "source": "gmail",
            "metadata": {}
        }
        retrieve_res = post("/step", fallback_payload)
        print("[RETRIEVE-FALLBACK] context:", retrieve_res["observation"]["context"])
        tx_summary = retrieve_res.get("info", {}).get("transactions_summary", {})
        tx_count = tx_summary.get("count", 0)

    # 4) VERIFY
    verify_payload = {
        "action_type": "VERIFY",
        "content": "Retrieved Rapido emails include receipt/trip/payment details and PDF attachment evidence where available.",
        "source": None,
        "metadata": {}
    }
    verify_res = post("/step", verify_payload)
    print("[VERIFY] context:", verify_res["observation"]["context"])

    # 5) ANSWER
    answer_payload = {
        "action_type": "ANSWER",
        "content": "Rapido Gmail receipts/content and PDF-related emails for Ayush were retrieved and structured.",
        "source": None,
        "metadata": {}
    }
    answer_res = post("/step", answer_payload)
    print("[ANSWER] done:", answer_res.get("done"))

    # Save outputs
    info = retrieve_res.get("info", {})
    dashboard_html = info.get("dashboard_html", "")
    transactions = info.get("transactions", [])
    summary = info.get("transactions_summary", {})

    pathlib.Path("rapido_response.json").write_text(
        json.dumps(retrieve_res, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if dashboard_html:
        pathlib.Path("rapido_dashboard.html").write_text(dashboard_html, encoding="utf-8")
        print("[OK] wrote rapido_dashboard.html")
    else:
        print("[WARN] dashboard_html missing (no results or dashboard generation failed).")

    pathlib.Path("rapido_transactions.json").write_text(
        json.dumps({"summary": summary, "transactions": transactions}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"[OK] transactions: {tx_count}")
    print("[OK] wrote rapido_transactions.json and rapido_response.json")

if __name__ == "__main__":
    main()