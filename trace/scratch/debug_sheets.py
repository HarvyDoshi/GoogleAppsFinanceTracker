import os
import sys
from environments.trace_env.tools.sheets_tool import fetch_and_summarize

def debug_sheets():
    print("Fetching data from Sheets...")
    summary = fetch_and_summarize()
    print(f"Summary: {summary.get('count')} items found.")
    print(f"Total Spend: {summary.get('total_spend')}")
    
    transactions = summary.get("transactions", [])
    for i, t in enumerate(transactions):
        print(f"[{i}] {t['date']} | {t['vendor']} | {t['total']} | ID: {t['id']}")

if __name__ == "__main__":
    debug_sheets()
