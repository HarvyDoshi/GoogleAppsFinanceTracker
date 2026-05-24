import os
import pickle
import logging
from typing import Optional
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly"
]

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_CREDENTIALS_PATH = os.path.join(_PROJECT_ROOT, "credentials.json")
_TOKEN_PATH = os.path.join(_PROJECT_ROOT, "token_sheets.pkl")
_LEDGER_ID_PATH = os.path.join(_PROJECT_ROOT, ".ledger_id")

# We will use these columns for the ledger
HEADERS = ["Date", "Vendor", "Category", "Total", "Payment Method", "Order ID", "Notes", "Message ID"]


def get_sheets_service():
    """Authenticate and return the Google Sheets API service."""
    import base64
    # Reconstruct files from Hugging Face Secrets if available
    if os.environ.get("SHEETS_TOKEN_B64") and not os.path.exists(_TOKEN_PATH):
        try:
            with open(_TOKEN_PATH, "wb") as f:
                f.write(base64.b64decode(os.environ["SHEETS_TOKEN_B64"]))
        except Exception as e:
            logger.error(f"Failed to decode SHEETS_TOKEN_B64: {e}")
            
    if os.environ.get("GCP_CREDENTIALS_B64") and not os.path.exists(_CREDENTIALS_PATH):
        try:
            with open(_CREDENTIALS_PATH, "wb") as f:
                f.write(base64.b64decode(os.environ["GCP_CREDENTIALS_B64"]))
        except Exception as e:
            logger.error(f"Failed to decode GCP_CREDENTIALS_B64: {e}")

    creds = None
    if os.path.exists(_TOKEN_PATH):
        with open(_TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                os.remove(_TOKEN_PATH)
                creds = None

        if not creds:
            if not os.path.exists(_CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Missing {_CREDENTIALS_PATH}. Please download it from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(_CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(_TOKEN_PATH, "wb") as f:
            pickle.dump(creds, f)

    return build("sheets", "v4", credentials=creds)


def find_spreadsheet_by_name(service, name: str) -> Optional[str]:
    """Search for a spreadsheet by name and return its ID."""
    try:
        from googleapiclient.discovery import build
        drive_service = build('drive', 'v3', credentials=service._http.credentials)
        # Search for both native sheets and also handle dummy/xlsx naming
        query = f"mimeType='application/vnd.google-apps.spreadsheet' and name contains '{name}' and trashed = false"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        if files:
            return files[0]['id']
    except Exception as e:
        logger.warning(f"[SHEETS] Auto-discovery failed: {e}")
    return None


def get_or_create_ledger(service) -> str:
    """Return the existing ledger ID or create a new one with headers."""
    if os.environ.get("SHEETS_LEDGER_ID"):
        return os.environ.get("SHEETS_LEDGER_ID")
        
    if os.path.exists(_LEDGER_ID_PATH):
        with open(_LEDGER_ID_PATH, "r") as f:
            spreadsheet_id = f.read().strip()
            if spreadsheet_id:
                return spreadsheet_id

    # Auto-discovery attempt before creating new
    discovered_id = find_spreadsheet_by_name(service, "Trace_Financial_Audit_Dummy_Sheet")
    if not discovered_id:
        discovered_id = find_spreadsheet_by_name(service, "Trace Financial Ledger")
    
    if discovered_id:
        with open(_LEDGER_ID_PATH, "w") as f:
            f.write(discovered_id)
        return discovered_id

    # Create a new spreadsheet
    spreadsheet = {
        'properties': {
            'title': 'Trace Financial Ledger'
        }
    }
    spreadsheet = service.spreadsheets().create(body=spreadsheet, fields='spreadsheetId').execute()
    spreadsheet_id = spreadsheet.get('spreadsheetId')
    
    # Save ID
    with open(_LEDGER_ID_PATH, "w") as f:
        f.write(spreadsheet_id)
        
    # Write headers
    body = {
        'values': [HEADERS]
    }
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"A1:{chr(64 + len(HEADERS))}1",
        valueInputOption="RAW",
        body=body
    ).execute()
    
    # Optional: Make headers bold
    requests = [
        {
            "repeatCell": {
                "range": {
                    "sheetId": 0,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": len(HEADERS)
                },
                "cell": {
                    "userEnteredFormat": {
                        "textFormat": {
                            "bold": True
                        }
                    }
                },
                "fields": "userEnteredFormat.textFormat.bold"
            }
        }
    ]
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={'requests': requests}
    ).execute()
    
    logger.info(f"[SHEETS] Created new ledger. ID: {spreadsheet_id}")
    return spreadsheet_id


def append_transactions(transactions: list[dict]) -> str:
    """
    Append new transactions to the Google Sheet ledger with deduplication.
    We return the URL of the Google Sheet.
    """
    if not transactions:
        return ""
        
    try:
        service = get_sheets_service()
        spreadsheet_id = get_or_create_ledger(service)
        
        # ── Step 1: Fetch existing Message IDs to prevent duplicates ────────
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range="H:H" # Message ID is column H
        ).execute()
        existing_ids = {row[0] for row in result.get('values', []) if row}
        
        # ── Step 2: Prepare new rows ────────────────────────────────────────
        rows = []
        for t in transactions:
            msg_id = t.get("id")
            if not msg_id or msg_id in existing_ids:
                continue # Skip duplicates or missing IDs
                
            import re
            amount_str = re.sub(r'[^\d.]', '', t.get("total") or "")
            amount = float(amount_str) if amount_str else 0.0
            
            # If no amount was found AND no useful metadata exists, skip
            has_useful_metadata = (
                t.get("vendor", "").lower() not in ("", "unknown")
                or t.get("subject", "").strip()
                or t.get("body_preview", "").strip()
            )
            if amount == 0 and not has_useful_metadata:
                continue
            
            # Extract notes from details
            details = []
            if t.get("details"):
                for k, v in t["details"].items():
                    if isinstance(v, list):
                        details.append(f"{k}: {len(v)} items")
                    else:
                        details.append(f"{k}: {v}")
            # Fallback: include subject/body preview for unparsed entries
            if not details:
                subj = t.get("subject", "").strip()
                if subj:
                    details.append(subj[:120])
                elif t.get("body_preview", "").strip():
                    details.append(t["body_preview"][:120])
            notes_str = " | ".join(details)
            
            row = [
                t.get("date", ""),
                t.get("vendor", ""),
                t.get("category", ""),
                amount,
                t.get("payment_method", "Unknown"),
                t.get("order_id", ""),
                notes_str,
                msg_id
            ]
            rows.append(row)
            
        if not rows:
            logger.info("[SHEETS] No new transactions to append.")
            return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"

        body = {
            'values': rows
        }
        result = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range="A:H",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body=body
        ).execute()
        
        logger.info(f"[SHEETS] Appended {result.get('updates', {}).get('updatedRows', 0)} new rows.")
        return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        
    except Exception as e:
        logger.error(f"[SHEETS] Error appending to Sheets: {e}")
        return ""


def fetch_and_summarize() -> dict:
    """
    Fetch all rows from the Google Sheet and generate a summary.
    """
    try:
        service = get_sheets_service()
        spreadsheet_id = get_or_create_ledger(service)
        
        # Log metadata to help debugging
        spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        title = spreadsheet.get('properties', {}).get('title')
        logger.info(f"[SHEETS] Fetching from '{title}' (ID: {spreadsheet_id})")
        
        # Read the full range to ensure we get all data
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range="A:H"
        ).execute()
        
        values = result.get('values', [])
        logger.info(f"[SHEETS] Found {len(values)} rows (including header).")
        if not values or len(values) <= 1:
            logger.info("[SHEETS] Ledger is empty or only contains headers.")
            return {
                "total_spend": 0.0,
                "count": 0,
                "by_category": {},
                "by_vendor": {}
            }
            
        # Skip header row
        rows = values[1:]
        
        total_spend = 0.0
        by_category = {}
        by_vendor = {}
        count = 0
        transactions = [] # Added to return the actual data
        
        import re
        for row in rows:
            # Basic validation: must have at least Date, Vendor, Category, Total
            if len(row) < 4:
                continue
                
            vendor = row[1] or "Unknown"
            category = row[2] or "unknown"
            
            # Extract amount from Column D (index 3)
            amount_val = row[3]
            if not amount_val:
                continue
                
            try:
                # Remove everything except digits and decimal point
                cleaned_amt = re.sub(r'[^\d.]', '', str(amount_val))
                if not cleaned_amt:
                    amount = 0.0
                else:
                    amount = float(cleaned_amt)
            except (ValueError, IndexError) as e:
                logger.warning(f"[SHEETS] Failed to parse amount '{amount_val}': {e}")
                amount = 0.0
                
            # Get Message ID from Column H (index 7)
            msg_id = row[7] if len(row) > 7 else None
                
            if amount > 0:
                total_spend += amount
                by_category[category] = by_category.get(category, 0) + amount
                by_vendor[vendor] = by_vendor.get(vendor, 0) + amount
                count += 1
                
                # Add to transactions list
                transactions.append({
                    "date": row[0],
                    "vendor": vendor,
                    "category": category,
                    "total": str(amount),
                    "payment_method": row[4] if len(row) > 4 else "Unknown",
                    "order_id": row[5] if len(row) > 5 else "",
                    "notes": row[6] if len(row) > 6 else "",
                    "id": msg_id,
                    "_source": "sheets"
                })
            
        summary = {
            "total_spend": round(total_spend, 2),
            "count": count,
            "by_category": {k: round(v, 2) for k, v in sorted(by_category.items(), key=lambda x: -x[1])},
            "by_vendor": {k: round(v, 2) for k, v in sorted(by_vendor.items(), key=lambda x: -x[1])},
            "transactions": transactions, # Include the list
            "sheet_url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        }
        logger.info(f"[SHEETS] Fetched summary: ₹{total_spend:,.2f} across {count} items.")
        return summary
        
    except Exception as e:
        logger.error(f"[SHEETS] Error fetching from Sheets: {e}")
        return {"error": str(e), "total_spend": 0.0}
