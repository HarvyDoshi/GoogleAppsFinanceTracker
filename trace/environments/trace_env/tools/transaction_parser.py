"""
environments/trace_env/tools/transaction_parser.py

Universal Transaction Parser for Trace.

Detects and parses ALL types of transactional Gmail:
  - Ride receipts      (Uber, Ola, Rapido)
  - Food orders        (Swiggy, Zomato, Dunzo)
  - Shopping           (Amazon, Flipkart, Meesho, Myntra)
  - Payments           (PhonePe, GPay, Paytm, NEFT/IMPS)
  - Subscriptions      (Netflix, Spotify, YouTube, SaaS)
  - Travel             (IRCTC, MakeMyTrip, Air India, IndiGo)
  - Banking            (Bank alerts, EMI, credit card statements)
  - Utility bills      (Electricity, gas, water, telecom)
  - Education          (Udemy, Coursera, college fees)
  - Healthcare         (Pharmacy, hospital, insurance)

Works on:
  - Gmail body_text  (plaintext extracted by gmail_tool)
  - image_analyses   (from image_tool Ollama pipeline)
  - doc_analyses     (from doc_tool Docling pipeline)
"""

from __future__ import annotations
import re
from datetime import datetime
from typing import Optional


# ── Vendor → Category mapping ─────────────────────────────────────────────────

VENDOR_MAP = {
    # Rides
    "uber": ("ride", "Uber"),
    "ola": ("ride", "Ola"),
    "rapido": ("ride", "Rapido"),
    "namma yatri": ("ride", "Namma Yatri"),
    # Food
    "swiggy": ("food", "Swiggy"),
    "zomato": ("food", "Zomato"),
    "dunzo": ("food", "Dunzo"),
    "blinkit": ("food", "Blinkit"),
    "zepto": ("food", "Zepto"),
    # Shopping
    "amazon": ("shopping", "Amazon"),
    "flipkart": ("shopping", "Flipkart"),
    "myntra": ("shopping", "Myntra"),
    "meesho": ("shopping", "Meesho"),
    "ajio": ("shopping", "AJIO"),
    "nykaa": ("shopping", "Nykaa"),
    "snapdeal": ("shopping", "Snapdeal"),
    # Payments
    "phonepe": ("payment", "PhonePe"),
    "gpay": ("payment", "Google Pay"),
    "google pay": ("payment", "Google Pay"),
    "paytm": ("payment", "Paytm"),
    "bhim": ("payment", "BHIM UPI"),
    "neft": ("payment", "NEFT Transfer"),
    "imps": ("payment", "IMPS Transfer"),
    "upi": ("payment", "UPI Transfer"),
    # Subscriptions
    "netflix": ("subscription", "Netflix"),
    "spotify": ("subscription", "Spotify"),
    "youtube": ("subscription", "YouTube Premium"),
    "hotstar": ("subscription", "Disney+ Hotstar"),
    "prime": ("subscription", "Amazon Prime"),
    "notion": ("subscription", "Notion"),
    "github": ("subscription", "GitHub"),
    "openai": ("subscription", "OpenAI"),
    "anthropic": ("subscription", "Anthropic"),
    # Travel
    "irctc": ("travel", "IRCTC"),
    "makemytrip": ("travel", "MakeMyTrip"),
    "indigo": ("travel", "IndiGo"),
    "air india": ("travel", "Air India"),
    "goibibo": ("travel", "Goibibo"),
    "cleartrip": ("travel", "Cleartrip"),
    "agoda": ("travel", "Agoda"),
    "booking.com": ("travel", "Booking.com"),
    "airbnb": ("travel", "Airbnb"),
    "oyo": ("travel", "OYO Rooms"),
    "fabhotels": ("travel", "FabHotels"),
    "taj": ("travel", "Taj Hotels"),
    "marriott": ("travel", "Marriott"),
    "vistara": ("travel", "Vistara"),
    "spicejet": ("travel", "SpiceJet"),
    "akasa": ("travel", "Akasa Air"),
    "emirates": ("travel", "Emirates"),
    # Banking
    "hdfc": ("banking", "HDFC Bank"),
    "icici": ("banking", "ICICI Bank"),
    "sbi": ("banking", "SBI"),
    "axis": ("banking", "Axis Bank"),
    "kotak": ("banking", "Kotak Bank"),
    "idfc": ("banking", "IDFC Bank"),
    # Utilities
    "bescom": ("utility", "BESCOM"),
    "airtel": ("utility", "Airtel"),
    "jio": ("utility", "Jio"),
    "vi ": ("utility", "Vi"),
    "bsnl": ("utility", "BSNL"),
    # Education
    "udemy": ("education", "Udemy"),
    "coursera": ("education", "Coursera"),
    "unacademy": ("education", "Unacademy"),
    "byju": ("education", "BYJU'S"),
    # Healthcare
    "apollo": ("healthcare", "Apollo Pharmacy"),
    "1mg": ("healthcare", "1mg"),
    "practo": ("healthcare", "Practo"),
}

# ── Category display config ───────────────────────────────────────────────────

CATEGORY_CONFIG = {
    "ride":         {"icon": "🚗", "color": "#1a1a1a",  "label": "Ride"},
    "food":         {"icon": "🍔", "color": "#e85d04",  "label": "Food & Delivery"},
    "shopping":     {"icon": "🛍️", "color": "#7c6fcd",  "label": "Shopping"},
    "payment":      {"icon": "💸", "color": "#1a7a4a",  "label": "Payment"},
    "subscription": {"icon": "🔄", "color": "#0077b6",  "label": "Subscription"},
    "travel":       {"icon": "✈️", "color": "#6d4c41",  "label": "Travel"},
    "banking":      {"icon": "🏦", "color": "#b5179e",  "label": "Banking"},
    "utility":      {"icon": "⚡", "color": "#4a6741",  "label": "Utility Bill"},
    "education":    {"icon": "📚", "color": "#e63946",  "label": "Education"},
    "healthcare":   {"icon": "💊", "color": "#d62828",  "label": "Healthcare"},
    "unknown":      {"icon": "📧", "color": "#6b6560",  "label": "Transactional"},
}


def detect_category(text: str) -> tuple[str, str]:
    """Detect transaction category and vendor from email text."""
    text_lower = text.lower()
    for keyword, (category, vendor) in VENDOR_MAP.items():
        if keyword in text_lower:
            return category, vendor
    return "unknown", "Unknown"


def extract_amounts(text: str) -> list[str]:
    """Extract all monetary amounts from text."""
    patterns = [
        r'(?:₹|Rs\.?|INR|USD|\$|€|£)\s?[-\u2010\u2212]?\s*[\d,]+(?:\.\d{1,2})?',
        r'[-\u2010\u2212]?\s*[\d,]+(?:\.\d{1,2})?\s*(?:₹|Rs\.?|INR)',
        r'(?:Total(?: Amount)?|Grand Total|Amount Paid|You paid|Net Amount|Final Amount|Total Fare|Amount|Total Payable)\b.{0,100}?[-\u2010\u2212]?\s*([\d,]+(?:\.\d{1,2})?)\b',
    ]
    amounts = []
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        amounts.extend(found)
    return list(dict.fromkeys(amounts))[:10]  # deduplicate, max 10


def extract_total(text: str, structured_total: Optional[str] = None) -> Optional[str]:
    """
    Extract the final total amount. Priority:
    1. Explicit 'Grand Total' / 'Total Payable' label (highest confidence)
    2. Structured tool output (from VLM/Docling JSON)
    3. Amounts inside Docling Markdown table pipes
    4. Largest Rs/INR amount in text (fallback)
    """
    # Priority 1: Labelled totals - find label then grab number within 150 chars
    labelled_patterns = [
        r'(?:Grand\s+Total|Total\s+Payable|Total\s+Amount\s+Paid|Amount\s+Paid|Net\s+Amount|You\s+Paid|Paid|Sent|Payment\s+of|Transaction\s+of|Final\s+Amount|Total\s+Fare|Total\s+Due|Amount\s+Due|Invoice\s+Total)\b.{0,150}?(?:[\u20b9]|INR|Rs\.?|₹)\s*([\d,]+(?:\.\d{1,2})?)',
        r'(?:Grand\s+Total|Total\s+Payable|Total\s+Amount\s+Paid|Amount\s+Paid|Net\s+Amount|You\s+Paid|Paid|Sent|Payment\s+of|Transaction\s+of|Final\s+Amount|Total\s+Fare|Total\s+Due|Amount\s+Due|Invoice\s+Total)\b.{0,150}?([\d,]+(?:\.\d{1,2})?)',
    ]
    for pattern in labelled_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            val = match.group(1).replace(',', '')
            try:
                if float(val) > 0:
                    return f"\u20b9{match.group(1)}"
            except ValueError:
                pass

    if structured_total:
        return structured_total

    # Priority 2: Docling Markdown table cells with currency
    pipe_match = re.search(
        r'\|\s*(?:[\u20b9]|INR|Rs\.?)\s*([\d,]+(?:\.\d{1,2})?)\s*\|',
        text, re.IGNORECASE
    )
    if pipe_match:
        return f"\u20b9{pipe_match.group(1)}"

    # Priority 3: Largest currency amount in text (avoids picking up tax sub-totals)
    all_amounts = re.findall(
        r'(?:[\u20b9]|INR|Rs\.?)\s*([\d,]+(?:\.\d{1,2})?)',
        text, re.IGNORECASE
    )
    if all_amounts:
        try:
            largest = max(all_amounts, key=lambda x: float(x.replace(',', '')))
            if float(largest.replace(',', '')) > 0:
                return f"\u20b9{largest}"
        except ValueError:
            pass

    return None



def extract_dates(text: str) -> list[str]:
    """Extract dates from text."""
    patterns = [
        r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{4}-\d{2}-\d{2}',
    ]
    dates = []
    for p in patterns:
        dates.extend(re.findall(p, text, re.IGNORECASE))
    return list(dict.fromkeys(dates))[:5]


def extract_order_id(text: str) -> Optional[str]:
    """Extract order/transaction ID."""
    patterns = [
        r'Order\s*(?:ID|#|No\.?)\s*:?\s*([A-Z0-9\-]{6,20})',
        r'Transaction\s*(?:ID|#|No\.?)\s*:?\s*([A-Z0-9\-]{6,20})',
        r'Booking\s*(?:ID|#|No\.?)\s*:?\s*([A-Z0-9\-]{6,20})',
        r'(?:UPI|Ref)\s*(?:Ref|ID|No\.?)\s*:?\s*([A-Z0-9]{8,20})',
        r'#([A-Z0-9\-]{8,20})',
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def parse_ride_details(text: str) -> dict:
    """Extract ride-specific details."""
    details = {}
    km = re.search(r'([\d.]+)\s*(?:km|kilometres?|kms)', text, re.IGNORECASE)
    if km:
        details["distance"] = f"{km.group(1)} km"
    mins = re.search(r'(\d+)\s*(?:minutes?|mins?)', text, re.IGNORECASE)
    if mins:
        details["duration"] = f"{mins.group(1)} min"
    plate = re.search(r'\b([A-Z]{2}\d{2}[A-Z]{1,2}\d{4})\b', text)
    if plate:
        details["license_plate"] = plate.group(1)
    # pickup/drop from common patterns
    from_match = re.search(r'(?:From|Pickup|Start)\s*:?\s*(.{10,60}?)(?:\n|To:|Drop)', text, re.IGNORECASE)
    to_match = re.search(r'(?:To|Drop|End|Destination)\s*:?\s*(.{10,60}?)(?:\n|$)', text, re.IGNORECASE)
    if from_match:
        details["from"] = from_match.group(1).strip()[:60]
    if to_match:
        details["to"] = to_match.group(1).strip()[:60]
    return details


def parse_food_details(text: str) -> dict:
    """Extract food order details."""
    details = {}
    restaurant = re.search(r'(?:from|restaurant|ordered from)\s*:?\s*([A-Za-z\s&\']{3,40})', text, re.IGNORECASE)
    if restaurant:
        details["restaurant"] = restaurant.group(1).strip()
    items = re.findall(r'^\s*[\d]+\s*[xX×]\s*(.{5,50}?)(?:\s+₹|\s+Rs)', text, re.MULTILINE)
    if items:
        details["items"] = items[:5]
    delivery = re.search(r'Delivery\s*(?:fee|charge)\s*:?\s*(?:₹|Rs\.?)?\s*([\d,]+(?:\.\d{2})?)', text, re.IGNORECASE)
    if delivery:
        details["delivery_fee"] = f"₹{delivery.group(1)}"
    return details


def parse_payment_details(text: str) -> dict:
    """Extract payment/UPI details."""
    details = {}
    to_match = re.search(r'(?:To|Paid to|Sent to|Recipient)\s*:?\s*(.{3,50}?)(?:\n|$)', text, re.IGNORECASE)
    from_match = re.search(r'(?:From|Paid from|Account)\s*:?\s*(.{3,50}?)(?:\n|$)', text, re.IGNORECASE)
    if to_match:
        details["to"] = to_match.group(1).strip()
    if from_match:
        details["from"] = from_match.group(1).strip()
    upi = re.search(r'UPI\s*(?:ID|Ref)\s*:?\s*([\w@.]+)', text, re.IGNORECASE)
    if upi:
        details["upi_ref"] = upi.group(1)
    status = re.search(r'\b(Success(?:ful)?|Failed|Pending|Declined|Completed)\b', text, re.IGNORECASE)
    if status:
        details["status"] = status.group(1)
    return details


def parse_shopping_details(text: str) -> dict:
    """Extract shopping order details."""
    details = {}
    items = re.findall(r'^\s*[-•]\s*(.{5,80}?)(?:\s+₹|\s+Rs|\n)', text, re.MULTILINE)
    if not items:
        items = re.findall(r'Item\s*:?\s*(.{5,60}?)(?:\n|$)', text, re.IGNORECASE)
    if items:
        details["items"] = [i.strip() for i in items[:5]]
    delivery = re.search(r'(?:Expected|Delivery|Estimated)\s*(?:by|date|on)?\s*:?\s*(.{5,40}?)(?:\n|$)', text, re.IGNORECASE)
    if delivery:
        details["delivery_date"] = delivery.group(1).strip()
    return details


def parse_travel_details(text: str) -> dict:
    """Extract travel (flight/hotel/train) details."""
    details = {}
    
    # Flight details
    flight_pnr = re.search(r'(?:PNR|Booking Ref|Booking Reference)\s*:?\s*([A-Z0-9]{5,8})', text, re.IGNORECASE)
    if flight_pnr:
        details["pnr"] = flight_pnr.group(1)
        
    flight_no = re.search(r'(?:Flight|Flight No)\s*:?\s*([A-Z0-9]{2,6}\s*\d{1,4})', text, re.IGNORECASE)
    if flight_no:
        details["flight_no"] = flight_no.group(1).strip()
        
    # Hotel details
    check_in = re.search(r'(?:Check-in|Check in|Arrival)\s*:?\s*([\w\s,-]{5,20})(?:\n|$)', text, re.IGNORECASE)
    if check_in:
        details["check_in"] = check_in.group(1).strip()
        
    check_out = re.search(r'(?:Check-out|Check out|Departure)\s*:?\s*([\w\s,-]{5,20})(?:\n|$)', text, re.IGNORECASE)
    if check_out:
        details["check_out"] = check_out.group(1).strip()
        
    # Generic Travel
    passenger = re.search(r'(?:Passenger|Guest|Name)\s*:?\s*([A-Za-z\s]{3,40})(?:\n|$)', text, re.IGNORECASE)
    if passenger:
        details["passenger"] = passenger.group(1).strip()
        
    return details


def parse_transaction(email: dict) -> dict:
    """
    Main entry point. Parse a Gmail email dict into a structured transaction.

    Input: email dict from gmail_tool.search_gmail_with_attachments()
    Output: enriched transaction dict for the dashboard renderer.
    """
    body = email.get("body_text", "") or email.get("snippet", "")
    
    # Check Structured Tool Outputs FIRST to avoid hallucinations
    structured_total = None
    for analysis in email.get("image_analyses", []):
        if isinstance(analysis, dict) and analysis.get("totals", {}).get("total"):
            structured_total = analysis["totals"]["total"]
            break
            
    if not structured_total:
        for doc in email.get("doc_analyses", []):
            if isinstance(doc, dict):
                amounts = doc.get("entities", {}).get("amounts", [])
                if amounts: # Docling identifies the largest amounts as totals
                    structured_total = amounts[0]
                    break

    # Append text from image attachments
    for analysis in email.get("image_analyses", []):
        if isinstance(analysis, dict):
            body += "\n" + analysis.get("extracted_text", "")
            body += "\n" + analysis.get("summary", "")

    # Append text from document attachments
    for doc in email.get("doc_analyses", []):
        if isinstance(doc, dict):
            body += "\n" + doc.get("extracted_text", "")
            # Doc analyses might have embedded image analyses too
            for img_analysis in doc.get("image_analyses", []):
                 if isinstance(img_analysis, dict):
                     body += "\n" + img_analysis.get("extracted_text", "")

    subject = email.get("subject", "")
    full_text = subject + " " + body

    # Detect category
    category, vendor = detect_category(full_text)

    # Extract universals
    total = extract_total(full_text, structured_total)
    amounts = extract_amounts(full_text)
    dates = extract_dates(full_text)
    order_id = extract_order_id(full_text)

    # Category-specific parsing
    details = {}
    if category == "ride":
        details = parse_ride_details(body)
    elif category == "food":
        details = parse_food_details(body)
    elif category == "payment":
        details = parse_payment_details(body)
    elif category == "shopping":
        details = parse_shopping_details(body)
    elif category == "travel":
        details = parse_travel_details(body)

    # Build transaction object
    transaction = {
        "id": email.get("id", ""),
        "category": category,
        "vendor": vendor,
        "category_config": CATEGORY_CONFIG.get(category, CATEGORY_CONFIG["unknown"]),
        "subject": subject,
        "from_email": email.get("from", ""),
        "date": email.get("date", ""),
        "total": total,
        "amounts": amounts,
        "dates": dates,
        "order_id": order_id,
        "details": details,
        "snippet": email.get("snippet", "")[:200],
        "body_preview": body[:800],
        "image_analyses": email.get("image_analyses", []),
        "doc_analyses": email.get("doc_analyses", []),
        "attachment_count": email.get("attachment_count", 0),
        "reimbursable": _is_reimbursable(body, category),
        "payment_method": _detect_payment_method(full_text),
    }

    return transaction


def _is_reimbursable(text: str, category: str) -> bool:
    """Check if receipt is reimbursable."""
    if category in ("ride", "food", "travel", "shopping"):
        keywords = ["reimburs", "official", "business", "expense", "tax invoice", "gst"]
        return any(k in text.lower() for k in keywords)
    return False


def _detect_payment_method(text: str) -> str:
    """Detect payment method used."""
    text_lower = text.lower()
    if "cash" in text_lower:
        return "Cash"
    if "upi" in text_lower:
        return "UPI"
    if "credit card" in text_lower or "credit" in text_lower:
        return "Credit Card"
    if "debit card" in text_lower or "debit" in text_lower:
        return "Debit Card"
    if "wallet" in text_lower:
        return "Wallet"
    if "net banking" in text_lower or "netbanking" in text_lower:
        return "Net Banking"
    return "Unknown"


def parse_transactions_bulk(emails: list[dict]) -> dict:
    """
    Parse a list of emails into transactions and compute summary stats.

    Returns:
    {
        "transactions": [...],
        "summary": {
            "total_spend": float,
            "count": int,
            "by_category": {...},
            "by_vendor": {...},
        }
    }
    """
    # ── Step 1: Parse and Deduplicate ───────────────────────────────────────
    merged = {}
    for item in emails:
        # Check if this is already a parsed transaction (e.g. from Google Sheets)
        if item.get("_source") == "sheets":
            t = item
        else:
            t = parse_transaction(item)
            
        tid = t.get("id")
        if not tid:
            if t.get("_source") == "sheets":
                import uuid
                tid = f"sheets_legacy_{uuid.uuid4().hex[:8]}"
                t["id"] = tid
            else:
                continue
            
        # If we already have this ID, MERGE the data
        if tid in merged:
            existing = merged[tid]
            
            # Identify which version is which
            new_is_sheets = t.get("_source") == "sheets"
            existing_is_sheets = existing.get("_source") == "sheets"
            
            # If we have one Gmail and one Sheets version, merge them
            if (new_is_sheets and not existing_is_sheets) or (not new_is_sheets and existing_is_sheets):
                gmail_v = existing if not existing_is_sheets else t
                sheets_v = existing if existing_is_sheets else t
                
                # Use Sheets version for editable fields (may contain user overrides)
                merged_t = gmail_v.copy()
                for field in ["category", "vendor", "total", "payment_method", "notes"]:
                    if sheets_v.get(field) and str(sheets_v[field]).lower() not in ("", "unknown"):
                        merged_t[field] = sheets_v[field]
                
                # Special handling for notes: combine Sheets notes with Gmail details
                sheet_notes = sheets_v.get("notes", "")
                if sheet_notes:
                    if "details" not in merged_t:
                        merged_t["details"] = {}
                    merged_t["details"]["sheet_notes"] = sheet_notes
                
                merged_t["_source"] = "merged"
                merged[tid] = merged_t
            else:
                # Both are same source, use previous heuristic (keep one with more info)
                existing_has_total = bool(existing.get("total") and re.sub(r'[^\d.]', '', str(existing["total"])))
                new_has_total = bool(t.get("total") and re.sub(r'[^\d.]', '', str(t["total"])))
                
                if new_has_total and not existing_has_total:
                    merged[tid] = t
                elif new_has_total and existing_has_total:
                    if len(t.get("details", {})) > len(existing.get("details", {})):
                        merged[tid] = t
        else:
            merged[tid] = t
            
    transactions = list(merged.values())

    # ── Step 2: Compute summary stats ──────────────────────────────────────
    total_spend = 0.0
    by_category = {}
    by_vendor = {}

    for t in transactions:
        if t["total"]:
            amount_str = re.sub(r'[^\d.]', '', t["total"])
            try:
                amount = float(amount_str)
                total_spend += amount
                cat = t["category"]
                by_category[cat] = by_category.get(cat, 0) + amount
                vendor = t["vendor"]
                by_vendor[vendor] = by_vendor.get(vendor, 0) + amount
            except ValueError:
                pass

    return {
        "transactions": transactions,
        "summary": {
            "total_spend": round(total_spend, 2),
            "count": len(transactions),
            "by_category": {k: round(v, 2) for k, v in sorted(by_category.items(), key=lambda x: -x[1])},
            "by_vendor": {k: round(v, 2) for k, v in sorted(by_vendor.items(), key=lambda x: -x[1])},
        }
    }
