"""
environments/trace_env/core/world_model.py

SemanticWorldModel — the hidden state of the federated digital-life simulation.

This is the "true world" that the agent cannot see directly.
It stores ground-truth data (simulated emails, documents, events)
and exposes a snapshot that is always partially observable —
the agent must plan and retrieve to fill in the gaps.

Key design decisions:
  - Data never leaves its origin (federated).
  - The world model tracks what has been retrieved vs. what is still hidden.
  - Schema drift is modelled (APIs change format over time).

Extended with:
  - "image" source: stores VLM analysis results (LLaMA 4 Scout).
  - inject_image_analysis(): convenience wrapper for image results.
"""

from __future__ import annotations
import random
from datetime import datetime, timedelta
from typing import Any


class SemanticWorldModel:
    """
    Simulates a user's fragmented digital life across multiple sources.
    Partial observability: agent sees only what it retrieves.
    """

    def __init__(self, config: dict):
        self.config = config
        self._sources: dict[str, list[dict]] = {}
        self._retrieved_ids: set[str] = set()
        self._ground_truth: dict[str, Any] = {}
        self._schema_version: int = 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, task: dict):
        """
        Populate the simulated world for a new episode.
        Difficulty controls data volume and schema drift.
        """
        difficulty = task.get("difficulty", "easy")
        sources = task.get("available_sources", ["gmail"])
        self._ground_truth = task.get("ground_truth", {})
        self._retrieved_ids.clear()
        self._sources.clear()

        n_items = {"easy": 10, "medium": 50, "hard": 200}[difficulty]

        for source in sources:
            self._sources[source] = self._generate_source_data(
                source, n_items, difficulty
            )

        # Schema drift: hard tasks occasionally change field names mid-episode
        self._schema_version = 1 if difficulty != "hard" else random.choice([1, 2])

    def update(self, retrieved_data: list):
    
     for item in retrieved_data:
        if isinstance(item, dict) and "id" in item:
            self._retrieved_ids.add(item["id"])
        elif isinstance(item, list):
            for subitem in item:
                if isinstance(subitem, dict) and "id" in subitem:
                    self._retrieved_ids.add(subitem["id"])
    def inject_real_data(self, source: str, items: list[dict]):
   
     if source not in self._sources:
        self._sources[source] = []

     for item in items:
        # Ensure every item has an id
        if "id" not in item:
            item["id"] = f"{source}_real_{len(self._sources[source])}"
        
        # Add to sources if not already there
        existing_ids = {i["id"] for i in self._sources[source]}
        if item["id"] not in existing_ids:
            self._sources[source].append(item)
        
        # Mark as retrieved (visible)
        self._retrieved_ids.add(item["id"])
    def inject_image_analysis(self, analysis: dict):
        """
        Convenience method: store a VLM image analysis result in the world model
        as an 'image' source item, and mark it as retrieved (visible).

        Args:
            analysis: dict returned by image_tool.analyse_image() or
                      image_tool.analyse_image_from_bytes().
        """
        self.inject_real_data("image", [analysis])

    def inject_doc_analysis(self, analysis: dict):
        """
        Convenience method: store a document extraction result in the world model
        as a 'document' source item, and mark it as retrieved (visible).

        Args:
            analysis: dict returned by doc_tool.extract_document().
        """
        self.inject_real_data("document", [analysis])

    def snapshot(self) -> dict[str, Any]:
     visible = []
     hidden_counts = {}
  
     for source, items in self._sources.items():
        hidden = 0
        for item in items:
            item_id = item.get("id", "")
            if item_id in self._retrieved_ids:
                visible.append(item)
            else:
                hidden += 1
        hidden_counts[source] = hidden

     # Build concise image analysis previews for the agent observation
     image_analyses_preview = [
        {
            "source_ref":     item.get("source_ref", ""),
            "summary":        item.get("summary", "")[:200],
            "extracted_text": item.get("extracted_text", "")[:300],
            "entities":       item.get("entities", {}),
        }
        for item in self._sources.get("image", [])
        if item.get("id", "") in self._retrieved_ids
     ]

     # Build concise document extraction previews
     doc_analyses_preview = [
        {
            "filename":       item.get("filename", ""),
            "parser_used":    item.get("parser_used", ""),
            "summary":        item.get("summary", "")[:200],
            "extracted_text": item.get("extracted_text", "")[:500],
            "page_count":     item.get("page_count"),
            "images_found":   item.get("images_found", 0),
        }
        for item in self._sources.get("document", [])
        if item.get("id", "") in self._retrieved_ids
     ]

     return {
        "visible_items": len(visible),
        "hidden_counts": hidden_counts,
        "schema_version": self._schema_version,
        "retrieved_preview": visible[:3],
        "total_sources": list(self._sources.keys()),
        "total_items": sum(len(v) for v in self._sources.values()),
        "image_analyses": image_analyses_preview,
        "doc_analyses": doc_analyses_preview,
     }

    def query(self, source: str, query: str, limit: int = 10) -> list[dict]:
     if source not in self._sources:
        return []

     results = []
     query_words = query.lower().split()

     for item in self._sources[source]:
        text = " ".join(str(v) for v in item.values()).lower()
        # Match if ANY query word appears in the item
        if any(word in text for word in query_words):
            results.append(self._apply_schema(item))
        if len(results) >= limit:
            break

    # If still no results, return first `limit` items
     if not results:
        results = [self._apply_schema(i) for i in self._sources[source][:limit]]

     return results
    def ground_truth_answer(self) -> Any:
        return self._ground_truth.get("answer")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_source_data(
        self, source: str, n: int, difficulty: str
    ) -> list[dict]:
        """Generate synthetic data for a given source with adversarial examples."""
        items = []
        base_date = datetime(2024, 1, 1)
        
        # Determine number of adversarial cases based on difficulty
        adversarial_prob = 0.0 if difficulty == "easy" else (0.2 if difficulty == "medium" else 0.4)

        for i in range(n):
            date = base_date + timedelta(days=random.randint(0, 900))
            item_id = f"{source}_{i:04d}"
            is_adversarial = random.random() < adversarial_prob

            if source == "gmail":
                if is_adversarial:
                    # Adversarial cases for Gmail
                    adv_type = random.choice(["noise", "refund", "conflict", "multi_item"])
                    if adv_type == "noise":
                        item = {
                            "id": item_id, "date": date.isoformat(),
                            "from": "marketing@swiggy.in",
                            "subject": "Weekend Offers! Get 50% off",
                            "body_preview": "Order now and get massive discounts. No receipt here.",
                        }
                    elif adv_type == "refund":
                        item = {
                            "id": item_id, "date": date.isoformat(),
                            "from": "receipts@uber.com",
                            "subject": "Trip Cancelled - Refund Processed",
                            "amount": 0.0,
                            "body_preview": "Your trip was cancelled. ₹150 has been refunded to your source.",
                        }
                    elif adv_type == "conflict":
                        item = {
                            "id": item_id, "date": date.isoformat(),
                            "from": "receipts@uber.com",
                            "subject": "Uber Eats Order Receipt",
                            "amount": 320.0,
                            "body_preview": "Your food order from Uber Eats. Total: ₹320.00.",
                        }
                    else: # multi_item
                        item = {
                            "id": item_id, "date": date.isoformat(),
                            "from": "orders@amazon.in",
                            "subject": "Amazon Invoice - Multiple Items",
                            "amount": 1500.0,
                            "body_preview": "Item 1: ₹500, Item 2: ₹800, Shipping: ₹100, Tax: ₹100. Grand Total: ₹1500.",
                        }
                else:
                    # Standard Gmail cases
                    vendor = random.choice(["Uber", "MakeMyTrip", "Rapido", "Airtel", "Netflix"])
                    item = {
                        "id": item_id,
                        "date": date.isoformat(),
                        "from": f"receipts@{vendor.lower()}.com",
                        "subject": f"{vendor} Receipt / Invoice",
                        "amount": round(random.uniform(50, 5000), 2),
                        "body_preview": f"Your receipt from {vendor} is attached. Total amount paid.",
                    }

            elif source == "sheets":
                # Generate Google Sheets ledger rows
                amount = round(random.uniform(50, 5000), 2)
                vendor = random.choice(["Uber", "MakeMyTrip", "Amazon", "Swiggy"])
                
                if is_adversarial:
                    adv_type = random.choice(["refund", "schema_drift"])
                    if adv_type == "refund":
                        item = {
                            "id": item_id, "Date": date.strftime("%d/%m/%Y"),
                            "Vendor": vendor, "Amount": -amount,
                            "Category": "Refund", "Status": "Reverted"
                        }
                    else: # schema_drift
                        item = {
                            "id": item_id, "Transaction Date": date.strftime("%Y-%m-%d"),
                            "Merchant": vendor, "Cost": amount,
                            "Type": "Debit", "Notes": "Messy headers"
                        }
                else:
                    item = {
                        "id": item_id, 
                        "Date": date.strftime("%Y-%m-%d"),
                        "Vendor": vendor, 
                        "Category": "Expense",
                        "Amount": amount,
                        "Method": random.choice(["UPI", "Card", "Cash"])
                    }

            elif source == "image":
                vendor = random.choice(["Amazon", "Uber", "Starbucks"])
                amount = round(random.uniform(5, 250), 2)
                item = {
                    "id": item_id, "source_type": "simulated",
                    "source_ref": f"receipt_{i}.jpg",
                    "extracted_text": f"{vendor}\nOrder #{i}\nTotal: ${amount}",
                    "summary": f"Receipt from {vendor} for ${amount}",
                    "entities": {"amounts": [f"${amount}"], "vendors": [vendor]},
                }
            else:
                item = {"id": item_id, "date": date.isoformat(), "content": f"Item {i}"}

            items.append(item)

        return items

    def _apply_schema(self, item: dict) -> dict:
        """Apply schema drift for hard difficulty (simulates API changes)."""
        if self._schema_version == 1:
            return item
        # Schema v2: rename some fields (Patronus AI sub-theme)
        renamed = {}
        renames = {"date": "timestamp", "from": "sender", "created": "creation_date"}
        for k, v in item.items():
            renamed[renames.get(k, k)] = v
        return renamed
