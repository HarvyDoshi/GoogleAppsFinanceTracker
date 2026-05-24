"""
environments/trace_env/tools/image_tool.py

Hybrid Image Analysis Pipeline for Trace — Local-First, RAM-Aware.

Architecture (3-stage multi-agent router):
─────────────────────────────────────────────────────────────────
  Stage 1 │ ROUTER     │ moondream (~2GB RAM)
           │            │ Classifies: financial | tech_doc | scene
           ▼            ▼
  Stage 2A │ FINANCIAL  │ qwen2-vl (~6GB RAM)
           │            │ High-precision OCR → Markdown table → JSON
           │            │ Temperature: 0.0 (no hallucination)
           ▼            ▼
  Stage 2B │ GENERAL    │ llama3.2-vision (~6GB RAM)
           │            │ Contextual summary + keyword extraction
           │            │ Temperature: 0.7 (creative reasoning)
           ▼            ▼
  Stage 3  │ UNIFIED    │ Python logic gate
           │ OUTPUT     │ Standard JSON → WorldModel

Memory strategy:
  - Ollama serves one model at a time — never loads all three simultaneously.
  - Router runs first (lightest), then swapped out before the specialist loads.
  - Peak RAM: ~6-7GB (well within 16GB).

Requires Ollama running locally: https://ollama.com
  ollama pull moondream
  ollama pull qwen2-vl
  ollama pull llama3.2-vision
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# ── Default model config (overridden via env_config.yaml) ────────────────────

_ROUTER_MODEL    = "moondream"           # ~1.7GB — classification only
_FINANCIAL_MODEL = "qwen2.5vl:3b"        # ~3.2GB — receipt/invoice OCR
_GENERAL_MODEL   = "llama3.2-vision"     # ~7.8GB — general scene / tech docs
_OLLAMA_HOST     = "http://localhost:11434"
_FINANCIAL_TEMP  = 0.0                  # strict: exact numbers
_GENERAL_TEMP    = 0.7                  # creative: rich descriptions


# ── Image-type categories (router output → specialist branch) ─────────────────

_FINANCIAL_KEYWORDS = {
    "financial", "receipt", "invoice", "statement", "bill",
    "payment", "transaction", "bank", "tax", "expense"
}
_TECH_KEYWORDS = {
    "technical", "screenshot", "tech_doc", "code", "diagram",
    "terminal", "ui", "interface", "chart", "graph"
}


# ── Public configuration ──────────────────────────────────────────────────────

def configure(config: dict):
    """
    Called once at startup with the image_analysis section of env_config.yaml.

    config = {
        "router_model":    "moondream",
        "financial_model": "qwen2-vl",
        "general_model":   "llama3.2-vision",
        "ollama_host":     "http://localhost:11434",
        "financial_temp":  0.0,
        "general_temp":    0.7,
    }
    """
    global _ROUTER_MODEL, _FINANCIAL_MODEL, _GENERAL_MODEL
    global _OLLAMA_HOST, _FINANCIAL_TEMP, _GENERAL_TEMP

    _ROUTER_MODEL    = config.get("router_model",    _ROUTER_MODEL)
    _FINANCIAL_MODEL = config.get("financial_model", _FINANCIAL_MODEL)
    _GENERAL_MODEL   = config.get("general_model",   _GENERAL_MODEL)
    _OLLAMA_HOST     = config.get("ollama_host",     _OLLAMA_HOST)
    _FINANCIAL_TEMP  = config.get("financial_temp",  _FINANCIAL_TEMP)
    _GENERAL_TEMP    = config.get("general_temp",    _GENERAL_TEMP)

    logger.info(
        f"[IMAGE_TOOL] Hybrid pipeline configured — "
        f"router={_ROUTER_MODEL}, financial={_FINANCIAL_MODEL}, general={_GENERAL_MODEL}"
    )


# ── Public API (same schema as before — rest of codebase unchanged) ───────────

def analyse_image(
    source: str,
    question: Optional[str] = None,
    source_type: Optional[str] = None,
) -> dict:
    """
    Analyse an image through the 3-stage hybrid pipeline.

    Args:
        source:      Image URL, local file path, or base64-encoded string.
        question:    Optional specific question. If given, skips routing and
                     asks the general model directly.
        source_type: "url" | "path" | "base64" — auto-detected if None.

    Returns:
        Unified structured dict (see module docstring for full schema).
    """
    result = _empty_result(source)
    stype = source_type or _detect_source_type(source)
    result["source_type"] = stype

    try:
        # ── Prepare image bytes for Ollama ─────────────────────────────
        image_bytes = _load_image_bytes(source, stype)

        if question:
            # Direct Q&A — skip routing, use general model
            logger.info(f"[IMAGE_TOOL] Direct Q&A mode: {question[:60]}")
            raw = _call_ollama(
                model=_GENERAL_MODEL,
                image_bytes=image_bytes,
                prompt=_qa_prompt(question),
                temperature=_GENERAL_TEMP,
            )
            parsed = _parse_json_response(raw)
            result["extracted_text"] = parsed.get("extracted_text", "")
            result["summary"] = f"Q: {question} → A: {parsed.get('answer', raw)}"
            result["entities"]["other"].append(
                f"Q: {question} → A: {parsed.get('answer', '')}"
            )
            result["raw_model_output"] = raw
            result["pipeline_branch"] = "direct_qa"
            return result

        # ── Stage 1: Route ─────────────────────────────────────────────
        logger.info("[IMAGE_TOOL] Stage 1 — Routing with moondream...")
        doc_type = _route(image_bytes)
        result["doc_type"] = doc_type
        logger.info(f"[IMAGE_TOOL] Router classified as: '{doc_type}'")

        # ── Stage 2: Specialized processing ───────────────────────────
        if doc_type == "financial":
            result = _financial_branch(image_bytes, result)
        else:
            result = _general_branch(image_bytes, result, doc_type)

    except OllamaNotRunningError as e:
        result["error"] = str(e)
        logger.error(f"[IMAGE_TOOL] {e}")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        result["error"] = err
        logger.error(f"[IMAGE_TOOL] Pipeline failed: {err}")

    return result


def analyse_image_from_bytes(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    filename: str = "attachment",
    question: Optional[str] = None,
) -> dict:
    """
    Analyse raw image bytes (e.g. from a Gmail attachment).

    Args:
        image_bytes: Raw image bytes.
        mime_type:   MIME type (e.g. "image/jpeg").
        filename:    Original filename for reference in the result.
        question:    Optional specific question about the image.

    Returns:
        Unified analysis dict.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    result = analyse_image(b64, question=question, source_type="base64")
    result["source_ref"] = filename
    return result


# ── Stage 1: Router ───────────────────────────────────────────────────────────

_ROUTER_PROMPT = """Classify this image into ONLY one category:
- financial: Any receipt, invoice, bill, or bank statement (even if forwarded).
- tech_doc: Screenshots of code, terminals, or system UI.
- scene: General photos or people.

CRITICAL: If it contains currency symbols (₹, $) or the word 'Total', it is 'financial'.
Reply with ONLY the word."""

# ── Stage 2A: Financial specialist - Strict temperature for accuracy 
_FINANCIAL_TEMP = 0.0  # Zero temperature for deterministic extraction

# ── Stage 2B: General specialist - Lowered slightly for better OCR reliability
_GENERAL_TEMP = 0.4 


def _route(image_bytes: bytes) -> str:
    """
    Use moondream to classify the image type.
    Returns: 'financial' | 'tech_doc' | 'scene'
    """
    try:
        raw = _call_ollama(
            model=_ROUTER_MODEL,
            image_bytes=image_bytes,
            prompt=_ROUTER_PROMPT,
            temperature=0.0,  # always deterministic for routing
        )
        classification = raw.strip().lower().split()[0] if raw.strip() else "scene"

        # Normalise to our three categories
        if any(kw in classification for kw in _FINANCIAL_KEYWORDS):
            return "financial"
        if any(kw in classification for kw in _TECH_KEYWORDS):
            return "tech_doc"
        return "scene"

    except Exception as e:
        logger.warning(f"[IMAGE_TOOL] Router failed ({e}), defaulting to 'scene'")
        return "scene"


# ── Stage 2A: Financial Branch (qwen2-vl, temp=0.0) ──────────────────────────

_FINANCIAL_PROMPT = """You are a precise financial document extractor. Analyse this receipt/invoice/statement and extract ALL data.

Respond ONLY with this exact JSON schema (no markdown, no extra text):
{
  "extracted_text": "<all text visible in the document, verbatim and complete>",
  "summary": "<one sentence: vendor, date, total amount>",
  "entities": {
    "amounts": ["<ALL monetary values found, e.g. '$42.99', 'USD 150.00', 'Tax: $3.50'>"],
    "dates": ["<all dates found>"],
    "vendors": ["<company/store name>"],
    "items": ["<each line item with price, e.g. 'Coffee x2: $8.00'>"],
    "other": ["<order numbers, addresses, payment method, any other key fields>"]
  },
  "totals": {
    "subtotal": "<subtotal amount or empty string>",
    "tax": "<tax amount or empty string>",
    "total": "<final total or empty string>",
    "currency": "<currency code or symbol>"
  }
}"""


def _financial_branch(image_bytes: bytes, result: dict) -> dict:
    """
    High-precision financial extraction using qwen2-vl at temperature 0.0.
    Extracts amounts, line items, tax, totals with maximum accuracy.
    """
    logger.info(f"[IMAGE_TOOL] Stage 2A — Financial extraction with {_FINANCIAL_MODEL}...")

    raw = _call_ollama(
        model=_FINANCIAL_MODEL,
        image_bytes=image_bytes,
        prompt=_FINANCIAL_PROMPT,
        temperature=_FINANCIAL_TEMP,  # 0.0 — no hallucination on numbers
    )
    result["raw_model_output"] = raw
    result["pipeline_branch"] = "financial"

    parsed = _parse_json_response(raw)

    result["extracted_text"] = parsed.get("extracted_text", "")
    result["summary"]        = parsed.get("summary", "")

    ents = parsed.get("entities", {})
    result["entities"]["amounts"] = ents.get("amounts", [])
    result["entities"]["dates"]   = ents.get("dates", [])
    result["entities"]["vendors"] = ents.get("vendors", [])
    result["entities"]["items"]   = ents.get("items", [])
    result["entities"]["other"]   = ents.get("other", [])

    # Financial-specific: attach totals breakdown
    result["totals"] = parsed.get("totals", {
        "subtotal": "", "tax": "", "total": "", "currency": ""
    })

    logger.info(
        f"[IMAGE_TOOL] Financial extraction complete — "
        f"total={result['totals'].get('total', 'n/a')}, "
        f"vendor={result['entities']['vendors']}"
    )
    return result


# ── Stage 2B: General/Scene Branch (llama3.2-vision, temp=0.7) ───────────────

_GENERAL_PROMPT = """Analyse this image carefully and provide a rich contextual description.

Respond ONLY with this exact JSON (no markdown, no extra text):
{
  "extracted_text": "<all visible text in the image, verbatim>",
  "summary": "<2-3 sentence description: what is shown, key context, important details>",
  "entities": {
    "amounts": ["<any prices or numbers found>"],
    "dates": ["<any dates found>"],
    "vendors": ["<any brand names, company names, locations>"],
    "items": ["<main subjects, objects, or topics visible>"],
    "other": ["<technical keywords, URLs, usernames, or any other significant entities>"]
  },
  "scene_tags": ["<5-10 descriptive tags for this image, e.g. 'outdoor', 'code_screenshot', 'handwritten'>"]
}"""

_TECH_PROMPT = """Analyse this technical screenshot or diagram and extract all information.

Respond ONLY with this exact JSON (no markdown, no extra text):
{
  "extracted_text": "<all visible text, code, labels, values>",
  "summary": "<what this screenshot shows: application, purpose, key data>",
  "entities": {
    "amounts": ["<any numeric values, percentages, metrics>"],
    "dates": ["<any timestamps or dates>"],
    "vendors": ["<application name, framework, service name>"],
    "items": ["<function names, error messages, key UI elements, config values>"],
    "other": ["<file paths, URLs, environment names, version numbers>"]
  },
  "scene_tags": ["<descriptive tags, e.g. 'error_log', 'api_response', 'dashboard'>"]
}"""


def _general_branch(image_bytes: bytes, result: dict, doc_type: str) -> dict:
    """
    Contextual reasoning using llama3.2-vision at temperature 0.7.
    Handles general scenes, photos, and technical screenshots.
    """
    prompt = _TECH_PROMPT if doc_type == "tech_doc" else _GENERAL_PROMPT
    logger.info(
        f"[IMAGE_TOOL] Stage 2B — {'Technical' if doc_type == 'tech_doc' else 'General'} "
        f"analysis with {_GENERAL_MODEL}..."
    )

    raw = _call_ollama(
        model=_GENERAL_MODEL,
        image_bytes=image_bytes,
        prompt=prompt,
        temperature=_GENERAL_TEMP,  # 0.7 — richer, more descriptive output
    )
    result["raw_model_output"] = raw
    result["pipeline_branch"] = doc_type  # "tech_doc" or "scene"

    parsed = _parse_json_response(raw)

    result["extracted_text"] = parsed.get("extracted_text", "")
    result["summary"]        = parsed.get("summary", "")

    ents = parsed.get("entities", {})
    result["entities"]["amounts"] = ents.get("amounts", [])
    result["entities"]["dates"]   = ents.get("dates", [])
    result["entities"]["vendors"] = ents.get("vendors", [])
    result["entities"]["items"]   = ents.get("items", [])
    result["entities"]["other"]   = ents.get("other", []) + parsed.get("scene_tags", [])

    result["scene_tags"] = parsed.get("scene_tags", [])

    logger.info(
        f"[IMAGE_TOOL] General analysis complete — "
        f"tags={result.get('scene_tags', [])[:3]}"
    )
    return result


# ── Ollama client ─────────────────────────────────────────────────────────────

class OllamaNotRunningError(RuntimeError):
    """Raised when Ollama is not running or not reachable."""
    pass


def _call_ollama(
    model: str,
    image_bytes: bytes,
    prompt: str,
    temperature: float = 0.0,
) -> str:
    """
    Call an Ollama vision model with an image and a text prompt.
    Ollama handles model loading/unloading — never loads two heavy models at once.
    """
    try:
        import ollama
    except ImportError:
        raise ImportError(
            "ollama Python package not installed. Run: pip install ollama"
        )

    # Configure Ollama host if non-default
    if _OLLAMA_HOST != "http://localhost:11434":
        os.environ["OLLAMA_HOST"] = _OLLAMA_HOST

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],   # Ollama accepts raw bytes directly
                }
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]

    except Exception as e:
        err_str = str(e).lower()
        if any(k in err_str for k in ["connection refused", "connect", "not running", "404"]):
            raise OllamaNotRunningError(
                f"Ollama is not running at {_OLLAMA_HOST}. "
                f"Start it with: ollama serve\n"
                f"Then pull models: ollama pull {model}"
            )
        # Model not pulled yet
        if "model" in err_str and ("not found" in err_str or "pull" in err_str):
            raise OllamaNotRunningError(
                f"Model '{model}' not found in Ollama. Pull it with: ollama pull {model}"
            )
        raise


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_image_bytes(source: str, stype: str) -> bytes:
    """Load image as raw bytes from URL, local path, or base64 string."""
    if stype == "path":
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {source}")
        return path.read_bytes()

    elif stype == "base64":
        # Handle both plain base64 and data:image/...;base64,... URIs
        if source.startswith("data:"):
            _, b64_part = source.split(",", 1)
            return base64.b64decode(b64_part)
        return base64.b64decode(source)

    elif stype == "url":
        # Download URL
        if source.startswith("data:"):
            _, b64_part = source.split(",", 1)
            return base64.b64decode(b64_part)
        import urllib.request
        with urllib.request.urlopen(source, timeout=15) as resp:
            return resp.read()

    raise ValueError(f"Unknown source type: {stype}")


def _detect_source_type(source: str) -> str:
    """Auto-detect whether source is a URL, file path, or base64 string."""
    if source.startswith(("http://", "https://", "data:")):
        return "url"
    if Path(source).exists():
        return "path"
    try:
        base64.b64decode(source, validate=True)
        return "base64"
    except Exception:
        pass
    return "path"


def _parse_json_response(text: str) -> dict:
    """
    Bulletproof JSON extraction. Digs through messy model output to find valid JSON.
    """
    if not text or not text.strip():
        return {}

    # 1. Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Try to find JSON inside markdown blocks or braces
    # Look for the last { and last } to catch the main object
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 3. Last resort: Clean common model errors (trailing commas, etc.)
            try:
                # Remove trailing commas before closing braces
                cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)
                # Ensure property names are quoted
                cleaned = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', cleaned)
                return json.loads(cleaned)
            except Exception:
                pass

    logger.warning(f"[IMAGE_TOOL] Failed to parse JSON from: {text[:100]}...")
    return {"extracted_text": text, "summary": text[:200], "entities": {}}


def _qa_prompt(question: str) -> str:
    return f"""Answer this question about the image: {question}

Respond ONLY with this JSON (no markdown):
{{
  "answer": "<direct answer to the question>",
  "extracted_text": "<all visible text in the image>",
  "confidence": "high|medium|low"
}}"""


def _empty_result(source: str) -> dict:
    """Return a blank result skeleton (same schema as before for compatibility)."""
    return {
        "id":               f"image_{uuid.uuid4().hex[:12]}",
        "source_type":      "url",
        "source_ref":       source,
        "doc_type":         "unknown",
        "pipeline_branch":  "unknown",
        "extracted_text":   "",
        "summary":          "",
        "entities": {
            "amounts":  [],
            "dates":    [],
            "vendors":  [],
            "items":    [],
            "other":    [],
        },
        "totals": {
            "subtotal": "",
            "tax":      "",
            "total":    "",
            "currency": "",
        },
        "scene_tags":       [],
        "raw_model_output": "",
        "error":            None,
    }
