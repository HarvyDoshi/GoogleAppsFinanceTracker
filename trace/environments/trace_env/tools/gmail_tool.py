"""
environments/trace_env/tools/gmail_tool.py

Gmail integration — searches the user's Gmail for messages.
Handles credential loading with proper path resolution and token refresh.

Extended with:
  - fetch_gmail_attachments(message_id): download image/PDF attachments
  - search_gmail_with_attachments(query): search + auto-analyse image attachments
"""

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import base64
import os
import pickle
import logging

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Resolve credential paths relative to the project root (trace/ directory)
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_CREDENTIALS_PATH = os.path.join(_PROJECT_ROOT, "credentials.json")
_TOKEN_PATH = os.path.join(_PROJECT_ROOT, "token_gmail.pkl")

# MIME types we treat as images for analysis
_IMAGE_MIME_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/gif",
    "image/webp", "image/bmp", "image/tiff",
}
# MIME types for documents that doc_tool can parse
_DOCUMENT_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/msword",  # .doc
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
}
# All MIME types we can analyse (images + documents)
_ANALYSABLE_MIME_TYPES = _IMAGE_MIME_TYPES | _DOCUMENT_MIME_TYPES


def get_gmail_service():
    import base64
    # Reconstruct files from Hugging Face Secrets if available
    if os.environ.get("GMAIL_TOKEN_B64") and not os.path.exists(_TOKEN_PATH):
        try:
            with open(_TOKEN_PATH, "wb") as f:
                f.write(base64.b64decode(os.environ["GMAIL_TOKEN_B64"]))
        except Exception as e:
            logger.error(f"Failed to decode GMAIL_TOKEN_B64: {e}")
            
    if os.environ.get("GCP_CREDENTIALS_B64") and not os.path.exists(_CREDENTIALS_PATH):
        try:
            with open(_CREDENTIALS_PATH, "wb") as f:
                f.write(base64.b64decode(os.environ["GCP_CREDENTIALS_B64"]))
        except Exception as e:
            logger.error(f"Failed to decode GCP_CREDENTIALS_B64: {e}")

    creds = None
    # Token saved after first login
    if os.path.exists(_TOKEN_PATH):
        with open(_TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)

    # Refresh expired credentials or run auth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                # Refresh failed — re-authenticate
                creds = None

        if not creds:
            if not os.path.exists(_CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Google credentials not found at {_CREDENTIALS_PATH}. "
                    f"Download from Google Cloud Console and place as credentials.json "
                    f"in the trace/ directory."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                _CREDENTIALS_PATH, SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(_TOKEN_PATH, "wb") as f:
            pickle.dump(creds, f)

    return build("gmail", "v1", credentials=creds)


def search_gmail(query: str, max_results: int = 10) -> list[dict]:
    """
    Search Gmail and return simplified message dicts.
    query examples: "receipt 2022", "invoice after:2022/1/1 before:2023/1/1"
    """
    try:
        service = get_gmail_service()
        results = service.users().messages().list(
            userId="me", q=query, maxResults=max_results
        ).execute()

        messages = results.get("messages", [])
        output = []

        for msg in messages:
            detail = service.users().messages().get(
                userId="me", id=msg["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"]
            ).execute()

            headers = {h["name"]: h["value"] for h in detail["payload"]["headers"]}
            output.append({
                "id": msg["id"],
                "date": headers.get("Date", ""),
                "from": headers.get("From", ""),
                "subject": headers.get("Subject", ""),
                "snippet": detail.get("snippet", ""),
            })

        return output
    except FileNotFoundError as e:
        print(f"[GMAIL_TOOL] {e}")
        return []
    except Exception as e:
        print(f"[GMAIL_TOOL] Error searching Gmail: {e}")
        return []


def fetch_gmail_attachments(message_id: str, analyse_images: bool = True) -> list[dict]:
    """
    Download attachments from a specific Gmail message.

    Args:
        message_id:     Gmail message ID (from search_gmail result).
        analyse_images: If True, automatically analyse image attachments using
                        the image_tool (LLaMA 4 Scout via HF Inference API).

    Returns:
        List of attachment dicts:
        {
            "attachment_id":  str,
            "filename":       str,
            "mime_type":      str,
            "size_bytes":     int,
            "data":           bytes,        # raw decoded bytes
            "image_analysis": dict | None,  # populated if analyse_images=True
        }
    """
    attachments = []
    body_text = ""
    try:
        service = get_gmail_service()

        # Get full message to inspect payload parts
        message = service.users().messages().get(
            userId="me", id=message_id, format="full"
        ).execute()

        parts = _flatten_parts(message.get("payload", {}))

        for part in parts:
            filename = part.get("filename", "")
            mime_type = part.get("mimeType", "")
            body = part.get("body", {})
            attachment_id = body.get("attachmentId")
            size_bytes = body.get("size", 0)

            # Extract email body text (plain or HTML)
            if mime_type in ("text/plain", "text/html") and body.get("data"):
                try:
                    text_bytes = base64.urlsafe_b64decode(body["data"] + "==")
                    text = text_bytes.decode("utf-8", errors="ignore")
                    if mime_type == "text/html":
                        import re
                        # Strip basic HTML tags to get plain text
                        text = re.sub(r'<[^>]+>', ' ', text)
                        # Unescape basic HTML entities
                        text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                        # Compress whitespace
                        text = re.sub(r'\s+', ' ', text).strip()
                    body_text += text + "\n"
                except Exception:
                    pass

            # ── Handle forwarded email containers (message/rfc822) ─────────────
            # The nested RFC822 email is stored as an attachment in Gmail.
            # We download it, parse it with Python's email module, and extract
            # any PDF/doc parts inside (e.g. Uber receipt, IndiGo invoice).
            if mime_type == "message/rfc822" and attachment_id:
                try:
                    att_data = service.users().messages().attachments().get(
                        userId="me", messageId=message_id, id=attachment_id
                    ).execute()
                    raw_b64 = att_data.get("data", "")
                    eml_bytes = base64.urlsafe_b64decode(raw_b64 + "==")

                    import email as _email
                    nested_msg = _email.message_from_bytes(eml_bytes)

                    # Walk the nested message to find PDF/doc parts
                    for nested_part in nested_msg.walk():
                        nested_ct = nested_part.get_content_type()
                        nested_fn = nested_part.get_filename() or ""
                        payload = nested_part.get_payload(decode=True)

                        if not payload:
                            continue

                        # Also accumulate plain text and html text from the nested email body
                        if nested_ct in ("text/plain", "text/html") and not nested_fn:
                            try:
                                text_bytes = nested_part.get_payload(decode=True)
                                text = text_bytes.decode("utf-8", errors="ignore")
                                if nested_ct == "text/html":
                                    import re
                                    text = re.sub(r'<[^>]+>', ' ', text)
                                    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                                    text = re.sub(r'\s+', ' ', text).strip()
                                body_text += text + "\n"
                            except Exception:
                                pass
                            continue

                        if nested_ct not in _DOCUMENT_MIME_TYPES and nested_ct not in _IMAGE_MIME_TYPES:
                            continue

                        # Give it a fallback filename based on content type
                        if not nested_fn:
                            ext_map = {"application/pdf": ".pdf", "image/jpeg": ".jpg", "image/png": ".png"}
                            nested_fn = f"nested_attachment{ext_map.get(nested_ct, '.bin')}"

                        nested_att = {
                            "attachment_id": f"nested_{nested_fn}",
                            "filename": nested_fn,
                            "mime_type": nested_ct,
                            "size_bytes": len(payload),
                            "data": payload,
                            "image_analysis": None,
                            "doc_analysis": None,
                        }

                        if nested_ct in _DOCUMENT_MIME_TYPES:
                            try:
                                from .doc_tool import extract_document
                                logger.info(f"[GMAIL_TOOL] Extracting nested doc: {nested_fn}")
                                doc_result = extract_document(
                                    file_bytes=payload,
                                    filename=nested_fn,
                                    mime_type=nested_ct,
                                    analyse_images=False,
                                )
                                nested_att["doc_analysis"] = doc_result
                                logger.info(
                                    f"[GMAIL_TOOL] Nested doc extracted: parser={doc_result.get('parser_used')}, "
                                    f"text_len={len(doc_result.get('extracted_text', ''))}, "
                                    f"amounts={doc_result.get('entities', {}).get('amounts', [])}"
                                )
                            except Exception as e:
                                logger.warning(f"[GMAIL_TOOL] Nested doc extraction failed: {e}")

                        attachments.append(nested_att)

                except Exception as e:
                    logger.warning(f"[GMAIL_TOOL] Failed to parse nested RFC822 message: {e}")
                continue  # Skip the normal attachment flow for rfc822 parts

            # Only process named standard attachments
            if not filename or not attachment_id:
                continue

            # Download attachment bytes
            att_data = service.users().messages().attachments().get(
                userId="me", messageId=message_id, id=attachment_id
            ).execute()

            raw_b64 = att_data.get("data", "")
            # Gmail uses URL-safe base64
            raw_bytes = base64.urlsafe_b64decode(raw_b64 + "==")

            attachment = {
                "attachment_id": attachment_id,
                "filename": filename,
                "mime_type": mime_type,
                "size_bytes": size_bytes,
                "data": raw_bytes,
                "image_analysis": None,
                "doc_analysis": None,
            }

            # Analyse image attachments with Ollama VLM pipeline
            # NOTE: Disabled by default — very slow on CPU (30-60s per image).
            # Amounts come from PDF text layer, not image logos.
            # Enable only if you specifically need VLM image OCR.
            if False and analyse_images and mime_type in _IMAGE_MIME_TYPES:
                try:
                    from .image_tool import analyse_image_from_bytes
                    logger.info(
                        f"[GMAIL_TOOL] Analysing image attachment: {filename} ({mime_type})"
                    )
                    analysis = analyse_image_from_bytes(
                        image_bytes=raw_bytes,
                        mime_type=mime_type,
                        filename=filename,
                    )
                    attachment["image_analysis"] = analysis
                    logger.info(
                        f"[GMAIL_TOOL] Image analysis complete for {filename}: "
                        f"extracted {len(analysis.get('extracted_text', ''))} chars"
                    )
                except Exception as e:
                    logger.warning(f"[GMAIL_TOOL] Image analysis failed for {filename}: {e}")

            # Extract text/images from document attachments (PDF, DOCX, PPTX)
            if mime_type in _DOCUMENT_MIME_TYPES:
                try:
                    from .doc_tool import extract_document
                    logger.info(
                        f"[GMAIL_TOOL] Extracting document: {filename} ({mime_type})"
                    )
                    doc_result = extract_document(
                        file_bytes=raw_bytes,
                        filename=filename,
                        mime_type=mime_type,
                        analyse_images=analyse_images,
                    )
                    attachment["doc_analysis"] = doc_result
                    logger.info(
                        f"[GMAIL_TOOL] Document extraction complete for {filename}: "
                        f"parser={doc_result.get('parser_used')}, "
                        f"text_len={len(doc_result.get('extracted_text', ''))}, "
                        f"amounts={doc_result.get('entities', {}).get('amounts', [])}"
                    )
                except Exception as e:
                    logger.warning(f"[GMAIL_TOOL] Document extraction failed for {filename}: {e}")

            attachments.append(attachment)
            logger.info(f"[GMAIL_TOOL] Downloaded attachment: {filename} ({size_bytes} bytes)")

    except FileNotFoundError as e:
        logger.error(f"[GMAIL_TOOL] {e}")
    except Exception as e:
        logger.error(f"[GMAIL_TOOL] Error fetching attachments for {message_id}: {e}")

    return {"attachments": attachments, "body_text": body_text.strip()}


def search_gmail_with_attachments(
    query: str,
    max_results: int = 10,
    analyse_images: bool = True,
) -> list[dict]:
    """
    Search Gmail and automatically download + analyse attachments in PARALLEL.
    """
    messages = search_gmail(query, max_results)
    if not messages:
        return []

    from concurrent.futures import ThreadPoolExecutor
    
    def process_single_msg(msg):
        msg_copy = dict(msg)
        msg_copy["attachments"] = []
        msg_copy["image_analyses"] = []
        msg_copy["doc_analyses"] = []
        try:
            fetch_result = fetch_gmail_attachments(msg["id"], analyse_images=analyse_images)
            attachments = fetch_result["attachments"]
            if fetch_result["body_text"]:
                msg_copy["body_text"] = fetch_result["body_text"][:2000]
            
            msg_copy["attachments"] = [{k: v for k, v in a.items() if k != "data"} for a in attachments]
            
            for att in attachments:
                if att.get("image_analysis") and not att["image_analysis"].get("error"):
                    msg_copy["image_analyses"].append(att["image_analysis"])
                if att.get("doc_analysis") and not att["doc_analysis"].get("error"):
                    msg_copy["doc_analyses"].append(att["doc_analysis"])
        except Exception as e:
            logger.warning(f"[GMAIL_TOOL] Error in parallel fetch for {msg['id']}: {e}")
        return msg_copy

    # Use 4 workers to stay safe on 16GB RAM while boosting speed
    logger.info(f"[GMAIL_TOOL] Starting parallel processing of {len(messages)} messages with 4 workers...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        enriched = list(executor.map(process_single_msg, messages))

    return enriched


# ── Internal helpers ──────────────────────────────────────────────────────────

def _flatten_parts(payload: dict) -> list[dict]:
    """
    Recursively flatten multipart message payload into a list of leaf parts.
    
    Handles:
    - multipart/* : recurse into 'parts' list
    - message/rfc822 : forwarded email — recurse into nested payload's 'parts'
    - leaf parts    : return as-is (text/plain, application/pdf, image/*, etc.)
    """
    parts = []
    mime_type = payload.get("mimeType", "")

    if "parts" in payload:
        # multipart/* — recurse into each child part
        for part in payload["parts"]:
            parts.extend(_flatten_parts(part))
    elif mime_type == "message/rfc822":
        # Forwarded email container — the nested message is in body.attachmentId
        # but its parts are accessible via the nested payload (if present)
        nested = payload.get("body", {})
        if nested.get("attachmentId"):
            # The nested email itself is an attachment; expose it so
            # fetch_gmail_attachments can download and parse it.
            parts.append(payload)
    else:
        # Leaf part (text, pdf, image, etc.)
        parts.append(payload)

    return parts