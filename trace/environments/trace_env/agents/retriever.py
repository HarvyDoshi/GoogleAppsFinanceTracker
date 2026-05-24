"""
environments/trace_env/agents/retriever.py

RetrieverAgent — fetches data from virtual or real sources.
Supports:
  - "gmail"  : real Gmail API search (+ optional image attachment analysis)
  - "drive"  : real Google Drive API search
  - "image"  : direct image analysis via LLaMA 4 Scout (HF Inference API)
  - simulated: world model query (when retrieval_mode != "real")
"""

from environments.trace_env.tools.gmail_tool import search_gmail, search_gmail_with_attachments
from environments.trace_env.tools.drive_tool import search_drive
import yaml
import os
import logging

logger = logging.getLogger(__name__)


def _load_config():
    try:
        with open("configs/env_config.yaml") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


class RetrieverAgent:
    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("retrieval_mode", "simulated")

        # Image analysis config
        img_cfg = config.get("image_analysis", {})
        self._image_analysis_enabled = img_cfg.get("enabled", True)
        self._analyse_gmail_attachments = img_cfg.get("analyse_gmail_attachments", True)

        # Document extraction config
        doc_cfg = config.get("document_extraction", {})
        self._doc_extraction_enabled = doc_cfg.get("enabled", True)

        # Configure image_tool at startup if enabled
        if self._image_analysis_enabled:
            try:
                from environments.trace_env.tools.image_tool import configure as configure_image_tool
                configure_image_tool(img_cfg)
            except Exception as e:
                logger.warning(f"[RETRIEVER] Could not configure image_tool: {e}")

        # Configure doc_tool at startup if enabled
        if self._doc_extraction_enabled:
            try:
                from environments.trace_env.tools.doc_tool import configure as configure_doc_tool
                configure_doc_tool(doc_cfg)
            except Exception as e:
                logger.warning(f"[RETRIEVER] Could not configure doc_tool: {e}")

    def fetch(self, source: str, query: str, world_model=None, metadata=None) -> list[dict]:
        """
        Fetch data from a named source.

        source options:
          "gmail"  — search Gmail (+ auto image-attachment analysis if enabled)
          "drive"  — search Google Drive
          "image"  — analyse a single image (query = URL or local path)
          other    — query the simulated world model
        """
        if self.mode == "real":
            return self._fetch_real(source, query, world_model, metadata)
        else:
            return world_model.query(source, query, limit=10) if world_model else []

    # ── Real-mode fetchers ────────────────────────────────────────────────────

    def _fetch_real(self, source: str, query: str, world_model=None, metadata=None) -> list[dict]:
        try:
            if source == "gmail":
                return self._fetch_gmail(query, metadata)

            elif source == "drive":
                return search_drive(query)

            elif source == "image":
                return self._fetch_image(query)

            else:
                logger.warning(f"[RETRIEVER] Unknown source: {source}")
                return []

        except Exception as e:
            logger.error(f"[RETRIEVER] Real fetch failed for '{source}': {e}")
            # Gracefully fall back to simulated world model
            if world_model:
                logger.info(f"[RETRIEVER] Falling back to simulated data for '{source}'")
                return world_model.query(source, query, limit=10)
            return []

    def _fetch_gmail(self, query: str, metadata: dict = None) -> list[dict]:
        """
        Fetch Gmail messages. If image analysis is enabled, also download
        and analyse any image attachments using LLaMA 4 Scout.
        """
        metadata = metadata or {}
        
        # If the action explicitly requests a fast text-only pass, disable attachments.
        if metadata.get("pass") == "quick_text":
            run_attachments = False
        else:
            run_attachments = True

        if run_attachments:
            logger.info(f"[RETRIEVER] Gmail search with image attachment analysis: '{query}'")
            results = search_gmail_with_attachments(
                query,
                analyse_images=self._image_analysis_enabled and self._analyse_gmail_attachments,
            )
        else:
            logger.info(f"[RETRIEVER] Gmail search (text only): '{query}'")
            results = search_gmail(query)

        # Log summary of image analyses found
        total_img = sum(len(r.get("image_analyses", [])) for r in results)
        if total_img:
            logger.info(
                f"[RETRIEVER] Found {total_img} image analysis results "
                f"across {len(results)} messages"
            )

        # Log summary of document analyses found
        total_doc = sum(len(r.get("doc_analyses", [])) for r in results)
        if total_doc:
            logger.info(
                f"[RETRIEVER] Found {total_doc} document extraction results "
                f"across {len(results)} messages"
            )

        return results

    def _fetch_image(self, source: str) -> list[dict]:
        """
        Directly analyse an image by URL or local path.
        Returns a single-item list containing the analysis result.
        """
        if not self._image_analysis_enabled:
            logger.warning("[RETRIEVER] Image analysis is disabled in config.")
            return []

        try:
            from environments.trace_env.tools.image_tool import analyse_image
            logger.info(f"[RETRIEVER] Direct image analysis: {source[:80]}...")
            result = analyse_image(source)

            if result.get("error"):
                logger.error(f"[RETRIEVER] Image analysis error: {result['error']}")
            else:
                logger.info(
                    f"[RETRIEVER] Image analysis complete — "
                    f"extracted {len(result.get('extracted_text', ''))} chars, "
                    f"found {len(result.get('entities', {}).get('amounts', []))} amounts"
                )
            return [result]

        except ImportError as e:
            logger.error(f"[RETRIEVER] image_tool not available: {e}")
            return []