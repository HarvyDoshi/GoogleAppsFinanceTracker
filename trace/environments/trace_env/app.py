"""
environments/trace_env/app.py

FastAPI server — the OpenEnv-standard HTTP interface for TraceEnv.

Endpoints follow the OpenEnv spec:
  POST /reset         → TraceObservation
  POST /step          → {observation, reward, done, info}
  GET  /state         → EpisodeState
  GET  /health        → {"status": "ok"}
  POST /analyse_image → Full VLM image analysis (LLaMA 4 Scout)

Deploy on HuggingFace Spaces or run locally:
  uvicorn environments.trace_env.app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import yaml, os, logging

logger = logging.getLogger(__name__)

try:
    from .core.env import TraceEnv
    from .core.schemas import TraceAction, TraceObservation
except ImportError:
    from core.env import TraceEnv
    from core.schemas import TraceAction, TraceObservation


# ── Load config ──────────────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get("TRACE_CONFIG", "configs/env_config.yaml")
try:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config = {}


# ── Configure image_tool at startup (if enabled) ─────────────────────────────
_img_cfg = config.get("image_analysis", {})
if _img_cfg.get("enabled", True):
    try:
        from environments.trace_env.tools.image_tool import configure as _cfg_img
        _cfg_img(_img_cfg)
        logger.info("[APP] image_tool configured successfully")
    except Exception as _e:
        logger.warning(f"[APP] image_tool configuration skipped: {_e}")

# ── One env instance per server (single-session demo mode) ───────────────────
env = TraceEnv(config)

app = FastAPI(
    title="Trace — OpenEnv Environment",
    description="Federated digital-life RL environment for long-horizon planning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/response models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    instruction: str
    difficulty: str = "easy"
    available_sources: list[str] = ["gmail", "image"]
    ground_truth: dict = {}


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    img_cfg = config.get("image_analysis", {})
    return {
        "status": "ok",
        "env": "TraceEnv",
        "version": "1.0.0",
        "capabilities": {
            "image_analysis": img_cfg.get("enabled", True),
            "image_model":    img_cfg.get("model_id", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
            "gmail_attachments": img_cfg.get("analyse_gmail_attachments", True),
        },
    }


@app.post("/reset", response_model=dict)
def reset(req: ResetRequest):
    """Start a new episode."""
    task = req.dict()
    obs = env.reset(task)
    return obs.dict()


@app.post("/step", response_model=StepResponse)
async def step(action: TraceAction):
    """Execute one agent action (Async to prevent blocking)."""
    try:
        import anyio
        obs, reward, done, info = await anyio.to_thread.run_sync(env.step, action)
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation=obs.dict(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def state():
    """Return full episode state for debugging."""
    s = env.state()
    if s is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return {
        "episode_id": s.episode_id,
        "steps": s.steps,
        "plan": s.plan,
        "retrieved_count": len(s.retrieved_data),
        "verified": s.verified,
        "done": s.done,
    }


@app.get("/observation_prompt")
def observation_prompt():
    """Return current observation as a ready-to-use LLM prompt string."""
    s = env.state()
    if s is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    obs = env._build_obs("(current state)")
    return {"prompt": obs.to_prompt()}


# ── Image Analysis Endpoint ──────────────────────────────────────────────────

class ImageAnalyseRequest(BaseModel):
    """
    Request body for POST /analyse_image.

    source_type: "url"    — http/https URL or data: URL
                 "path"   — absolute local file path on the server
                 "base64" — raw base64-encoded image bytes
    data:        The image reference (URL string, path string, or base64 string).
    question:    Optional specific question to ask about the image.
                 If omitted, runs full OCR + entity extraction.
    """
    source_type: Optional[str] = None      # auto-detected if None
    data: str                              # URL / path / base64
    question: Optional[str] = None


@app.post("/analyse_image")
def analyse_image(req: ImageAnalyseRequest):
    """
    Analyse an image using LLaMA 4 Scout (HuggingFace Inference API).

    Returns extracted text, entities (amounts, dates, vendors, items),
    and a natural language summary of the image content.

    Requires HF_TOKEN environment variable set to your HuggingFace token.

    Example request:
    {
        "data": "https://example.com/receipt.jpg",
        "question": "What is the total amount on this receipt?"
    }
    """
    img_cfg = config.get("image_analysis", {})
    if not img_cfg.get("enabled", True):
        raise HTTPException(
            status_code=503,
            detail="Image analysis is disabled. Set image_analysis.enabled: true in env_config.yaml."
        )

    try:
        from environments.trace_env.tools.image_tool import analyse_image as _analyse
        result = _analyse(
            source=req.data,
            question=req.question,
            source_type=req.source_type,
        )

        # Also inject result into the active world model if episode is running
        if env.state() is not None:
            env.world_model.inject_image_analysis(result)
            logger.info(f"[APP] Image analysis injected into world model: {result['id']}")

        if result.get("error"):
            raise HTTPException(
                status_code=422,
                detail=f"Image analysis failed: {result['error']}"
            )
        return result

    except HTTPException:
        raise
    except EnvironmentError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e) + " Set HF_TOKEN environment variable."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/analyse_gmail_attachment")
def analyse_gmail_attachment(message_id: str, filename: Optional[str] = None):
    """
    Download and analyse image attachments from a specific Gmail message.

    Args:
        message_id: Gmail message ID (from /step RETRIEVE gmail results).
        filename:   Optional — only analyse the attachment matching this filename.

    Returns:
        List of image analysis results for all image attachments in the message.
    """
    img_cfg = config.get("image_analysis", {})
    if not img_cfg.get("enabled", True):
        raise HTTPException(status_code=503, detail="Image analysis is disabled.")

    try:
        from environments.trace_env.tools.gmail_tool import fetch_gmail_attachments
        attachments = fetch_gmail_attachments(message_id, analyse_images=True)

        results = []
        for att in attachments:
            if filename and att["filename"] != filename:
                continue
            if att.get("image_analysis"):
                # Inject into world model
                if env.state() is not None:
                    env.world_model.inject_image_analysis(att["image_analysis"])
                results.append(att["image_analysis"])

        if not results:
            return {"message": "No image attachments found or analysed.", "attachments_found": len(attachments)}
        return {"analyses": results, "count": len(results)}

    except EnvironmentError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


# ── Document Extraction Endpoint ─────────────────────────────────────────────

@app.post("/extract_document")
async def extract_document_endpoint(
    file: UploadFile = File(...),
    analyse_images: bool = Form(True)
):
    """
    Extract text and embedded images from a document (PDF, DOCX, PPTX).
    """
    try:
        from environments.trace_env.tools.doc_tool import extract_document as _extract
        
        file_bytes = await file.read()
        
        result = _extract(
            file_bytes=file_bytes,
            filename=file.filename,
            mime_type=file.content_type or "",
            analyse_images=analyse_images
        )

        # Inject result into the active world model if episode is running
        if env.state() is not None:
            # We can use the same image analysis injection for any images found
            for img_analysis in result.get("image_analyses", []):
                env.world_model.inject_image_analysis(img_analysis)
            
            # Since world model might need document text too, we could add a method
            # but for now we just log it.
            logger.info(f"[APP] Document extracted: {result['id']}")

        if result.get("error"):
            raise HTTPException(
                status_code=422,
                detail=f"Document extraction failed: {result['error']}"
            )
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
