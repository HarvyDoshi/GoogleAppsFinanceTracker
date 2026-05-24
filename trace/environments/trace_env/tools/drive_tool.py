"""
environments/trace_env/tools/drive_tool.py

Google Drive integration — searches the user's Drive for files.
Handles credential loading with proper path resolution and token refresh.
"""

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os, pickle

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Resolve credential paths relative to the project root (trace/ directory)
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_CREDENTIALS_PATH = os.path.join(_PROJECT_ROOT, "credentials.json")
_TOKEN_PATH = os.path.join(_PROJECT_ROOT, "token_drive.pkl")


def get_drive_service():
    creds = None
    if os.path.exists(_TOKEN_PATH):
        with open(_TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)

    # Refresh expired credentials or run auth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
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

    return build("drive", "v3", credentials=creds)


def search_drive(query: str, max_results: int = 10) -> list[dict]:
    """
    Search Drive files.
    query examples: "name contains 'receipt'", "modifiedTime > '2022-01-01'"
    """
    try:
        service = get_drive_service()
        results = service.files().list(
            q=query,
            pageSize=max_results,
            fields="files(id, name, mimeType, createdTime, modifiedTime, size)"
        ).execute()

        files = results.get("files", [])
        return [
            {
                "id": f["id"],
                "filename": f["name"],
                "type": f.get("mimeType", ""),
                "created": f.get("createdTime", ""),
                "size_kb": int(f.get("size", 0)) // 1024,
            }
            for f in files
        ]
    except FileNotFoundError as e:
        print(f"[DRIVE_TOOL] {e}")
        return []
    except Exception as e:
        print(f"[DRIVE_TOOL] Error searching Drive: {e}")
        return []