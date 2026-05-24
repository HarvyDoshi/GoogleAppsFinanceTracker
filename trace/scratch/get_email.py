from environments.trace_env.tools.sheets_tool import get_sheets_service
from googleapiclient.discovery import build

def get_email():
    service = get_sheets_service()
    # Need to use Oauth2 API or similar? 
    # Or just try to share a file and see who the owner is?
    # I'll try to get the user info via people API if enabled? No.
    # I'll try to get the about info from Drive API.
    drive = build('drive', 'v3', credentials=service._http.credentials)
    about = drive.about().get(fields="user").execute()
    print(f"Email: {about.get('user', {}).get('emailAddress')}")

if __name__ == "__main__":
    get_email()
