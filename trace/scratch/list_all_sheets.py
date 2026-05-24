from environments.trace_env.tools.sheets_tool import get_sheets_service

def list_all_sheets():
    service = get_sheets_service()
    from googleapiclient.discovery import build
    drive = build('drive', 'v3', credentials=service._http.credentials)
    
    results = drive.files().list(q="mimeType='application/vnd.google-apps.spreadsheet'", fields="files(id, name)").execute()
    files = results.get('files', [])
    for f in files:
        print(f"{f['name']} | ID: {f['id']}")

if __name__ == "__main__":
    list_all_sheets()
