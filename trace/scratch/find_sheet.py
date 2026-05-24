from environments.trace_env.tools.sheets_tool import get_sheets_service

def search_by_title():
    service = get_sheets_service()
    # Use drive API to search for spreadsheets
    drive_service = build_drive_service(service._http) # Need to create drive service
    # Actually just search via sheets? No, sheets API doesn't have list spreadsheets.
    # We use Drive API v3.
    pass

from googleapiclient.discovery import build
def build_drive_service(creds):
    return build('drive', 'v3', credentials=creds)

def main():
    service = get_sheets_service()
    drive = build_drive_service(service._http.credentials) # service._http.credentials might work
    
    query = "mimeType='application/vnd.google-apps.spreadsheet' and name contains 'Dummy_Sheet'"
    results = drive.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    
    if not files:
        print("No files found with 'Dummy_Sheet' in title.")
    else:
        for f in files:
            print(f"Found: {f['name']} (ID: {f['id']})")

if __name__ == "__main__":
    main()
