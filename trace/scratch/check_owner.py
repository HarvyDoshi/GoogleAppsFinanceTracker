from environments.trace_env.tools.sheets_tool import get_sheets_service
from googleapiclient.discovery import build

def check_owner():
    service = get_sheets_service()
    drive = build('drive', 'v3', credentials=service._http.credentials)
    spreadsheet_id = "1PrKMDygYfWleqoSfJQZ3XyOpM62MDvCel0CDIVfUeQY"
    try:
        f = drive.files().get(fileId=spreadsheet_id, fields="owners, name").execute()
        print(f"File Name: {f['name']}")
        for owner in f.get('owners', []):
            print(f"Owner: {owner.get('displayName')} ({owner.get('emailAddress')})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_owner()
