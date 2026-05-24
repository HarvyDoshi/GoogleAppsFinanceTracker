from environments.trace_env.tools.sheets_tool import get_sheets_service

def check_new_id():
    service = get_sheets_service()
    spreadsheet_id = "1PjoAjuMJJRaaO5dhffjnu7fuZrZz00ds"
    try:
        spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        print(f"Title: {spreadsheet.get('properties', {}).get('title')}")
        res = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range="A1:Z100").execute()
        vals = res.get('values', [])
        print(f"Rows: {len(vals)}")
        if len(vals) > 3:
            print(f"Row 3: {vals[3]}") # Should be Zomato
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_new_id()
