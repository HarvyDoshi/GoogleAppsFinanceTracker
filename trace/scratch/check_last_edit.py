from environments.trace_env.tools.sheets_tool import get_sheets_service

def check_last_edit():
    service = get_sheets_service()
    spreadsheet_id = "1PrKMDygYfWleqoSfJQZ3XyOpM62MDvCel0CDIVfUeQY"
    # Sheets API doesn't have last edit. Need Drive API.
    # I'll just check the spreadsheet properties again.
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    print(f"Spreadsheet Title: {spreadsheet.get('properties', {}).get('title')}")
    print(f"Spreadsheet ID: {spreadsheet.get('spreadsheetId')}")

if __name__ == "__main__":
    check_last_edit()
