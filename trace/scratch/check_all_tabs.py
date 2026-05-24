import os
from environments.trace_env.tools.sheets_tool import get_sheets_service

def check_all_tabs():
    print("Checking if there are other sheets in the same file...")
    service = get_sheets_service()
    spreadsheet_id = "1PrKMDygYfWleqoSfJQZ3XyOpM62MDvCel0CDIVfUeQY"
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheets = spreadsheet.get('sheets', [])
    for s in sheets:
        title = s['properties']['title']
        print(f"Sheet title: {title}")
        res = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=f"'{title}'!A1:Z100").execute()
        vals = res.get('values', [])
        print(f"  Rows in {title}: {len(vals)}")
        if len(vals) > 1:
            print(f"  First data row: {vals[1]}")

if __name__ == "__main__":
    check_all_tabs()
