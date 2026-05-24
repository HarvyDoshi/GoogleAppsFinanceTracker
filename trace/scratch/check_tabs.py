import os
from environments.trace_env.tools.sheets_tool import get_sheets_service, get_or_create_ledger

def check_tabs():
    service = get_sheets_service()
    spreadsheet_id = get_or_create_ledger(service)
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheets = spreadsheet.get('sheets', [])
    print(f"Spreadsheet Title: {spreadsheet.get('properties', {}).get('title')}")
    for i, sheet in enumerate(sheets):
        print(f"Sheet [{i}]: {sheet.get('properties', {}).get('title')} (ID: {sheet.get('properties', {}).get('sheetId')})")

if __name__ == "__main__":
    check_tabs()
