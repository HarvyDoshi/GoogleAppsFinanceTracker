import os
from environments.trace_env.tools.sheets_tool import get_sheets_service

def check_values():
    service = get_sheets_service()
    spreadsheet_id = "1PrKMDygYfWleqoSfJQZ3XyOpM62MDvCel0CDIVfUeQY"
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range="A:H"
    ).execute()
    values = result.get('values', [])
    print(f"Total rows found: {len(values)}")
    for i, row in enumerate(values):
        print(f"Row {i}: {row}")

if __name__ == "__main__":
    check_values()
