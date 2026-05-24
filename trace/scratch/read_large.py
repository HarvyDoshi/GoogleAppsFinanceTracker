from environments.trace_env.tools.sheets_tool import get_sheets_service

def read_large_range():
    service = get_sheets_service()
    spreadsheet_id = "1PrKMDygYfWleqoSfJQZ3XyOpM62MDvCel0CDIVfUeQY"
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range="A1:Z1000"
    ).execute()
    values = result.get('values', [])
    print(f"Total rows found in A1:Z1000: {len(values)}")
    for i, row in enumerate(values):
        if i < 20 or any("Zomato" in str(cell) for cell in row):
            print(f"Row {i}: {row}")

if __name__ == "__main__":
    read_large_range()
