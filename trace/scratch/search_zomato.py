from environments.trace_env.tools.sheets_tool import get_sheets_service

def search_zomato():
    service = get_sheets_service()
    spreadsheet_id = "1PrKMDygYfWleqoSfJQZ3XyOpM62MDvCel0CDIVfUeQY"
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range="A:Z" # Broader range
    ).execute()
    values = result.get('values', [])
    for i, row in enumerate(values):
        if any("Zomato" in str(cell) for cell in row):
            print(f"Found Zomato at Row {i}: {row}")
            return
    print("Zomato not found in the spreadsheet.")

if __name__ == "__main__":
    search_zomato()
