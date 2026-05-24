from environments.trace_env.tools.sheets_tool import get_sheets_service

def check_id_metadata(id):
    service = get_sheets_service()
    try:
        spreadsheet = service.spreadsheets().get(spreadsheetId=id).execute()
        print(f"ID: {id}")
        print(f"Title: {spreadsheet.get('properties', {}).get('title')}")
        sheets = spreadsheet.get('sheets', [])
        for i, s in enumerate(sheets):
            print(f"Sheet [{i}]: {s.get('properties', {}).get('title')} (ID: {s.get('properties', {}).get('sheetId')})")
    except Exception as e:
        print(f"Error for ID {id}: {e}")

if __name__ == "__main__":
    check_id_metadata("1PrKMDygYfWleqoSfJQZ3XyOpM62MDvCel0CDIVfUeQY")
