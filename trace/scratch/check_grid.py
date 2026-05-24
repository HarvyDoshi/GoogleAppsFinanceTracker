from environments.trace_env.tools.sheets_tool import get_sheets_service

def check_grid_data():
    service = get_sheets_service()
    spreadsheet_id = "1PrKMDygYfWleqoSfJQZ3XyOpM62MDvCel0CDIVfUeQY"
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id, includeGridData=True).execute()
    sheets = spreadsheet.get('sheets', [])
    for s in sheets:
        print(f"Sheet: {s['properties']['title']}")
        data = s.get('data', [])
        for grid in data:
            row_data = grid.get('rowData', [])
            print(f"  Rows in grid data: {len(row_data)}")
            for i, row in enumerate(row_data):
                if i < 20:
                    vals = [cell.get('formattedValue', '') for cell in row.get('values', [])]
                    print(f"    Row {i}: {vals}")

if __name__ == "__main__":
    check_grid_data()
