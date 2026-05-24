from environments.trace_env.tools.sheets_tool import get_sheets_service

def add_test_row():
    service = get_sheets_service()
    spreadsheet_id = "1PrKMDygYfWleqoSfJQZ3XyOpM62MDvCel0CDIVfUeQY"
    row = ["Test Date", "Test Vendor", "Test Category", "999.99", "Test Method", "TEST-ID", "Testing from AI", "msg_test_123"]
    body = {'values': [row]}
    result = service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range="A:H",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()
    print(f"Added row. Range: {result.get('updates', {}).get('updatedRange')}")

if __name__ == "__main__":
    add_test_row()
