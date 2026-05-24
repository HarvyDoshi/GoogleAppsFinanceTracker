from environments.trace_env.tools.sheets_tool import get_sheets_service

def check_account():
    service = get_sheets_service()
    # Sheets API doesn't have "who am i".
    # But we can try to get the user's email if we have the right scopes.
    # We don't.
    # Let's just try to create a dummy file and see who the owner is? No.
    print("Service initialized successfully.")

if __name__ == "__main__":
    check_account()
