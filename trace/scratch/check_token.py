import pickle
import os

def check_token():
    token_path = "token_sheets.pkl"
    if not os.path.exists(token_path):
        print("token_sheets.pkl does not exist.")
        return
    
    try:
        with open(token_path, "rb") as f:
            creds = pickle.load(f)
        print(f"Token loaded successfully.")
        print(f"Valid: {creds.valid}")
        print(f"Expired: {creds.expired}")
        print(f"Scopes: {creds.scopes}")
    except Exception as e:
        print(f"Error loading token: {e}")

if __name__ == "__main__":
    check_token()
