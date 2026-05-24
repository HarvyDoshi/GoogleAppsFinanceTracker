import base64
import os

files = ['credentials.json', 'token_gmail.pkl', 'token_sheets.pkl']
out = ""

for f in files:
    if os.path.exists(f):
        # Format the name: credentials.json -> GCP_CREDENTIALS_B64
        name = f.upper().replace('.', '_')
        if "CREDENTIALS" in name:
            name = "GCP_CREDENTIALS_B64"
        else:
            name = name.replace("_PKL", "_B64")
        
        with open(f, 'rb') as file:
            encoded = base64.b64encode(file.read()).decode('utf-8')
            out += f"{name}:\n{encoded}\n\n"

with open('hf_secrets.txt', 'w') as f:
    f.write(out)

print("Successfully created hf_secrets.txt!")
