import sys
import os
sys.path.append(os.path.abspath('trace'))
from environments.trace_env.tools.drive_tool import search_drive

results = search_drive("name = 'finance data' and mimeType = 'application/vnd.google-apps.spreadsheet'")
print("Found sheets:", results)
