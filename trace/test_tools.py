import sys
import os
import json

# Add current dir to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from environments.trace_env.tools import image_tool
from environments.trace_env.tools import doc_tool

def test_image_tool():
    print("================== Testing image_tool ==================")
    try:
        image_tool.configure({
            "router_model": "moondream",
            "financial_model": "qwen2-vl",
            "general_model": "llama3.2-vision",
            "ollama_host": "http://localhost:11434"
        })
        
        dummy_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        print("Calling analyse_image() with a dummy 1x1 image...")
        result = image_tool.analyse_image(source=dummy_base64, source_type="base64")
        print("Result:\n", json.dumps(result, indent=2))
        
    except Exception as e:
        print("Error during image_tool testing:", e)

def test_doc_tool():
    print("\n================== Testing doc_tool ==================")
    try:
        doc_tool.configure({
            "analyse_embedded_images": False,
        })
        
        dummy_content = b"Hello, this is a test document."
        print("Calling extract_document() with a dummy file...")
        result = doc_tool.extract_document(
            file_bytes=dummy_content,
            filename="test_document.txt",
            mime_type="text/plain",
            analyse_images=False
        )
        print("Result:\n", json.dumps(result, indent=2))
        
    except Exception as e:
        print("Error during doc_tool testing:", e)

if __name__ == "__main__":
    test_image_tool()
    test_doc_tool()
