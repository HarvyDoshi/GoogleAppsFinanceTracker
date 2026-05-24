import torch
import sys

# Shim for missing torch.int1 in torch < 2.6
if not hasattr(torch, "int1"):
    print("Patching torch.int1...")
    torch.int1 = torch.int8 

try:
    import transformers
    print("Transformers imported.")
    from transformers import AutoTokenizer
    print("AutoTokenizer imported!")
except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, "__cause__") and e.__cause__:
        print(f"Cause: {e.__cause__}")
        import traceback
        traceback.print_exception(type(e.__cause__), e.__cause__, e.__cause__.__traceback__)
