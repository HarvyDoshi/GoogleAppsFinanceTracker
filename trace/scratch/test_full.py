import torch
for i in range(1, 8):
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)

import importlib
try:
    mod = importlib.import_module("transformers.generation.utils")
    print("utils imported")
    from transformers import AutoTokenizer
    print("AutoTokenizer imported!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
