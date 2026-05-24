import torch
for i in [1, 2, 4]:
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)

import importlib
try:
    mod = importlib.import_module("transformers.generation.utils")
    print("utils imported")
except Exception as e:
    print(f"Error in utils: {e}")
    import traceback
    traceback.print_exc()
