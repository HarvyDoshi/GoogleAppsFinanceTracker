import torch
if not hasattr(torch, "int1"):
    print("Patching torch.int1...")
    torch.int1 = torch.int8 

try:
    from transformers.generation import GenerationMixin
    print("GenerationMixin imported!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
