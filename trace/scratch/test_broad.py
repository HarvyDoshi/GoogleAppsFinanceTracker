import torch
import sys

def patch_torch():
    for i in [1, 2, 4]:
        name = f"int{i}"
        if not hasattr(torch, name):
            print(f"Patching torch.{name}...")
            setattr(torch, name, torch.int8)

patch_torch()

try:
    from transformers import AutoTokenizer
    print("AutoTokenizer imported!")
    from unsloth import FastLanguageModel
    print("Unsloth imported!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
