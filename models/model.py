import torch
from transformers import AutoModel

def load_model(model_name):
    # Load the pre-trained MMS model
    model = AutoModel.from_pretrained(model_name)
    return model

if __name__ == "__main__":
    model_name = "your_model_name"  # Update with actual model name
    model = load_model(model_name)
    print(f"Loaded model: {model_name}")
