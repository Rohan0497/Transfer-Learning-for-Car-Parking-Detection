import torch
from src.components.model import create_model
from src.config.configuration import load_config

def evaluate():
    config = load_config()
    model = create_model(num_classes=2)
    model.load_state_dict(torch.load("outputs/trained_model.pth"))
    model.eval()
    print("Model evaluation completed.")

if __name__ == "__main__":
    evaluate()
