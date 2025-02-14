import torch
from src.components.model import create_model
from src.config.configuration import ConfigurationManager

def run_inference(image_path):
    config = ConfigurationManager().get_inference_config()
    model = create_model(num_classes=2)
    model.load_state_dict(torch.load("outputs/trained_model.pth"))
    model.eval()
    print(f"Inference completed for {image_path}.")

if __name__ == "__main__":
    test_image = "test.jpg"
    run_inference(test_image)