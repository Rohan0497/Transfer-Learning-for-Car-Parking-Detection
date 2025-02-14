import torch
from src.components.model import create_model
from src.components.data_ingestion import create_dataloader
from src.config.configuration import load_config

def train():
    config = load_config()
    model = create_model(num_classes=2).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    train_loader = create_dataloader(config["data"]["train_data_path"], config["data"]["batch_size"])
    
    for epoch in range(config["model"]["epochs"]):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), "outputs/trained_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()