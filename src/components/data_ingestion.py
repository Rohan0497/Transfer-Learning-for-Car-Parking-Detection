import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from src.config.configuration import load_config

class CarParkDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

def create_dataloader(data_dir, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    dataset = CarParkDataset(data_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    config = load_config()
    train_loader = create_dataloader(config["data"]["train_data_path"], config["data"]["batch_size"])
    print("Data ingestion completed. Number of batches:", len(train_loader))