from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    input_size: tuple
    num_classes: int
    learning_rate: float
    batch_size: int
    epochs: int
    model_save_path: Path