# SSD Object Detection System

## Overview
This repository provides a modular and scalable pipeline for training, inference, and evaluation of a **Single Shot MultiBox Detector (SSD)** for object detection. Built using **PyTorch** and **Torchvision**, the system is structured to ensure reproducibility, flexibility, and ease of deployment.

## Project Structure

```
car_park_detection/
├── config/                 # Configuration files
│   ├── config.yaml         # Configuration file for training, inference, and evaluation
│
├── src/
│   ├── components/         # Core model and processing modules
│   │   ├── data_ingestion.py  # Data loading and preprocessing
│   │   ├── model.py          # SSD model architecture
│   │   ├── training.py       # Training process
│   │   ├── evaluation.py     # Model evaluation
│   │   ├── inference.py      # Inference process
│   │
│   ├── utils/               # Helper functions
│   │   ├── common.py        # Logging, file handling, etc.
│   │
│   ├── pipeline/            # End-to-end workflow pipelines
│   │   ├── stage_01_data_ingestion.py  # Training pipeline
│   │   ├── stage_02_prepare_base_model.py   # Evaluation pipeline
│   │   ├── stage_03_training.py  # Inference pipeline
│   │   ├── stage_04_evaluation.py  # Inference pipeline
│   │   ├── predict.py  # Inference pipeline
│
│   ├── entity/              # Configuration entities
│   │   ├── config_entity.py  # Data class for configuration management
│   │   ├── model_entity.py   # Data class for model configuration
│
│   ├── constants/           # Global constants
│       ├── constants.py

│   ├── config/              # Contains all configuration
│       ├── configuration.py
│
├── tests/                   # Unit tests for different components
├── requirements.txt         # List of dependencies
├── params.yaml              # Training hyperparameters
├── main.py                  # End-to-end pipeline execution
├── app.py                   # Flask API for web-based interaction
├── README.md                # Documentation
└── LICENSE                  # Licensing information
```

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/car-park-detection.git
cd car-park-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python main.py
```
This will execute all stages, including data ingestion, model training, and evaluation.

### 4. Run inference on new images
```bash
python src/pipeline/predict.py --image_path /path/to/image.jpg
```

### 5. Start the web-based API (Flask)
```bash
python app.py
```

## Configuration (`config.yaml` & `params.yaml`)
- `config.yaml` defines **data paths, logging configurations, and directory structure**.
- `params.yaml` contains **training hyperparameters** like learning rate, batch size, and number of epochs.

## Features
✅ **Pretrained SSD Model**: Uses `torchvision.models.detection.ssd300_vgg16`.
✅ **Modular Codebase**: Ensures easy customization and extension.
✅ **Configurable Pipelines**: Modify training & inference settings using `config.yaml`.
✅ **Web API for Inference**: Deploy with Flask for real-time detection.
✅ **Logging & Debugging**: Integrated logging with structured error handling.

## Results & Evaluation
The trained model outputs:
- **Bounding boxes with class labels**
- **Evaluation metrics (mAP, IoU, precision-recall)**
- **Annotated images with detected objects**

## Contributing
Feel free to **fork, modify, and contribute** to this project. Issues and pull requests are welcome!

## License
This project is licensed under the **MIT License**.

---
For any queries, reach out via [GitHub Issues](https://github.com/your-username/car-park-detection/issues) 🚀

