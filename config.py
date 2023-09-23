import torch


NUM_WORKERS = 4 # Number of parallel workers for data loading.
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_DIR = 'C:\Users\Rohan Thorat\Desktop\QMUL\SEM3\Final\CODE\Improved_base_SSD\PKLot.v2-640.voc\train'
# Validation images and XML files directory.
VALID_DIR = 'C:\Users\Rohan Thorat\Desktop\QMUL\SEM3\Final\CODE\Improved_base_SSD\PKLot.v2-640.voc\valid'

# Classes: 0 index is reserved for background.
CLASSES = [
   '__background__','space-empty','space-occupied'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after creating the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False
VISUALIZE_WITH_BBOX = False
VISUALIZE_WITHOUT_BBOX = False
# Hyperparameters
LR = 0.0001             # Learning rate
NUM_EPOCHS = 30        # Number of epochs
BATCH_SIZE = 8        # Batch size
RESIZE_TO = 640        # Resize the image for training and transforms.
MOMENTUM = 0.9         # Momentum for SGD
WEIGHT_DECAY = 0.0005  # Weight decay for regularization
GAMMA = 0.1            # Learning rate decay factor
STEP_SIZE = 5         # How many epochs before decreasing learning rate

# Location to save model and plots.
OUT_DIR = 'outputs'

base_epoch = 5  # Change resolution every 5 epochs
resolutions = [   
 (740, 640), 
 (840,640),
 (940,680),
 (1024, 680), 
]

RESOLUTION_SCHEDULE = {}
for idx, res in enumerate(resolutions):
    epoch = base_epoch * (idx + 1)
  
    RESOLUTION_SCHEDULE[epoch] = res