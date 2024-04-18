import torch
from PIL import Image
import numpy as np
import os
import glob as glob
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from torchvision.transforms import functional as F

from xml.etree import ElementTree as et
from config import (
    CLASSES, RESIZE_TO, TRAIN_DIR, BATCH_SIZE, VISUALIZE_WITH_BBOX,VISUALIZE_WITHOUT_BBOX
)
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform


import glob
import matplotlib.pyplot as plt
from xml.etree import ElementTree as et

class Dataset(Dataset):
    """
    Dataset class for car parking detection using the PKLOT dataset.
    This class is responsible for loading, preprocessing, and visualizing the data.
    """
    
    def __init__(self, dir_path, width, height, classes, transforms=None):
        """
        Initialize the dataset.
        
        Parameters:
        - dir_path (str): Path to the directory containing the dataset.
        - width (int): Width to resize the images.
        - height (int): Height to resize the images.
        - classes (list): List of class names.
        - transforms (callable, optional): Optional transforms to be applied on the image.
        """
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = self._get_all_image_paths()
        self.all_images = sorted([image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths])

    def _get_all_image_paths(self):
        """Helper function to retrieve all image paths."""
        all_paths = []
        for file_type in self.image_file_types:
            all_paths.extend(glob.glob(os.path.join(self.dir_path, file_type)))
        return all_paths

    def _get_image_annotations(self, image_name):
        """
        Extracts and corrects bounding box annotations from XML files.
        
        Args:
            image_name (str): Name of the image file.
        
        Returns:
            list: List of bounding box coordinates.
            list: List of labels corresponding to each bounding box.
        """
        # Derive the annotation filename from the image filename and get its path.
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # Extract image width and height from the XML. 
       
        size = root.find('size')
        if size is None:
            raise ValueError(f"XML {annot_filename} doesn't contain 'size' tag.")
        
        image_width = int(size.find('width').text)
        image_height = int(size.find('height').text)

        # Iterate through each 'object' in the XML to extract bounding box and labels.
        for member in root.findall('object'):
            # Map label names to their respective indices.
            labels.append(self.classes.index(member.find('name').text))
            
            # Extract bounding box coordinates.
            bndbox = member.find('bndbox')
            if bndbox is None:
                continue  # Skip if no bounding box found
            
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Adjust bounding box coordinates based on the resized dimensions.
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            # Ensure the coordinates are within the bounds of the resized image.
            if xmax_final > self.width:
                xmax_final = self.width
            if ymax_final > self.height:
                ymax_final = self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        return boxes, labels


    def _process_image(self, image_path):
        # Read the image
        image = Image.open(image_path)
        image = image.convert("RGB")  # Ensure the image is in RGB format

        # Resize the image using PIL
        image_resized = image.resize((self.width, self.height))
        
        # Convert the PIL image back to a numpy array
        image_resized = np.array(image_resized)

        # Normalize the image
        image_resized = image_resized.astype(np.float32) / 255.0

        return image_resized

    def _visualize_image_without_bbox(self, image, idx):
        """Helper function to visualize and save a processed image without bounding boxes."""
        # plt.imshow(image)
        # plt.imshow(image.permute(1, 2, 0))

        plt.axis('off')
        if VISUALIZE_WITHOUT_BBOX:

            filename = f"image_without_bbox_{idx}.png"
            output_dir = "outputs/images_without_bb"
            # plt.savefig(f"outputs/images_bb/1")
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
            plt.show()
        plt.close()

    def _visualize_image_with_bbox(self, image, idx, boxes):
        """Helper function to visualize and save a processed image with bounding boxes."""
        # plt.imshow(image)
        # plt.imshow(image.permute(1, 2, 0))

        plt.axis('off')
        for box in boxes:
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                  linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
        if VISUALIZE_WITH_BBOX:

            filename = f"image_with_bbox_{idx}.png"
            output_dir = "outputs/images_with_bb"
            # plt.savefig(f"outputs/images_bb/1")
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
            plt.show()
        plt.close()

    def __getitem__(self, idx):
        """Retrieve an item from the dataset."""
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        image_resized = self._process_image(image_path)
        
        # Get annotations
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)
        boxes, labels = self._get_image_annotations(annot_file_path)
        
        
        # Bounding box to tensor.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Apply the image transforms.
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        
        self._visualize_image_without_bbox(image_resized, idx)
        self._visualize_image_with_bbox(image_resized, idx, boxes)
        
        return image_resized, target
    
    def set_resolution(self, width, height):
        # Method to dynamically set the resolution for multi-resolution training
        self.width = width
        self.height = height

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.all_images)  



# Prepare the final datasets and data loaders.
def create_train_dataset(DIR):
    train_dataset = Dataset(
        DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform()
    )
    return train_dataset
def create_valid_dataset(DIR):
    valid_dataset = Dataset(
        DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform()
    )
    return valid_dataset
def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return train_loader
def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return valid_loader

if __name__ == '__main__':
    # For standalone execution to visualize samples
    
    # Instantiate the dataset
    dataset = Dataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of images in the dataset: {len(dataset)}")

    # Visualize a few random samples
    NUM_SAMPLES_TO_VISUALIZE = 5  # number of samples to visualize
    indices_to_visualize = random.sample(range(len(dataset)), NUM_SAMPLES_TO_VISUALIZE)
    
    for idx in indices_to_visualize:
        image, target = dataset[idx]  # this will also visualize the samples due to the logic in __getitem__
        if VISUALIZE_WITH_BBOX:
            dataset._visualize_image_with_bbox(image,idx, target['boxes'])
        if VISUALIZE_WITHOUT_BBOX:
            dataset._visualize_image_without_bbox(image,idx)

def extract_box_dimensions(dataset):
    widths = []
    heights = []
    for _, target in dataset:
        for box in target["boxes"]:
            xmin, ymin, xmax, ymax = box
            widths.append(xmax - xmin)
            heights.append(ymax - ymin)
    return widths, heights

def compute_kmeans_aspect_ratios():
    # Load your dataset
    dataset = Dataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    ) 
    widths, heights = extract_box_dimensions(dataset)

    # Normalize the Dimensions
    image_size = 640  # input size is
    widths = np.array(widths) / image_size
    heights = np.array(heights) / image_size

    # K-means Clustering
    X = np.column_stack((widths, heights))
    kmeans = KMeans(n_clusters=5)  # Using 5 clusters as an example
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_

    # Determine Aspect Ratios
    aspect_ratios = centroids[:, 1] / centroids[:, 0]

    print("Determined Aspect Ratios:", aspect_ratios)
    return aspect_ratios,centroids



def visualize_default_boxes(dataset, centroids, num_images=5):
    """
    Visualize the determined default bounding boxes on some random images from the dataset.
    
    Parameters:
    - dataset: The dataset containing the images.
    - centroids: The centroids determined from K-means clustering.
    - num_images: Number of random images to visualize.
    """
    fig, axs = plt.subplots(1, num_images, figsize=(20, 5))
    
    for ax in axs:
        # Randomly select an image from the dataset
        img, _ = dataset[random.randint(0, len(dataset) - 1)]
        
        # print(f"Image tensor shape: {img.shape}")  # Debugging print statement
        
        # Check if img is a numpy array and convert to tensor if needed
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        
        # If the image tensor has an unexpected batch-like dimension, remove it
        if len(img.shape) == 4:
            img = img.squeeze(0)
        
        # Convert the tensor to CHW format if it's in HWC format
        if len(img.shape) == 3 and img.shape[2] in [1, 3, 4]:
            img = img.permute(2, 0, 1)
        
        # Ensure the image tensor is of type torch.uint8
        if img.dtype != torch.uint8:
            img = (img * 255).byte()
        
        img = F.to_pil_image(img)
        ax.imshow(img)

        # Overlay the default bounding boxes
        img_width, img_height = img.size
        for width, height in centroids:
            rect = patches.Rectangle(
                ((img_width - img_width * width) / 2, (img_height - img_height * height) / 2), 
                img_width * width, 
                img_height * height,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
        
        ax.axis('off')

    plt.savefig(f"outputs/visualize_default_boxes3")
    plt.show()


