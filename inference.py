import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse

from model import create_model

from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

np.random.seed(42)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', 
    help='path to input image directory',
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    help='image resize shape'
)
parser.add_argument(
    '--threshold',
    default=0.25,
    type=float,
    help='detection threshold'
)
args = vars(parser.parse_args())

# Ensure output directories exist.
os.makedirs('inference_outputs/images', exist_ok=True)

# Blue for "space-empty" and Green for "space-occupied"
COLORS = [[0, 0, 0],[255, 0, 0], [0, 255, 0]]

# Load the best model and trained weights.
model = create_model(num_classes=NUM_CLASSES, size=640)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Retrieve list of test images.
DIR_TEST = args['input']
test_images = glob.glob(f"{DIR_TEST}/*.jpg")
print(f"Test instances: {len(test_images)}")

frame_count = 0  # Counter for frames.
total_fps = 0   # Total frames per second..

for i in range(len(test_images)):
    # Get the image file name for saving output later on.
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()

    # Resize if required.
    if args['imgsz'] is not None:
        image = cv2.resize(image, (args['imgsz'], args['imgsz']))
    print(image.shape)
    # Convert from BGR to RGB and normalize.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0

    # Bring color channels to front (H, W, C) => (C, H, W).
    image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # Convert to tensor.
    image_input = torch.tensor(image_input, dtype=torch.float).cuda()
    # Add batch dimension.
    image_input = torch.unsqueeze(image_input, 0)


    start_time = time.time()
    # Predictions
    with torch.no_grad():
        outputs = model(image_input.to(DEVICE))
    end_time = time.time()

    # Get the current frame per second.
    fps = 1 / (end_time - start_time)
    # Total Frame Per Second till current frame.
    total_fps += fps
    frame_count += 1

    # Load all detections to CPU.
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    
    if len(outputs[0]['boxes']) != 0:

        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        labels = outputs[0]['labels'].cpu().numpy()

        valid_score_indices = scores >= args['threshold']

        boxes = boxes[valid_score_indices].astype(np.int32)
        labels = labels[valid_score_indices]

        pred_classes = [CLASSES[i] for i in labels]

        print(f"Length of boxes: {len(boxes)}")
        print(f"Length of pred_classes: {len(pred_classes)}")
        valid_indices = [i for i, cls in enumerate(pred_classes) if cls != 'background']
        filtered_boxes = [boxes[i] for i in valid_indices]
        filtered_classes = [pred_classes[i] for i in valid_indices]

        draw_boxes = filtered_boxes
        pred_classes = filtered_classes

        # Draw the bounding boxes and write the class name on top of it.
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            
            try:
                color = COLORS[CLASSES.index(class_name)]
            except IndexError:
                print(f"Error: Class '{class_name}' not found in CLASSES list.")
                continue  # Skip the current iteration

            # Rescale boxes.
            xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
            ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
            xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
            ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])
            
            # Draw bounding box
            cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), color[::-1], 3)
            
            # # Draw class label
            # cv2.putText(orig_image, 
            #             class_name, 
            #             (xmin, ymin-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 
            #             0.8, 
            #             color[::-1], 
            #             2, 
            #             lineType=cv2.LINE_AA)


        cv2.waitKey(1)
        cv2.imwrite(f"inference_outputs/images/{image_name}.jpg", orig_image)
    print(f"Image {i+1} done...")
    print('-'*50)

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()
# Calculate and print the average Frame Per Second.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
