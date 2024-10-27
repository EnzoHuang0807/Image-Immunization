import os
import cv2
import torch
import argparse

import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

def get_parser():
    parser = argparse.ArgumentParser(description='Mask Generation Arguments')
    
    parser.add_argument('--sam_checkpoint', default='../models/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument('--model_type', default='vit_h', type=str)

    parser.add_argument('--background', action='store_true', help='segment background instead of face')
    parser.add_argument('--input_image', default='../images/elonmusk.webp', type=str, help='the image for segmentation')
    parser.add_argument('--output_mask', default='../images/mask.png', type=str, help='the output mask')

    parser.add_argument('--demo', action='store_true', help='demonstrate segmentation result')
    parser.add_argument('--demo_path', default='../images/demo', type=str, help='the directory to store demonstration images')

    parser.add_argument('--GPU_ID', default='4', type=str)
    return parser.parse_args()

def main():

    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    image = cv2.imread(args.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    yolo_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    yolo = YOLO(yolo_path)

    output = yolo(image)
    boxes = Detections.from_ultralytics(output[0]).xyxy
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    boxes = np.stack([boxes[:, 0] - 0.4 * widths, boxes[:, 1] - 0.4 * heights,
                      boxes[:, 2] + 0.4 * widths, boxes[:, 3] + 0.4 * heights], axis=1)
    boxes[boxes < 0] = 0

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    if args.background:

        input_labels = np.array([1] * boxes.shape[0])
        input_points = np.stack([(boxes[:, 2] + boxes[:, 0]) / 2,
                                 (boxes[:, 3] + boxes[:, 1]) / 2], axis=1)
        
        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

        mask_image = np.where(masks[2], 255, 0).astype(np.uint8)
        plt.imsave(args.output_mask, mask_image, cmap='gray', vmin=0, vmax=255)

    else:
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=True,
        )

        mask_image = np.where(masks[1] | masks[2], 255, 0).astype(np.uint8)
        plt.imsave(args.output_mask, mask_image, cmap='gray', vmin=0, vmax=255)

    if args.demo:
        for i, mask in enumerate(masks):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            
            if args.background:
                show_mask(mask, plt.gca())
                show_points(input_points, input_labels, plt.gca())
            else:
                show_mask(mask, plt.gca())
                show_box(boxes, plt.gca())

            plt.title(f"Mask {i+1}", fontsize=18)
            plt.axis('on')
            plt.savefig(os.path.join(args.demo_path, f"mask_{i+1}.jpg"))


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(boxes, ax):
    for box in boxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

  
if __name__ == '__main__':
    main()  
