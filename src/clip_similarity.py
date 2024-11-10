import cv2
import torch
from transformers import CLIPImageProcessor, CLIPModel
from PIL import Image

from torchmetrics.functional.multimodal import clip_score
from functools import partial

# Load the CLIP model
model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID)
clip_score_fn = partial(clip_score, model_name_or_path=model_ID)

preprocess = CLIPImageProcessor.from_pretrained(model_ID)

# Define a function to load an image and preprocess it for CLIP
def load_and_preprocess_image(image_path):
    # Load the image from the specified path
    image = Image.open(image_path)

    # Apply the CLIP preprocessing to the image
    image = preprocess(image, return_tensors="pt")

    # Return the preprocessed image
    return image


def clip_sim(a, b):
    

    # Load the two images and preprocess them for CLIP
    image_a = load_and_preprocess_image(a)["pixel_values"]
    image_b = load_and_preprocess_image(b)["pixel_values"]

    # Calculate the embeddings for the images using the CLIP model
    with torch.no_grad():
        embedding_a = model.get_image_features(image_a)
        embedding_b = model.get_image_features(image_b)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)

    return similarity_score.item()


def calculate_clip_score(image_path, prompts):
    img = cv2.imread(image_path).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(img).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)
