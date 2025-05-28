import os
import requests
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection, AutoImageProcessor, AutoModel
from PIL import Image, ImageDraw, ImageFont
from diffusers.utils import make_image_grid
import glob
import numpy as np
import torch.nn.functional as F

def get_bbox_coordinates(mask):
    """
    Given a 2D mask array with 0 and 1 values, find the bounding box coordinates of the rectangle region with 1 values.

    Args:
    mask (np.ndarray): 2D numpy array of shape (1024, 1024) with 0 and 1 values.

    Returns:
    tuple: (min_row, min_col, max_row, max_col) coordinates of the bounding box.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    min_row = np.where(rows)[0][0]
    max_row = np.where(rows)[0][-1]
    min_col = np.where(cols)[0][0]
    max_col = np.where(cols)[0][-1]
    
    return (min_row, min_col, max_row, max_col)
def get_bbox_coordinates_square(mask):
    """
    Given a 2D mask array with 0 and 1 values, find the bounding box coordinates of the square region with 1 values.

    Args:
    mask (np.ndarray): 2D numpy array of shape (1024, 1024) with 0 and 1 values.

    Returns:
    tuple: (min_row, min_col, max_row, max_col) coordinates of the square bounding box.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    min_row = np.where(rows)[0][0]
    max_row = np.where(rows)[0][-1]
    min_col = np.where(cols)[0][0]
    max_col = np.where(cols)[0][-1]
    
    # Calculate the width and height of the current bounding box
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    # Determine the size of the square bounding box
    side_length = max(height, width)
    
    # Adjust the bounding box to be a square
    max_row = min_row + side_length - 1
    max_col = min_col + side_length - 1
    
    # Ensure that the bounding box does not exceed the image boundaries
    max_row = min(max_row, 1024 - 1)
    max_col = min(max_col, 1024 - 1)
    
    return (min_row, min_col, max_row, max_col)
def cosine_similarity_tensors(tensor_a, tensor_b):
    """
    Calculate the cosine similarity between two tensors.

    Parameters:
    tensor_a (torch.Tensor): A tensor of shape (a, 768)
    tensor_b (torch.Tensor): A tensor of shape (b, 768)

    Returns:
    torch.Tensor: A tensor of shape (a, b) containing the cosine similarities.
    """
    # Normalize the tensors along the last dimension
    a_normalized = F.normalize(tensor_a, p=2, dim=1)
    b_normalized = F.normalize(tensor_b, p=2, dim=1)

    # Compute the cosine similarity matrix
    cosine_similarity_matrix = torch.mm(a_normalized, b_normalized.T)

    return cosine_similarity_matrix

owlv2_path = "google/owlv2-base-patch16-ensemble"
dinov2_path = "facebook/dinov2-base"


class eval_with_dino:
    def __init__(self, device='cuda'):
        self.processor = Owlv2Processor.from_pretrained(owlv2_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(owlv2_path).to(device)
        self.dino_preprocessor = AutoImageProcessor.from_pretrained(dinov2_path)
        self.dino_model = AutoModel.from_pretrained(dinov2_path).to(device)
        self.device = device
        self.query_dict = {}

    def get_scores(self, images, mask=None, owl_threshold=0.05, owl_query="logo"):

        # crop images in masked region
        if mask:
            if isinstance(mask, str):
                mask = Image.open(mask)
            bbox = get_bbox_coordinates_square(mask)
            bbox = (bbox[1], bbox[0], bbox[3], bbox[2]) # flip for PIL crop

        if isinstance(images[0], str):
            images = [Image.open(image_path) for image_path in images]
            
        if mask:
            images = [image.crop(bbox) for image in images]

        # detect objects in cropped images
        logo_regions = []
        detected_objects = []
        metadata = {}
        for image in images:
            query = [owl_query]
            text = query

            inputs = self.processor(text=query, images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            target_sizes = torch.Tensor([image.size[::-1]])
            
            results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=owl_threshold)
            boxes, scores, labels = results[0]["boxes"].tolist(), results[0]["scores"].tolist(), results[0]["labels"].tolist()
                
            results = {}
            bboxes = []
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                box = [round(i, 2) for i in box]
                bboxes.append(box)

            image_ = image.copy()
            draw = ImageDraw.Draw(image_)

            region = []
            for box, label in zip(boxes, labels):
                label_ = text[label]
                draw.rectangle(box, outline="green", width=10)
                region.append(image.crop(box))
                draw.text((box[0], box[1]), label_, fill="black", font=ImageFont.load_default(50))

            logo_regions.append(region)
            detected_objects.append(image_)

        similarities = []
        max_similarities = []
        with torch.no_grad():
            for crop in logo_regions:
                # crop: list of cropped regions
                crop_embeddings = []
                for region in crop:
                    dino_input = self.dino_preprocessor(images=region, return_tensors='pt').to(self.device)
                    emb = self.dino_model(**dino_input).last_hidden_state.mean(dim=1) # 1, 768
                    crop_embeddings.append(emb.to('cpu'))
                if crop_embeddings:
                    crop_embeddings = torch.cat(crop_embeddings, dim=0) # n_crops, 768
                    # compute cosine similarity
                    cosine_similarity = cosine_similarity_tensors(crop_embeddings.to(self.device), self.query_dict['ref_embeddings']).to('cpu')
                    # max_similarity = cosine_similarity.max()
                    similarity = cosine_similarity.max(dim=1).values
                    similarities.append(similarity.tolist())
                    max_similarity = similarity.max()
                    max_similarities.append(max_similarity.item())
                else:
                    similarities.append([0])
                    max_similarities.append(0)

        metadata['logo_regions'] = logo_regions
        metadata['detected_objects'] = detected_objects
        metadata['similarities'] = similarities

        return max_similarities, metadata
    
    def query_dict_update(self, ref_images):
        if isinstance(ref_images[0], str):
            ref_images = [Image.open(ref_image).convert("RGB") for ref_image in ref_images]
        ref_embeddings = []
        for ref in ref_images:
            dino_input = self.dino_preprocessor(images=ref, return_tensors='pt').to(self.device)
            emb = self.dino_model(**dino_input).last_hidden_state.mean(dim=1) # 1, 768
            ref_embeddings.append(emb)
        ref_embeddings = torch.cat(ref_embeddings, dim=0) 
        self.query_dict['ref_embeddings'] = ref_embeddings # n_refs, 768 