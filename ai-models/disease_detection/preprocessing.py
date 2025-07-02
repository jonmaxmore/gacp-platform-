# ai-models/disease_detection/preprocessing.py

import numpy as np
from PIL import Image

def preprocess_image(image_path, image_size=224):
    # Load image
    img = Image.open(image_path)
    
    # Resize with aspect ratio preservation
    ratio = min(image_size / img.width, image_size / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size, Image.LANCZOS)
    
    # Center crop
    left = (new_size[0] - image_size) / 2
    top = (new_size[1] - image_size) / 2
    right = (new_size[0] + image_size) / 2
    bottom = (new_size[1] + image_size) / 2
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy array
    img_array = np.array(img).astype(np.float32)
    
    # Apply channel-wise normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_array = (img_array / 255.0 - mean) / std
    
    return np.expand_dims(img_array, axis=0)