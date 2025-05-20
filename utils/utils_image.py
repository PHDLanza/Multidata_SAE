import numpy as np
import cv2
from PIL import Image, ImageFilter

def create_image_patches(image_object: str | Image.Image, grid_size=24):
    """
    Convert image to PIL format and split into equal sized patches
    
    Args:
        image_object: Input image as either a file path string or PIL Image object
        grid_size: Number of rows/columns to split image into (default 24x24 grid)
        
    Returns:
        list: List of PIL Image patches
    """
    if isinstance(image_object, Image.Image):
        img = image_object
    else:
        img = Image.open(image_object)
        
    img = img.resize((336, 336))
    width, height = img.size
    
    # Calculate patch dimensions
    patch_width = width // grid_size
    patch_height = height // grid_size
    patches = []

    # Create patches by cropping
    for row in range(grid_size):
        for col in range(grid_size):
            left = col * patch_width
            top = row * patch_height
            right = left + patch_width
            bottom = top + patch_height
            patch = img.crop((left, top, right, bottom))
            patches.append(patch)
            
    return patches

def reconstruct_image(patches, mask, grid_size=24):
    """
    Reconstruct full image from patches using a binary mask
    
    Args:
        patches (list): List of PIL Image patches
        mask (np.array): Binary mask array indicating which patches to include (1) or black out (0)
        grid_size (int): Number of rows/columns in the grid (default 24)
        
    Returns:
        np.array: Reconstructed image as numpy array
    """
    # Calculate patch dimensions from first patch
    patch_width, patch_height = patches[0].size
    
    # Create blank image
    full_width = patch_width * grid_size
    full_height = patch_height * grid_size
    reconstructed = Image.new('RGB', (full_width, full_height))
    
    # Place patches according to mask
    for idx, (patch, mask_val) in enumerate(zip(patches, mask)):
        row = idx // grid_size
        col = idx % grid_size
        
        # Calculate position
        left = col * patch_width
        top = row * patch_height
        
        # If mask is 0, use black patch instead of original
        if mask_val == 0:
            patch = Image.new('RGB', (patch_width, patch_height), 'black')
            
        reconstructed.paste(patch, (left, top))
        
    return np.array(reconstructed)

def combine_images_horizontally(images):
    """
    Combine a list of images horizontally
    
    Args:
        images (list): List of PIL Images to combine
        
    Returns:
        PIL.Image: Combined image
    """
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    
    combined = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
        
    return combined

def reconstruct_image_blurring(patches, mask, grid_size=24):
    """
    Reconstruct full image from patches using a binary mask
    
    Args:
        patches (list): List of PIL Image patches
        mask (np.array): Binary mask array indicating which patches to include (1) or black out (0)
        grid_size (int): Number of rows/columns in the grid (default 24)
        
    Returns:
        np.array: Reconstructed image as numpy array
    """
    # Calculate patch dimensions from first patch
    patch_width, patch_height = patches[0].size
    
    # Create blank image
    full_width = patch_width * grid_size
    full_height = patch_height * grid_size
    reconstructed = Image.new('RGB', (full_width, full_height))
    
    # Place patches according to mask
    for idx, (patch, mask_val) in enumerate(zip(patches, mask)):
        row = idx // grid_size
        col = idx % grid_size
        
        # Calculate position
        left = col * patch_width
        top = row * patch_height
        
        # If mask is 0, use black patch instead of original
        if mask_val == 0:
            patch= patch.filter(ImageFilter.GaussianBlur(radius=5))
            
        reconstructed.paste(patch, (left, top))
        
    return np.array(reconstructed)
