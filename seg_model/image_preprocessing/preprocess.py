import numpy as np
import cv2
from typing import Optional
from PIL import Image

def smart_pad_and_resize_ecg(image, target_size=(224, 224),
                            resizing_strategy = cv2.INTER_LANCZOS4):
    """
    Smarter padding that detects background color automatically
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Auto-detect background color (assume corners are background)
    corners = [
        image[0, 0], image[0, -1], 
        image[-1, 0], image[-1, -1]
    ]
    
    # Use most common corner color as background
    if len(image.shape) == 3:
        bg_color = np.median(corners, axis=0).astype(image.dtype)
    else:
        bg_color = np.median(corners).astype(image.dtype)
    
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    # Create padded canvas
    if len(image.shape) == 3:
        padded = np.full((max_dim, max_dim, image.shape[2]), bg_color, dtype=image.dtype)
    else:
        padded = np.full((max_dim, max_dim), bg_color, dtype=image.dtype)
    
    # Center the image
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = image

    # sharpen image
    kernel = np.array(
                    [[-0.5, -0.5, -0.5],
                    [-0.5,  5.0, -0.5],
                    [-0.5, -0.5, -0.5]]
                    )

    sharpened = cv2.filter2D(padded, -1, kernel)
    # Blend original and sharpened (subtle effect)
    enhanced = cv2.addWeighted(padded, 0.7, sharpened, 0.3, 0)
    
    # Resize with appropriate interpolation
    resized = cv2.resize(padded, target_size, interpolation=resizing_strategy)
    
    # crop to remove the now irrelevant background
    
    # Calculate where actual content is after resize
    scale_factor = target_size[0] / max_dim
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    # Calculate crop bounds in resized image
    y_start = (target_size[0] - new_h) // 2
    x_start = (target_size[1] - new_w) // 2
    
    cropped = resized[y_start:y_start + new_h, x_start:x_start + new_w]
    
    return cropped

def unsharp_mask(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.5, threshold: int = 0) -> np.ndarray:
    """
    Apply unsharp masking to enhance image sharpness.
    """
    
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Create mask by subtracting blurred from original
    mask = cv2.subtract(image, blurred)
    
    # Apply threshold to mask if specified
    if threshold > 0:
        mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Add weighted mask back to original
    sharpened = cv2.addWeighted(image, 1.0, mask, amount, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to enhance local contrast.
    """
    if len(image.shape) == 3:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:,:,0] = clahe.apply(lab[:,:,0])  # Apply only to L channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
    
    return enhanced

def generate_output(image: np.ndarray, corners: np.ndarray, scale: Optional[tuple] = None): 
    """
    Generates a perspective-transformed output image with enhancement.
    """
    corners = order_points(corners)

    if scale is not None:
        corners *= np.array(scale, dtype=np.float32)

    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(corners, destination_corners)

    max_width, max_height = map(int, destination_corners[2])
    out = cv2.warpPerspective(image, M, (max_width, max_height), flags=cv2.INTER_LANCZOS4)

    # Step 1: Enhance contrast to make faint traces more visible
    out = enhance_contrast(out, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Step 2: Sharpen the now-visible traces
    out = unsharp_mask(out, kernel_size=5, sigma=1.0, amount=1.2)
    
    return np.clip(out, 0, 255).astype(np.uint8)

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Rearranges points to: top-left, top-right, bottom-right, bottom-left.
    """
    pts = np.asarray(pts, dtype=np.float32)  # Ensure NumPy array
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect  # Already a NumPy array, no conversion needed

def find_dest(pts: np.ndarray) -> np.ndarray:
    """
    Computes destination points based on max width and height.
    """
    (tl, tr, br, bl) = pts  # Unpack ordered points

    # Compute max width and height
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))) # type - None
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))) # type - None

    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

def ecg_processing_pipeline(
            input_image, 
            contour_data,  
            BUFFER=10):
    
    # convert og_image to grayscale 
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    og_image = cv2.merge([gray, gray, gray])
    imH, imW, _ = og_image.shape
    
    page = contour_data["contour"].astype(np.int32)
    half = contour_data["half"]
    scale_x = contour_data["scale_x"]
    scale_y = contour_data["scale_y"]
    
    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)
    corners = np.concatenate(corners).astype(np.float32)

    corners[:, 0] -= half
    corners[:, 1] -= half
    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y
    
    if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):

        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

        box_corners = cv2.boxPoints(cv2.minAreaRect(corners.reshape((-1, 1, 2)))).astype(np.int32)

        box_x_min, box_y_min = np.min(box_corners, axis=0)
        box_x_max, box_y_max = np.max(box_corners, axis=0)

        # Find corner point which doesn't satify the image constraint
        # and record the amount of shift required to make the box
        # corner satisfy the constraint

        left_pad, right_pad = max(-box_x_min, 0) + BUFFER, max(box_x_max - imW, 0) + BUFFER
        top_pad, bottom_pad = max(-box_y_min, 0) + BUFFER, max(box_y_max - imH, 0) + BUFFER

        # # new image with additional zeros pixels
        # # adjust original image within the new 'image_extended'

        image_extended = np.pad(og_image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')

        # shifting 'box_corners' the required amount
        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        og_image = image_extended

    corners = sorted(corners.tolist())
    return generate_output(og_image, corners)
