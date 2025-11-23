import numpy as np
import cv2
from typing import Optional
from PIL import Image

def smart_pad_and_resize_ecg(image, target_size=(224, 224), resize_strategy=cv2.INTER_AREA):
    """
    Resize image to fit within target_size while maintaining aspect ratio,
    then pad with white to reach exact target_size.
    
    Args:
        image: Input image (numpy array or PIL Image) - grayscale or color
        target_size: Tuple of (height, width) for output size
        
    Returns:
        Resized and padded image of exactly target_size dimensions
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor to fit within target size (maintains aspect ratio)
    scale = min(target_h / h, target_w / w)
    
    # Calculate new dimensions
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize image maintaining aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=resize_strategy)
    
    # Create white canvas of target size (255 = white)
    if len(image.shape) == 3:
        # Color/3-channel image
        padded = np.full((target_h, target_w, image.shape[2]), 255, dtype=image.dtype)
    else:
        # Grayscale image
        padded = np.full((target_h, target_w), 255, dtype=image.dtype)
    
    # Calculate padding offsets to center the image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # Place resized image in center of white canvas
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return padded

def crop_to_bounding_box(image: np.ndarray, corners: np.ndarray):
    """
    Crop the axis-aligned bounding box defined by corners.
    corners MUST already be in the same coordinate frame as image.
    """

    # get bounding box
    x_min = int(np.floor(corners[:,0].min()))
    y_min = int(np.floor(corners[:,1].min()))
    x_max = int(np.ceil(corners[:,0].max()))
    y_max = int(np.ceil(corners[:,1].max()))

    h, w = image.shape[:2]

    # clip to image bounds
    x_min = max(0, min(x_min, w - 1))
    x_max = max(1, min(x_max, w))
    y_min = max(0, min(y_min, h - 1))
    y_max = max(1, min(y_max, h))

    cropped = image[y_min:y_max, x_min:x_max]
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

def generate_cropped_output(image: np.ndarray, corners: np.ndarray):
    """
    Crop the axis-aligned bounding box of the contour and apply enhancements.
    No perspective transform.
    """
    # corners: Nx2 array (float)

    # Compute axis-aligned bounding box
    x_min, y_min = corners.min(axis=0).astype(int)
    x_max, y_max = corners.max(axis=0).astype(int)

    # Clip to image bounds (safe crop)
    h, w = image.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w - 1, x_max)
    y_max = min(h - 1, y_max)

    # Crop
    cropped = image[y_min:y_max, x_min:x_max]

    # Enhance (just like your existing pipeline)
    cropped = enhance_contrast(cropped)
    cropped = unsharp_mask(cropped, kernel_size=5, sigma=1.0, amount=1.2)

    return cropped

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
    imH, imW = gray.shape
    
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

        image_extended = np.pad(gray, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')

        # shifting 'box_corners' the required amount
        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        gray = image_extended

    corners = sorted(corners.tolist())
    return generate_output(gray, corners)

def ecg_processing_pipeline_no_perspective_distortion(
        input_image, 
        contour_data,
        BUFFER=10,
        target_size=(224, 224)):

    # Convert to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    imH, imW = gray.shape

    # Extract and scale contour points
    page = contour_data["contour"].astype(np.int32)
    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)
    corners = np.concatenate(corners).astype(np.float32)

    # Undo half offset and apply scale factors
    corners[:, 0] = (corners[:, 0] - contour_data["half"]) * contour_data["scale_x"]
    corners[:, 1] = (corners[:, 1] - contour_data["half"]) * contour_data["scale_y"]

    # Check if padding is needed
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)

    pad_left   = max(-x_min, 0) + BUFFER
    pad_top    = max(-y_min, 0) + BUFFER
    pad_right  = max(x_max - imW, 0) + BUFFER
    pad_bottom = max(y_max - imH, 0) + BUFFER

    # If padding needed, pad both image and corners
    if any(v > 0 for v in [pad_left, pad_top, pad_right, pad_bottom]):
        gray = np.pad(gray,
                      ((int(pad_top), int(pad_bottom)),
                       (int(pad_left), int(pad_right))),
                      mode="constant",
                      constant_values=255)

        corners[:, 0] += pad_left
        corners[:, 1] += pad_top

    # --- NOW corners and gray are in same coordinate frame ---

    # Crop to bounding box (this now WORKS)
    cropped = crop_to_bounding_box(gray, corners)

    # Enhance
    cropped = enhance_contrast(cropped)
    cropped = unsharp_mask(cropped, kernel_size=5, sigma=1.0, amount=1.2)
    
    return cropped

