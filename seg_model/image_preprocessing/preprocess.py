import numpy as np
import cv2
from typing import Optional

def generate_output(image: np.ndarray, corners: np.ndarray, scale: Optional[tuple] = None): 
    """Generates a perspective-transformed output image."""
    corners = order_points(corners)  # Ensure correct point order

    if scale is not None:
        corners *= np.array(scale, dtype=np.float32)  # Vectorized multiplication

    destination_corners = find_dest(corners)  # Get transformation target points
    M = cv2.getPerspectiveTransform(corners, destination_corners)  # Perspective matrix

    max_width, max_height = map(int, destination_corners[2])  # Extract output size
    out = cv2.warpPerspective(image, M, (max_width, max_height), flags=cv2.INTER_LANCZOS4)

    return np.clip(out, 0, 255).astype(np.uint8)  # Ensure valid pixel values

def order_points(pts: np.ndarray) -> np.ndarray:
    """Rearranges points to: top-left, top-right, bottom-right, bottom-left."""
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
    """Computes destination points based on max width and height."""
    (tl, tr, br, bl) = pts  # Unpack ordered points

    # Compute max width and height
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))) # type - None
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))) # type - None

    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

def deep_learning_scan(
            og_image, 
            contour_data,  
            BUFFER=10):
    
    imH, imW, C = og_image.shape
    
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
    return generate_output(og_image, corners), page
