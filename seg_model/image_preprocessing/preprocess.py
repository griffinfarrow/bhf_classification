import torchvision.transforms as T
import numpy as np
import cv2
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
import torch
import gc
from torch import nn
import warnings

def get_model(model_path, device=None):
    checkpoints = torch.load(model_path, map_location=device, weights_only=True)
    model = deeplabv3_mobilenet_v3_large(num_classes=2, aux_loss=True).to(device)
    model.load_state_dict(checkpoints, strict=False)
    return model 

def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    common_transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std),])
    return common_transforms

def compute_segmentation_mask(model, img_tens):
    model.eval()
    with torch.inference_mode():
        out = model(img_tens)["out"]
        out_mask = torch.argmax(out, dim=1, keepdim=True).permute(0, 2, 3, 1)[0].cpu().numpy().squeeze().astype(np.int32)
    return out_mask
    
def calculate_contours(mask, half):
    mask_padded = np.pad(mask * 255, pad_width=((half, half), (half, half)), mode='constant')
    
    canny = cv2.Canny(mask_padded.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour 

def output_contours_for_img(model: nn.Module,
                            img: np.ndarray,
                            img_path: str, 
                            image_size: int = 384, 
                            preprocess_transforms=image_preprocess_transforms(), 
                            device: torch.device = torch.device("cpu")):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img should be a np.ndarray, got {type(img)}")
    
    # pad image for segmentation to work
    # need smaller image (since that is what seg model was trained on)
    half = image_size // 2
    imH, imW, C = img.shape
    resized_img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    scale_x = imW / image_size
    scale_y = imH / image_size
    
    # convert image to tensor and normalize
    img_tens = preprocess_transforms(resized_img)
    img_tens = torch.unsqueeze(img_tens, dim=0)
    
    model = model.to(device)
    img_tens = img_tens.to(device)
    
    mask = compute_segmentation_mask(model, img_tens)
    
    contour = calculate_contours(mask, half)
    
    if contour is None: 
        warnings.warn(f"No contour found for image {img_path}")
        return None
    
    return {"contour": contour.astype(np.int16),
            "half": half, 
            "scale_x": scale_x, 
            "scale_y": scale_y}
    
def precompute_contours_pipeline(model_path, img_paths, device): 
    model = get_model(model_path, device=device)
    contour_set = []
    for path in img_paths: 
        

# def generate_output(image: np.ndarray, corners: np.ndarray, scale: tuple = None):
#     """Generates a perspective-transformed output image."""
#     corners = order_points(corners)  # Ensure correct point order

#     if scale is not None:
#         corners *= np.array(scale, dtype=np.float32)  # Vectorized multiplication

#     destination_corners = find_dest(corners)  # Get transformation target points
#     M = cv2.getPerspectiveTransform(corners, destination_corners)  # Perspective matrix

#     max_width, max_height = map(int, destination_corners[2])  # Extract output size
#     out = cv2.warpPerspective(image, M, (max_width, max_height), flags=cv2.INTER_LANCZOS4)

#     return np.clip(out, 0, 255).astype(np.uint8)  # Ensure valid pixel values

# def order_points(pts: np.ndarray) -> np.ndarray:
#     """Rearranges points to: top-left, top-right, bottom-right, bottom-left."""
#     pts = np.asarray(pts, dtype=np.float32)  # Ensure NumPy array
#     s = pts.sum(axis=1)
#     diff = np.diff(pts, axis=1)

#     rect = np.zeros((4, 2), dtype=np.float32)
#     rect[0] = pts[np.argmin(s)]  # Top-left
#     rect[2] = pts[np.argmax(s)]  # Bottom-right
#     rect[1] = pts[np.argmin(diff)]  # Top-right
#     rect[3] = pts[np.argmax(diff)]  # Bottom-left

#     return rect  # Already a NumPy array, no conversion needed

# def find_dest(pts: np.ndarray) -> np.ndarray:
#     """Computes destination points based on max width and height."""
#     (tl, tr, br, bl) = pts  # Unpack ordered points

#     # Compute max width and height
#     width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
#     height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

#     return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

def deep_learning_scan(og_image, 
                       trained_model, 
                       image_size=384, 
                       BUFFER=10, 
                       preprocess_transforms=image_preprocess_transforms(), 
                       device='cpu'):

    half = image_size // 2
    imH, imW, C = og_image.shape
    image_model = cv2.resize(og_image, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    scale_x = imW / image_size
    scale_y = imH / image_size
    
    image_model = preprocess_transforms(image_model)
    image_model = torch.unsqueeze(image_model, dim=0)
    
    image_model = image_model.to('cpu')

    # # Device on CPU
    model_cpu = trained_model.to('cpu')
    model_cpu.eval()
    
    # Rest of your preprocessing remains the same, but use model_cpu
    with torch.no_grad():
        out = model_cpu(image_model)["out"]

    out = torch.argmax(out, dim=1, keepdim=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
    r_H, r_W = out.shape

    out = np.pad(out * 255, pad_width=((half, half), (half, half)), mode='constant')

    # Edge Detection.
    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours: 
        return None

    page = max(contours, key=cv2.contourArea)

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

# def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
#     common_transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std),])
#     return common_transforms

