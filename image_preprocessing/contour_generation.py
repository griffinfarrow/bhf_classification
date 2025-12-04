import torch 
import numpy as np 
import cv2 
import torchvision.transforms as T 
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torch import nn 
import warnings 
import h5py
from tqdm import tqdm
import os

def get_model(model_path, device=None):
    checkpoints = torch.load(model_path, map_location=device, weights_only=True)
    model = deeplabv3_mobilenet_v3_large(num_classes=2, aux_loss=True).to(device)
    model.load_state_dict(checkpoints, strict=False)
    return model 

def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    common_transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std),])
    return common_transforms

def compute_segmentation_mask(model, img_tens):
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
    img_tens = torch.unsqueeze(img_tens, dim=0) # type: ignore 
    
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
    
def save_contours_to_hdf5(contour_dict, filepath='contours.h5'):
    """
    Save all contours to a single HDF5 file
    """
    with h5py.File(filepath, 'w') as f:
        for img_id, data in tqdm(contour_dict.items(), desc="Saving"):
            if data is None:  # Skip failed segmentations
                continue
            
            # Create group for this image
            grp = f.create_group(img_id)
            
            # Save contour array
            grp.create_dataset('contour', data=data['contour'], 
                                compression='gzip', dtype='int16')
            
            # Save metadata as attributes
            grp.attrs['scale_x'] = data['scale_x']
            grp.attrs['scale_y'] = data['scale_y'] 
            grp.attrs['half'] = data['half']    

def precompute_contours_pipeline(model_path, img_paths, device, output_path='contours.h5'): 
    model = get_model(model_path, device=device)
    model.eval()
    
    contour_set = {}
    failed_images = []
    
    for path in tqdm(img_paths, desc="Processing images"):
        try:
            # extract image id 
            img_id = os.path.splitext(os.path.basename(path))[0]
            
            img = cv2.imread(path)
            if img is None:
                print(f"Failed to load: {path}")
                failed_images.append(path)
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            contour_item = output_contours_for_img(
                model=model, 
                img=img, 
                img_path=path, 
                device=device
            )
            
            contour_set[img_id] = contour_item  # Fixed: use img_id not "img_id"
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            failed_images.append(path)
    
    save_contours_to_hdf5(contour_set, filepath=output_path)
    
    print(f"Completed. Processed: {len(contour_set)}, Failed: {len(failed_images)}")
    if failed_images:
        print(f"Failed images: {failed_images[:10]}")  # Show first 10
    
    return contour_set, failed_images