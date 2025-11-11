from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A 
import numpy as np 
import random 

class CornerCutout(ImageOnlyTransform):
    """
    Randomly cut out corners to simulate bent/missing edges of ECG printouts
    """
    
    def __init__(self, max_cut_size=0.15, num_corners=2, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.max_cut_size = max_cut_size  # As fraction of image dimension
        self.num_corners = num_corners     # How many corners to cut
        
    def apply(self, img, **params):
        h, w = img.shape[:2]
        
        # Randomly select which corners to cut
        corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        selected_corners = random.sample(corners, k=random.randint(1, self.num_corners))
        
        # Get background color (assume white for ECG printouts)
        fill_value = 255 if img.dtype == np.uint8 else 1.0
        
        for corner in selected_corners:
            # Random cut size
            cut_h = int(h * random.uniform(0.05, self.max_cut_size))
            cut_w = int(w * random.uniform(0.05, self.max_cut_size))
            
            # Create triangular mask for each corner
            if corner == 'top_left':
                for i in range(cut_h):
                    img[i, :int(cut_w * (cut_h - i) / cut_h)] = fill_value
            elif corner == 'top_right':
                for i in range(cut_h):
                    img[i, w - int(cut_w * (cut_h - i) / cut_h):] = fill_value
            elif corner == 'bottom_left':
                for i in range(cut_h):
                    img[h - cut_h + i, :int(cut_w * (i + 1) / cut_h)] = fill_value
            elif corner == 'bottom_right':
                for i in range(cut_h):
                    img[h - cut_h + i, w - int(cut_w * (i + 1) / cut_h):] = fill_value
        
        return img
    
class PaperFoldEffect(ImageOnlyTransform):
    def __init__(self, max_intensity=0.5, fold_width=20, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.max_intensity = max_intensity
        self.fold_width = fold_width

    def apply(self, img, **params):
        img = img.copy().astype(np.float32)
        h, w = img.shape[:2]
        fold_width = int(0.1 * w)
        fold_pos = random.uniform(0.2, 0.8)
        fold_pos_px = int(w * fold_pos)
        half_width = fold_width // 2

        left = max(0, fold_pos_px - half_width)
        right = min(w, fold_pos_px + half_width)

        # stronger darkening
        intensity = random.uniform(0.5, 0.8)

        img[:, left:right] *= intensity

        # Optional: add subtle bright highlight edges
        img[:, left-2:left] *= 1.1
        img[:, right:right+2] *= 1.1
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
class GradientShadow(ImageOnlyTransform):
    """
    Add gradient-based shadows to simulate uneven lighting
    """
    
    def __init__(self, intensity=0.3, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.intensity = intensity  # How dark the shadow gets (0-1)
        
    def apply(self, img, **params):
        h, w = img.shape[:2]
        
        # Random shadow direction
        direction = random.choice(['left', 'right', 'top', 'bottom', 'diagonal'])
        
        # Create gradient mask
        if direction == 'left':
            gradient = np.linspace(1 - self.intensity, 1, w)
            shadow_mask = np.tile(gradient, (h, 1))
        elif direction == 'right':
            gradient = np.linspace(1, 1 - self.intensity, w)
            shadow_mask = np.tile(gradient, (h, 1))
        elif direction == 'top':
            gradient = np.linspace(1 - self.intensity, 1, h)
            shadow_mask = np.tile(gradient.reshape(-1, 1), (1, w))
        elif direction == 'bottom':
            gradient = np.linspace(1, 1 - self.intensity, h)
            shadow_mask = np.tile(gradient.reshape(-1, 1), (1, w))
        elif direction == 'diagonal':
            x_grad = np.linspace(1 - self.intensity, 1, w)
            y_grad = np.linspace(1 - self.intensity, 1, h)
            shadow_mask = np.outer(y_grad, x_grad)
            shadow_mask = shadow_mask / shadow_mask.max()  # Normalize
        
        # Apply shadow
        if len(img.shape) == 3:
            shadow_mask = shadow_mask[:, :, np.newaxis]
        
        img_shadowed = (img * shadow_mask).astype(img.dtype)
        
        return img_shadow
    
class BottomBlur(ImageOnlyTransform):
    """
    Apply blur predominantly to bottom portion of image
    """
    
    def __init__(self, blur_limit=(3, 7), bottom_region=0.25, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.blur_limit = blur_limit  # Kernel size range (must be odd)
        self.bottom_region = bottom_region  # Fraction of image height to blur
        
    def apply(self, img, **params):
        h, w = img.shape[:2]
        
        # Random kernel size (must be odd)
        kernel_size = random.choice([3, 5])
        
        # Apply blur to entire image first
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Calculate transition zone
        bottom_start = int(h * (1 - self.bottom_region))
        
        # Create gradient mask for smooth transition
        # Top portion: original image (weight = 1)
        # Bottom portion: blurred image (weight increases towards bottom)
        mask = np.ones((h, 1))
        mask[bottom_start:] = np.linspace(0, 1, h - bottom_start).reshape(-1, 1)
        
        if len(img.shape) == 3:
            mask = mask[:, :, np.newaxis]
        
        # Blend original and blurred based on mask
        result = (img * mask + blurred * (1 - mask)).astype(img.dtype)
        
        return result