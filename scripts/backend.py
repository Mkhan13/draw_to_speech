import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class Preprocessor:
    def __init__(self):
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def process(self, img):
        if img.ndim == 3: # Invert colors
            img = 255 - np.min(img, axis=2) 
        else:
            img = 255 - img
        img = img.astype('float32')
        
        # Adjust thresholds to 0-255 scale
        if np.max(img) < 12.0: # Check for empty canvas
            return torch.zeros((1, 64, 64), dtype=torch.float32)

        y, x = np.where(img > 40.0)  #Bounding box
        
        if len(x) == 0 or len(y) == 0:
             return torch.zeros((1, 64, 64), dtype=torch.float32)

        x1, x2 = x.min(), x.max()
        y1, y2 = y.min(), y.max()
        
        # padding
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.shape[1], x2 + pad)
        y2 = min(img.shape[0], y2 + pad)
        
        crop = img[y1:y2, x1:x2] 

        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA) # Resize
        out = self.tf(crop)
        return out

class DoodleModel:
    '''
    Loads EfficientNet, applies correct preprocessing, and predicts doodle class
    '''
    def __init__(self, weights_path, class_names):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(1280, len(class_names))

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.class_names = class_names
        self.prep = Preprocessor()

    def predict(self, img):
        img_tensor = self.prep.process(img).unsqueeze(0).to(self.device) # Add unsqueeze to create batch dimension

        with torch.no_grad():
            logits = self.model(img_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()

        return self.class_names[pred_idx]