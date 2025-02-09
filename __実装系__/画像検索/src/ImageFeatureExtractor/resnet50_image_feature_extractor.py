# src/ImageFeatureExtractor/image_feature_extractor.py
import logging
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from .base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Resnet50ImageFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()  # 全結合層を除去して2048次元出力に
        self.model.eval()
        self.model.to(self.device)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.dim = 2048
        logger.info(f"ImageFeatureExtractor を初期化しました。device: {self.device}")

    def extract_feature(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"画像の読み込みに失敗しました。path: {image_path} error: {e}")
            raise
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(img_tensor)
        feature_np = feature.cpu().numpy().flatten()
        norm = np.linalg.norm(feature_np)
        if norm > 0:
            feature_np = feature_np / norm
        return feature_np
