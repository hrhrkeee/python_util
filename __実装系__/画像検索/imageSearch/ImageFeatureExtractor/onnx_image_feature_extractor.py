import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

from .base_feature_extractor import BaseFeatureExtractor

from imageSearch.utils.logger_util import configure_logger
logger = configure_logger("imageSearch")



class ONNXImageFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, onnx_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        provider = "CUDAExecutionProvider" if device.lower() == "cuda" else "CPUExecutionProvider"
        self.session = ort.InferenceSession(onnx_path, providers=[provider])
        inputs = self.session.get_inputs()
        if len(inputs) == 0:
            logger.error("ONNXモデルに入力が見つかりません。")
            raise ValueError("ONNX model has no inputs.")
        if len(inputs) > 1:
            logger.warning("ONNXモデルの入力が複数あります。最初の入力を使用します。")
        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape
        self.input_type = inputs[0].type
        logger.info(f"ONNX model input: name={self.input_name}, shape={self.input_shape}, type={self.input_type}")
        default_size = 224
        target_height = self.input_shape[2] if isinstance(self.input_shape[2], int) and self.input_shape[2] > 0 else default_size
        target_width  = self.input_shape[3] if isinstance(self.input_shape[3], int) and self.input_shape[3] > 0 else default_size
        self.preprocess = transforms.Compose([
            transforms.Resize(max(target_height, target_width)),
            transforms.CenterCrop((target_height, target_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.dim = self._compute_output_dim()
        logger.info(f"モデルの出力次元: {self.dim}")

    def _compute_output_dim(self):
        output_shape = self.session.get_outputs()[0].shape
        try:
            dims = [int(d) for d in output_shape[1:]]
            dim = np.prod(dims)
            return int(dim)
        except Exception as e:
            logger.warning(f"出力形状に動的次元が含まれています: {output_shape}. ダミー入力で出力次元を計算します。")
            dummy_input_shape = [1 if not isinstance(x, int) else x for x in self.input_shape]
            dummy_input = np.random.rand(*dummy_input_shape).astype(np.float32)
            output = self.session.run(None, {self.input_name: dummy_input})[0]
            dim = np.prod(output.shape[1:])
            return int(dim)

    def extract_feature(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"画像の読み込みに失敗しました。path: {image_path} error: {e}")
            raise
        img_tensor = self.preprocess(img).unsqueeze(0).numpy()
        output = self.session.run(None, {self.input_name: img_tensor})
        feature = output[0].flatten()
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature