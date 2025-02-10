import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from dreamsim import dreamsim

from .base_feature_extractor import BaseFeatureExtractor

from imageSearch.utils.logger_util import configure_logger
logger = configure_logger("imageSearch")



class DreamSimImageFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, cache_dir="../../model/DreamSim/", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.model, self.preprocess = dreamsim(pretrained=True, device=self.device, cache_dir=cache_dir)
        self.dim = 1792
        
    def extract_feature(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"画像の読み込みに失敗しました。path: {image_path} error: {e}")
            raise
        
        feature = self.model.embed(self.preprocess(img).to(self.device)).flatten()
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature

# --- 画像データの登録 ---
def register_images(image_dir_path, faiss_manager, extractor):
    """
    指定ディレクトリ内（再帰的探索）にある画像ファイル（.jpg, .jpeg, .png）を対象に、
    特徴抽出を行い、FAISSインデックスに登録します。

    :param image_dir_path: 画像ファイルが格納されているディレクトリのパス
    :param faiss_manager: FaissManager のインスタンス
    :param extractor: BaseFeatureExtractor を継承した画像特徴抽出クラスのインスタンス
    """
    image_dir = Path(image_dir_path)
    image_file_paths = [str(f) for f in image_dir.glob("**/*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    logger.info(f"{len(image_file_paths)} 件の画像ファイルを検出しました。")
    
    embeddings = []
    file_paths = []
    for image_file_path in tqdm(image_file_paths):
        try:
            feat = extractor.extract_feature(image_file_path)
            embeddings.append(feat)
            file_paths.append(image_file_path)
        except Exception as e:
            logger.error(f"特徴抽出エラー: {image_file_path} - {e}")
    if embeddings:
        faiss_manager.add_embeddings(np.vstack(embeddings), file_paths)
    logger.info("画像登録が完了しました。")
