"""
DreamSim ONNXモデルを利用するためのユーティリティ
"""

import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from torchvision import transforms
from typing import List, Union, Optional, Dict, Any, Tuple


class DreamSimONNX:
    """DreamSim ONNXモデルを扱うためのクラス"""
    
    def __init__(self, model_path: str, embed_model_path: Optional[str] = None, 
                preprocess_info_path: Optional[str] = None):
        """
        DreamSim ONNXモデルの初期化
        
        Args:
            model_path: DreamSimのONNXモデルパス（距離計算用）
            embed_model_path: 特徴抽出用のONNXモデルパス（指定がなければmodel_pathから推論）
            preprocess_info_path: 前処理設定のJSONファイルパス（指定がなければmodel_pathから推論）
        """
        self.model_path = model_path
        
        # embed_model_pathが指定されていなければ推測
        if embed_model_path is None:
            base_name = os.path.splitext(model_path)[0]
            embed_model_path = f"{base_name}_embed.onnx"
        self.embed_model_path = embed_model_path
        
        # preprocess_info_pathが指定されていなければ推測
        if preprocess_info_path is None:
            base_dir = os.path.dirname(model_path)
            model_type = os.path.basename(model_path).split('.')[0]
            preprocess_info_path = os.path.join(base_dir, f"{model_type}_preprocess_info.json")
        
        # JSONファイルから前処理の情報を読み込む
        if os.path.exists(preprocess_info_path):
            with open(preprocess_info_path, 'r') as f:
                self.preprocess_info = json.load(f)
        else:
            # デフォルト設定
            self.preprocess_info = {
                "img_size": 224,
                "interpolation": "bicubic",
                "model_type": "unknown"
            }
        
        # ONNX Runtimeセッションの初期化
        self.distance_session = ort.InferenceSession(model_path)
        self.embed_session = ort.InferenceSession(embed_model_path)
        
        # 前処理の設定
        self.preprocess = self._create_preprocess()
    
    def _create_preprocess(self) -> transforms.Compose:
        """前処理用のtransformsを作成"""
        img_size = self.preprocess_info.get("img_size", 224)
        interpolation_str = self.preprocess_info.get("interpolation", "bicubic")
        
        # 補間方法の文字列をtransforms.InterpolationModeに変換
        interpolation_map = {
            "bicubic": transforms.InterpolationMode.BICUBIC,
            "bilinear": transforms.InterpolationMode.BILINEAR,
            "nearest": transforms.InterpolationMode.NEAREST
        }
        interpolation = interpolation_map.get(interpolation_str, transforms.InterpolationMode.BICUBIC)
        
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=interpolation),
            transforms.ToTensor()
        ])
    
    def preprocess_image(self, img: Union[str, Image.Image]) -> np.ndarray:
        """
        画像を前処理する
        
        Args:
            img: PIL.ImageまたはX画像ファイルパス
            
        Returns:
            前処理済みの画像（numpy配列）
        """
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif not isinstance(img, Image.Image):
            raise TypeError("画像はPIL.Imageまたはファイルパスである必要があります")
        
        # 前処理を適用して[0-1]のテンソルを作成し、バッチ次元を追加
        img_tensor = self.preprocess(img).unsqueeze(0)
        
        # numpy配列に変換
        return img_tensor.numpy()
    
    def get_distance(self, img_a: Union[str, Image.Image, np.ndarray], 
                    img_b: Union[str, Image.Image, np.ndarray]) -> float:
        """
        2つの画像間のDreamSim距離を計算
        
        Args:
            img_a: 1つ目の画像（ファイルパス、PIL画像、前処理済みのnumpy配列）
            img_b: 2つ目の画像（ファイルパス、PIL画像、前処理済みのnumpy配列）
            
        Returns:
            距離スコア（値が大きいほど画像間の差が大きい）
        """
        # 画像の前処理
        if not isinstance(img_a, np.ndarray):
            img_a = self.preprocess_image(img_a)
        if not isinstance(img_b, np.ndarray):
            img_b = self.preprocess_image(img_b)
            
        # 推論実行
        inputs = {
            "input_a": img_a,
            "input_b": img_b
        }
        outputs = self.distance_session.run(None, inputs)
        
        # スカラー値を返す
        return float(outputs[0][0])
    
    def embed(self, img: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        画像の特徴埋め込みを取得
        
        Args:
            img: 画像（ファイルパス、PIL画像、前処理済みのnumpy配列）
            
        Returns:
            特徴埋め込みベクトル（numpy配列）
        """
        # 画像の前処理
        if not isinstance(img, np.ndarray):
            img = self.preprocess_image(img)
            
        # 推論実行
        inputs = {"input": img}
        outputs = self.embed_session.run(None, inputs)
        
        # 埋め込みを返す
        return outputs[0]


def get_DreamSim_features_onnx(dataset, model_path: str):
    """
    データセットの各画像からDreamSim特徴を抽出（ONNXバージョン）
    
    Args:
        dataset: 画像データセット（get_numpy_img(idx)メソッドを持つ）
        model_path: ONNXモデルのパス
        
    Returns:
        特徴ベクトル（torch.tensor）
    """
    import tqdm
    
    # DreamSim ONNXモデルのロード
    dreamsim_onnx = DreamSimONNX(model_path)
    
    features = []
    for idx in tqdm.tqdm(range(len(dataset))):
        # numpy配列からPIL.Imageに変換
        img = Image.fromarray(dataset.get_numpy_img(idx))
        # 特徴抽出
        feature = dreamsim_onnx.embed(img).squeeze()
        features.append(feature)
    
    return torch.tensor(np.array(features))


# 使用例
def example_usage():
    """使用例"""
    # 1. ONNXモデルをロードする
    model_path = "./onnx_models/dino_vitb16.onnx"
    dreamsim_onnx = DreamSimONNX(model_path)
    
    # 2. 2つの画像間の距離を計算
    img1_path = "./sample_data/coco_sample_datasets/sample_coco_train2017/000000187976.jpg"
    # img2_path = "./sample_data/coco_sample_datasets/sample_coco_train2017/000000187976.jpg"
    img2_path = "./sample_data/coco_sample_datasets/sample_coco_train2017/000000187989.jpg"
    distance = dreamsim_onnx.get_distance(img1_path, img2_path)
    print(f"Distance between images: {distance}")
    
    # # 3. 画像の特徴抽出
    # img = Image.open(img1_path).convert('RGB')
    # embedding = dreamsim_onnx.embed(img)
    # print(f"Embedding shape: {embedding.shape}")
    
    # # 4. オリジナルのget_DreamSim_features関数と同等の処理（ONNXバージョン）
    # class DummyDataset:
    #     def __init__(self, image_paths):
    #         self.image_paths = image_paths
            
    #     def __len__(self):
    #         return len(self.image_paths)
            
    #     def get_numpy_img(self, idx):
    #         return np.array(Image.open(self.image_paths[idx]).convert('RGB'))
    
    # dataset = DummyDataset([img1_path, img2_path])
    # features = get_DreamSim_features_onnx(dataset, model_path)
    # print(f"Features tensor shape: {features.shape}")


if __name__ == "__main__":
    example_usage()