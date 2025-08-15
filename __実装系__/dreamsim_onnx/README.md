# DreamSim ONNX 変換ツール

DreamSimモデルをONNX形式に変換し、高速な推論を可能にするツールです。

## 概要

[DreamSim](https://github.com/dreamgaussian/dreamsim)は画像間の知覚的な類似度を測定するための優れたモデルですが、推論速度を向上させるため、このリポジトリではDreamSimモデルをONNX形式に変換します。ONNX形式は様々な環境での効率的な実行を可能にします。

## 機能

- DreamSimモデルをONNX形式に変換
- 主要関数の変換:
  - 画像間の距離計算
  - 特徴埋め込み抽出
- 複数のモデルタイプのサポート (ensemble, dino_vitb16, dinov2_vitb14など)
- 元のDreamSim APIと互換性のあるインターフェース

## 必要条件

- Python 3.8+
- PyTorch 1.9+
- ONNX 1.9+
- ONNXRuntime 1.9+
- DreamSim
- PIL (Pillow)
- NumPy
- tqdm

## インストール

```bash
# 必要なパッケージのインストール
pip install torch torchvision onnx onnxruntime dreamsim pillow numpy tqdm
```

## 使用方法

### モデルの変換

```bash
# 基本的な使用方法
python convert_dreamsim_to_onnx.py --model-type dino_vitb16 --cache-dir ./models --onnx-dir ./onnx_models

# 特定のモデルタイプを指定
python convert_dreamsim_to_onnx.py --model-type ensemble

# パッチモデルを使用（dino_vitb16とdinov2_vitb14のみ対応）
python convert_dreamsim_to_onnx.py --model-type dino_vitb16 --use-patch
```

### ONNX モデルの利用

```python
from dreamsim_onnx_utils import DreamSimONNX
from PIL import Image

# モデルの初期化
model_path = "./onnx_models/dino_vitb16.onnx"
dreamsim_onnx = DreamSimONNX(model_path)

# 2つの画像間の距離を計算
img1_path = "./sample_images/image1.jpg"
img2_path = "./sample_images/image2.jpg"
distance = dreamsim_onnx.get_distance(img1_path, img2_path)
print(f"画像間の距離: {distance}")

# 画像の特徴抽出
img = Image.open(img1_path).convert('RGB')
embedding = dreamsim_onnx.embed(img)
print(f"特徴埋め込みの形状: {embedding.shape}")
```

### データセットからの特徴抽出

元のDreamSim APIとの互換性のある特徴抽出関数を使用できます:

```python
from dreamsim_onnx_utils import get_DreamSim_features_onnx

# データセットの定義（ImageDatasetインターフェースを持つクラス）
class MyDataset:
    def __init__(self, image_paths):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def get_numpy_img(self, idx):
        # 画像をnumpy配列として読み込む
        return np.array(Image.open(self.image_paths[idx]).convert('RGB'))

# データセットを作成
dataset = MyDataset(["image1.jpg", "image2.jpg", "image3.jpg"])

# 特徴抽出
model_path = "./onnx_models/dino_vitb16.onnx"
features = get_DreamSim_features_onnx(dataset, model_path)
```

## サポートされているモデルタイプ

- ensemble (デフォルト、最高性能)
- dino_vitb16
- dinov2_vitb14
- open_clip_vitb32
- clip_vitb32
- synclr_vitb16
- dino_vitb16_patch (パッチモデル)
- dinov2_vitb14_patch (パッチモデル)

## 元のDreamSimとの比較

元のDreamSim APIのコード:

```python
from dreamsim import dreamsim
import torch
from PIL import Image
from tqdm import tqdm

def get_DreamSim_features(dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_model, ds_preprocess = dreamsim(pretrained=True, cache_dir="../__model__/00_misc/DreamSim/", device=device)
    
    features = []
    for idx in tqdm(range(len(dataset))):
        # torch.tensor to PIL.Image
        img = Image.fromarray(dataset.get_numpy_img(idx))
        img = ds_preprocess(img).to(device)
        features.append(ds_model.embed(img).detach().cpu().squeeze().numpy())
        
    return torch.tensor(features)
```

ONNX版のコード:

```python
from dreamsim_onnx_utils import get_DreamSim_features_onnx

def get_DreamSim_features_onnx_version(dataset):
    model_path = "./onnx_models/dino_vitb16.onnx"
    return get_DreamSim_features_onnx(dataset, model_path)
```

## ライセンス

このプロジェクトは元のDreamSimモデルのライセンスに従います。