"""
DreamSim ONNX変換の設定ファイル
"""

# モデルの設定
DEFAULT_MODEL_TYPE = "dino_vitb16"  # デフォルトのモデルタイプ
DEFAULT_CACHE_DIR = "./models"  # モデルのキャッシュディレクトリ
DEFAULT_ONNX_DIR = "./onnx_models"  # ONNXモデルの保存ディレクトリ

# 画像の設定
IMG_SIZE = 224  # 入力画像のサイズ

# ONNXのエクスポート設定
ONNX_OPSET_VERSION = 12  # ONNXのOpSetバージョン

# モデルタイプのリスト
SUPPORTED_MODELS = [
    "ensemble",
    "dino_vitb16",
    "dinov2_vitb14",
    "open_clip_vitb32",
    "clip_vitb32",
    "synclr_vitb16",
    "dino_vitb16_patch",
    "dinov2_vitb14_patch"
]

# 各モデルタイプの設定
MODEL_CONFIGS = {
    "ensemble": {
        "description": "デフォルトのアンサンブルモデル (dino_vitb16, clip_vitb16, open_clip_vitb16)",
        "patch_supported": False
    },
    "dino_vitb16": {
        "description": "DINO ViT-B/16 単一モデル",
        "patch_supported": True
    },
    "clip_vitb32": {
        "description": "CLIP ViT-B/32 単一モデル",
        "patch_supported": False
    },
    "open_clip_vitb32": {
        "description": "OpenCLIP ViT-B/32 単一モデル",
        "patch_supported": False
    },
    "dinov2_vitb14": {
        "description": "DINOv2 ViT-B/14 単一モデル",
        "patch_supported": True
    },
    "synclr_vitb16": {
        "description": "SynCLR ViT-B/16 単一モデル",
        "patch_supported": False
    }
}

# 画像サイズ設定
IMAGE_SIZE = 224

# モデルの正規化パラメータ (モデルによって異なる可能性があります)
NORMALIZATION = {
    "dino": {
        "mean": [0.485, 0.456, 0.406],  # ImageNet default
        "std": [0.229, 0.224, 0.225]
    },
    "clip": {
        "mean": [0.48145466, 0.4578275, 0.40821073],  # CLIP default
        "std": [0.26862954, 0.26130258, 0.27577711]
    },
    "openai_clip": {
        "mean": [0.48145466, 0.4578275, 0.40821073],  # OpenAI CLIP default
        "std": [0.26862954, 0.26130258, 0.27577711]
    },
    "mae": {
        "mean": [0.485, 0.456, 0.406],  # ImageNet default
        "std": [0.229, 0.224, 0.225]
    },
    "default": {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    }
}