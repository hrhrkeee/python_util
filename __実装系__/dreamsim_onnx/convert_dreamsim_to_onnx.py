"""
DreamSimモデルをONNXに変換するスクリプト
"""

import os
import sys
import argparse
import torch
import onnx
import numpy as np
from PIL import Image
from torchvision import transforms
from dreamsim import dreamsim
from dreamsim.model import PerceptualModel

from config import DEFAULT_MODEL_TYPE, DEFAULT_CACHE_DIR, DEFAULT_ONNX_DIR, ONNX_OPSET_VERSION, SUPPORTED_MODELS


def create_dummy_input(batch_size=1, channels=3, height=224, width=224):
    """ダミーの入力テンソルを作成する"""
    return torch.randn(batch_size, channels, height, width)


def convert_model_to_onnx(model_type, cache_dir, onnx_dir, use_patch_model=False, opset_version=12):
    """DreamSimモデルをONNXに変換する関数"""
    print(f"モデル '{model_type}' をONNXに変換します...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # DreamSimモデルをロード
    model, preprocess_fn = dreamsim(
        pretrained=True,
        cache_dir=cache_dir,
        device=device,
        dreamsim_type=model_type,
        use_patch_model=use_patch_model
    )
    
    # モデルを評価モードに設定
    model.eval()
    
    # ダミー入力の作成
    dummy_input = create_dummy_input().to(device)
    
    # onnx_filenameの設定
    if use_patch_model and model_type in ["dino_vitb16", "dinov2_vitb14"]:
        onnx_filename = f"{model_type}_patch.onnx"
    else:
        onnx_filename = f"{model_type}.onnx"
    
    onnx_path = os.path.join(onnx_dir, onnx_filename)
    
    # モデルのembedメソッドをONNXにエクスポート
    try:
        torch.onnx.export(
            model,
            (dummy_input, dummy_input),  # モデルは2つの入力を取る
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_a", "input_b"],
            output_names=["distance"],
            dynamic_axes={
                "input_a": {0: "batch_size"},
                "input_b": {0: "batch_size"},
                "distance": {0: "batch_size"}
            }
        )
        
        print(f"モデルを {onnx_path} に保存しました")
        
        # embedメソッド用のONNXモデルも作成
        embed_onnx_path = os.path.join(onnx_dir, f"{model_type}_embed.onnx")
        
        # embed関数をラッピングするクラス
        class EmbedWrapper(torch.nn.Module):
            def __init__(self, model):
                super(EmbedWrapper, self).__init__()
                self.model = model
                
            def forward(self, x):
                return self.model.embed(x)
        
        embed_model = EmbedWrapper(model)
        
        torch.onnx.export(
            embed_model,
            dummy_input,
            embed_onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "embedding": {0: "batch_size"}
            }
        )
        
        print(f"埋め込みモデルを {embed_onnx_path} に保存しました")
        
        # 変換したONNXモデルを検証
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        embed_onnx_model = onnx.load(embed_onnx_path)
        onnx.checker.check_model(embed_onnx_model)
        
        print("ONNXモデルの検証が完了しました")
        return onnx_path, embed_onnx_path
        
    except Exception as e:
        print(f"ONNXへの変換中にエラーが発生しました: {e}")
        return None, None


def save_preprocess_info(model_type, cache_dir, onnx_dir):
    """前処理情報をJSONファイルに保存する"""
    import json
    from dreamsim.config import dreamsim_args
    
    # 前処理情報
    preprocess_info = {
        "img_size": dreamsim_args["img_size"],
        "interpolation": "bicubic",
        "model_type": model_type
    }
    
    # 保存先のパスを作成
    preprocess_path = os.path.join(onnx_dir, f"{model_type}_preprocess_info.json")
    
    # JSONファイルに保存
    with open(preprocess_path, "w") as f:
        json.dump(preprocess_info, f, indent=4)
    
    print(f"前処理情報を {preprocess_path} に保存しました")
    return preprocess_path


def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="DreamSimモデルをONNXに変換するスクリプト")
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=SUPPORTED_MODELS,
        help=f"変換するモデルの種類 (デフォルト: {DEFAULT_MODEL_TYPE})"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"モデルのキャッシュディレクトリ (デフォルト: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default=DEFAULT_ONNX_DIR,
        help=f"ONNXモデルの保存ディレクトリ (デフォルト: {DEFAULT_ONNX_DIR})"
    )
    parser.add_argument(
        "--use-patch",
        action="store_true",
        help="パッチモデルを使用する (dino_vitb16とdinov2_vitb14のみ対応)"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=ONNX_OPSET_VERSION,
        help=f"ONNXのOpSetバージョン (デフォルト: {ONNX_OPSET_VERSION})"
    )
    
    args = parser.parse_args()
    
    # パッチモデルの検証
    if args.use_patch and args.model_type not in ["dino_vitb16", "dinov2_vitb14"]:
        print(f"警告: パッチモデルは {args.model_type} では使用できません。dino_vitb16またはdinov2_vitb14のみ対応しています。")
        args.use_patch = False
    
    # ディレクトリの作成
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.onnx_dir, exist_ok=True)
    
    # ONNXへの変換
    onnx_path, embed_onnx_path = convert_model_to_onnx(
        model_type=args.model_type,
        cache_dir=args.cache_dir,
        onnx_dir=args.onnx_dir,
        use_patch_model=args.use_patch,
        opset_version=args.opset_version
    )
    
    # 前処理情報の保存
    if onnx_path:
        preprocess_path = save_preprocess_info(args.model_type, args.cache_dir, args.onnx_dir)
        print(f"変換完了: {onnx_path}, {embed_onnx_path}, {preprocess_path}")


if __name__ == "__main__":
    main()