from pathlib import Path
from itertools import accumulate
from PIL import Image
import numpy as np
import logging
from tqdm import tqdm

from src.DatabaseManager.MilvusDatabaseManager import MilvusDatabaseManager
from src.ImageFeatureExtractor.image_feature_extractor import ImageFeatureExtractor


def register_images(image_dir, db_manager, extractor):
    image_dir = Path(image_dir)
    image_paths = [
        str(p) for p in image_dir.glob("**/*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    rows = []
    for img_path in tqdm(image_paths, desc="Registering images"):
        try:
            feature = extractor.extract_feature(img_path)
            rows.append({"embedding": feature.tolist(), "file_path": img_path})
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
    db_manager.insert_embeddings(rows)


def search_images(collection, extractor, query_image_path, target_height=150, limit=5, nprobe=10):
    # 特徴抽出
    query_feature = extractor.extract_feature(query_image_path)
    
    # 検索パラメータの設定
    search_params = {"metric_type": "COSINE", "params": {"nprobe": nprobe}}
    
    # Milvusで検索
    results = collection.search(
        data=[query_feature.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["id", "file_path"]
    )
    
    return results


def main():
    # Milvusの設定
    uri = "./milvus_demo.db"
    collection_name = "image_embeddings"
    index_params = {"index_type": "IVF_FLAT", "params": {"nlist": 128}}
    dim = 2048
    recreate = True

    # MilvusDatabaseManagerのインスタンス生成
    db_manager = MilvusDatabaseManager(uri, collection_name, index_params, dim, recreate)

    # ImageFeatureExtractorのインスタンス生成
    extractor = ImageFeatureExtractor()

    # 画像ディレクトリから画像登録
    image_directory = "path/to/images"  # 適切なパスに変更してください
    register_images(image_directory, db_manager, extractor)

    # クエリ画像を用いた画像検索
    query_image = "path/to/query.jpg"  # 適切なパスに変更してください
    results = search_images(db_manager.collection, extractor, query_image)

    # 検索結果の表示
    if results and results[0]:
        for hit in results[0]:
            print(f"ID: {hit.id}, Distance: {hit.distance}, File: {hit.entity.get('file_path')}")
    else:
        print("検索結果が見つかりませんでした。")


if __name__ == "__main__":
    main()
