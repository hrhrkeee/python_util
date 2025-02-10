import os
import faiss
import pickle
import logging
import datetime
import numpy as np
from pathlib import Path

from .base_database_manager import BaseDatabaseManager
from imageSearch.utils.logger_util import configure_logger

# 複数の OpenMP ランタイムが初期化される場合のワークアラウンド
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = configure_logger("imageSearch")

class FAISSDatabaseManager(BaseDatabaseManager):
    def __init__(self, dim=None, index_file=None, image_paths_file=None, recreate=False):
        self.dim = dim
        if index_file is not None and not Path(index_file).exists() and dim is None:
            raise ValueError("次元数が指定されていません。")
        
        self.index_file = Path(index_file) if index_file is not None else None
        self.image_paths_file = Path(image_paths_file) if image_paths_file is not None else None
        
        self.index = None
        self.file_paths = []  # 各ベクトルに対応する画像ファイルパスのリスト
        
        self.connect()
        self.create_or_get_collection(recreate)

    def connect(self):
        logger.info("FAISSでは接続処理は不要です。")

    def create_or_get_collection(self, recreate: bool):
        
        if self.index_file is None and self.image_paths_file is None:
            self.create_collection()
            return

        if self.index_file is None:
            self.index_file = self.image_paths_file.parent / f"{self.image_paths_file.stem}.index"
        elif self.image_paths_file is None:
            self.image_paths_file = self.index_file.parent / f"{self.index_file.stem}.pkl"
        
        assert self.index_file.suffix == ".index", "インデックスファイルの拡張子は '.index' である必要があります。"
        assert self.image_paths_file.suffix == ".pkl", "パスリストの拡張子は '.pkl' である必要があります。"
        
        if self.index_file.exists() and self.image_paths_file.exists() and not recreate:
            self.load_collection()
        else:
            self.create_collection()
            
        return 

    def create_collection(self):
        self.index = faiss.IndexFlatIP(self.dim)
        logger.info("新しいFAISSインデックスを作成しました。")
        
    def load_collection(self):
        self.index = faiss.read_index(str(self.index_file))
        logger.info("FAISSインデックスを読み込みました。ファイル: %s", self.index_file)
        self.dim = self.index.d
        logger.info("次元数: %d", self.dim)
        
        with open(self.image_paths_file, "rb") as f:
            self.file_paths = pickle.load(f)
        logger.info("画像ファイルパスを読み込みました。ファイル: %s", self.image_paths_file)

    def insert_embeddings(self, data):
        embeddings = []
        file_paths = []
        for item in data:
            embedding = item.get("embedding")
            file_path = item.get("file_path")
            if embedding is None or file_path is None:
                raise ValueError("各データは 'embedding' と 'file_path' キーを含む必要があります。")
            embeddings.append(embedding)
            file_paths.append(file_path)
        embeddings = np.ascontiguousarray(np.array(embeddings, dtype=np.float32))
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"埋め込みの次元が不正です。期待値: (N, {self.dim})")
        self.index.add(embeddings)
        self.file_paths.extend(file_paths)
        logger.info("%d 件の埋め込みをインデックスに追加しました。", len(file_paths))
        
    def search(self, query_vector, k, **kwargs):
        query_vector = np.array(query_vector, dtype=np.float32)
        query_vector = np.squeeze(query_vector)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[1] != self.dim:
            raise ValueError(f"次元不一致: インデックスは {self.dim} 次元ですが、クエリは {query_vector.shape[1]} 次元です。")
        distances, indices = self.index.search(query_vector, k)
        return distances, indices

    def save(self, save_dir, file_name=""):
        
        save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        
        if file_name == "" and self.index_file is not None:
            file_name = Path(Path(self.index_file).name)
        elif file_name == "" and self.index_file is None:
            file_name = Path(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.index")
        else:
            file_name = Path(file_name)
        
        if file_name.suffix != ".index":
            raise ValueError("ファイル名には拡張子 '.index' を含めてください。")
            
        index_file = save_dir / file_name
        image_paths_file = save_dir / f"{file_name.stem}.pkl"
        
        faiss.write_index(self.index, str(index_file))
        logger.info("FAISSインデックスをファイルに保存しました: %s", str(index_file))
        
        with open(str(image_paths_file), "wb") as f:
            pickle.dump(self.file_paths, f)
        logger.info("画像ファイルパスをファイルに保存しました: %s", str(image_paths_file))
