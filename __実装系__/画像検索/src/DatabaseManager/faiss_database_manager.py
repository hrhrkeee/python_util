import os
import numpy as np
import faiss
import logging
from .base_database_manager import BaseDatabaseManager

# 複数の OpenMP ランタイムが初期化される場合のワークアラウンド
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class FAISSDatabaseManager(BaseDatabaseManager):
    def __init__(self, dim, index_file=None, recreate=True):
        self.dim = dim
        self.index_file = index_file
        self.file_paths = []  # 各ベクトルに対応する画像ファイルパスのリスト
        self.index = None
        self.connect()
        self.create_or_get_collection(recreate)

    def connect(self):
        logger.info("FAISSでは接続処理は不要です。")

    def create_or_get_collection(self, recreate: bool):
        if self.index_file is not None and os.path.exists(self.index_file) and not recreate:
            self.index = faiss.read_index(self.index_file)
            logger.info("FAISSインデックスを読み込みました。ファイル: %s", self.index_file)
        else:
            self.create_collection()

    def create_collection(self):
        self.index = faiss.IndexFlatIP(self.dim)
        logger.info("新しいFAISSインデックスを作成しました。")

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

    def save(self, index_file=None):
        target_file = index_file if index_file is not None else self.index_file
        if target_file is None:
            raise ValueError("保存するためのファイルパスが指定されていません。")
        faiss.write_index(self.index, target_file)
        logger.info("FAISSインデックスをファイルに保存しました: %s", target_file)
