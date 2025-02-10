import logging
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from .base_database_manager import BaseDatabaseManager
from imageSearch.utils.logger_util import configure_logger

logger = configure_logger("imageSearch")


class MilvusDatabaseManager(BaseDatabaseManager):
    def __init__(self, uri, collection_name, index_params, dim, max_length=256, recreate=True):
        self.uri = uri
        self.collection_name = collection_name
        self.index_params = index_params
        self.dim = dim
        self.max_length = max_length
        self.connection_alias = "default"
        self.collection = None
        self.connect()
        self.create_or_get_collection(recreate)

    def connect(self):
        connections.connect(self.connection_alias, uri=self.uri)
        logger.info(f"Milvusに接続しました。URI: {self.uri}")

    def create_or_get_collection(self, recreate: bool):
        if utility.has_collection(self.collection_name):
            if recreate:
                utility.drop_collection(self.collection_name)
                logger.info(f"既存のコレクション '{self.collection_name}' を削除しました。")
                self.create_collection()
            else:
                self.collection = Collection(self.collection_name)
                logger.info(f"既存のコレクション '{self.collection_name}' を利用します。")
        else:
            self.create_collection()

    def create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=self.max_length)
        ]
        schema = CollectionSchema(fields, description="画像埋め込みコレクション (Image Embedding Collection)")
        self.collection = Collection(name=self.collection_name, schema=schema)
        logger.info(f"コレクション '{self.collection_name}' を作成しました。")
        self.create_index()

    def create_index(self):
        self.collection.create_index(field_name="embedding", index_params=self.index_params)
        logger.info("インデックスを作成しました。")

    def insert_embeddings(self, data):
        self.collection.insert(data)
        self.collection.flush()
        logger.info("データを挿入しました。")

    def search(self, query_vector, k, search_params=None):
        if search_params is None:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        return self.collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["id", "file_path"]
        )

    def save(self, index_file=None):
        raise NotImplementedError("MilvusDatabaseManagerはインデックスのファイル保存をサポートしていません。")
