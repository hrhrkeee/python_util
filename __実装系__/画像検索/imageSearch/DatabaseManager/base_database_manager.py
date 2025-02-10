import abc

class BaseDatabaseManager(abc.ABC):
    @abc.abstractmethod
    def connect(self):
        """データベースへの接続処理"""
        pass

    @abc.abstractmethod
    def create_or_get_collection(self, recreate: bool):
        """コレクションの作成または取得"""
        pass

    @abc.abstractmethod
    def create_collection(self):
        """コレクションの作成"""
        pass

    @abc.abstractmethod
    def insert_embeddings(self, data):
        """埋め込みベクトルデータの挿入"""
        pass

    @abc.abstractmethod
    def search(self, query_vector, k, **kwargs):
        """検索を行うメソッド"""
        pass

    @abc.abstractmethod
    def save(self, index_file=None):
        """インデックスを保存するメソッド"""
        pass
