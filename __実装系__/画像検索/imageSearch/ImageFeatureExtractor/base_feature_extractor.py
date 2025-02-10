import abc

# 画像特徴抽出のための基底クラス
class BaseFeatureExtractor(abc.ABC):
    @abc.abstractmethod
    def extract_feature(self, image_path):
        """
        画像ファイルから特徴を抽出する抽象メソッド
        :param image_path: 画像ファイルのパス（文字列）
        :return: 特徴ベクトル（NumPy 配列）
        """
        pass