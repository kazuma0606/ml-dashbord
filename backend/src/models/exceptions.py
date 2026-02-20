"""
カスタム例外クラス定義
"""


class DatasetNotFoundError(Exception):
    """データセットが見つからない場合の例外"""
    pass


class ModelTrainingError(Exception):
    """モデル学習中のエラー"""
    pass


class DatabaseError(Exception):
    """データベース操作エラー"""
    pass
