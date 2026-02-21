"""
Redisキャッシュ管理モジュール

Redis接続、リトライロジック、キャッシュキー生成、TTL管理を提供
"""
import time
import logging
from typing import Optional, Any
import pickle
import redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from ..config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Redisキャッシュマネージャー
    
    Redis接続の管理、リトライロジック、キャッシュ操作を提供
    """
    
    def __init__(
        self,
        redis_url: str = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Args:
            redis_url: Redis接続URL（デフォルト: settings.get_redis_url()）
            max_retries: 最大リトライ回数
            retry_delay: 初期リトライ遅延（秒）、指数バックオフで増加
        """
        self.redis_url = redis_url or settings.get_redis_url()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Optional[redis.Redis] = None
    
    def _connect_with_retry(self) -> redis.Redis:
        """リトライロジック付きでRedisに接続
        
        指数バックオフ（1s, 2s, 4s）でリトライ
        
        Returns:
            redis.Redis: Redis接続クライアント
            
        Raises:
            RedisConnectionError: 最大リトライ回数後も接続失敗
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Redis URLから接続（redis://またはrediss://をサポート）
                client = redis.from_url(
                    self.redis_url,
                    decode_responses=False,  # バイナリデータを扱うためFalse
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # 接続テスト
                client.ping()
                logger.info(f"Redis接続成功: {self.redis_url.split('@')[0] if '@' in self.redis_url else self.redis_url.split('//')[1].split(':')[0]}")
                return client
            except (RedisError, RedisConnectionError) as e:
                last_error = e
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Redis接続失敗 (試行 {attempt + 1}/{self.max_retries}): {e}. "
                    f"{delay}秒後にリトライ..."
                )
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
        
        error_msg = f"Redis接続失敗: {self.max_retries}回の試行後も接続できませんでした"
        logger.error(f"{error_msg}: {last_error}")
        raise RedisConnectionError(error_msg)
    
    @property
    def client(self) -> redis.Redis:
        """Redisクライアントを取得（遅延初期化）
        
        Returns:
            redis.Redis: Redis接続クライアント
        """
        if self._client is None:
            self._client = self._connect_with_retry()
        return self._client
    
    @staticmethod
    def generate_cache_key(prefix: str, identifier: str) -> str:
        """キャッシュキーを生成
        
        Args:
            prefix: キープレフィックス（例: "dataset"）
            identifier: 識別子（例: データセット名）
            
        Returns:
            str: 生成されたキャッシュキー（例: "dataset:iris"）
        """
        return f"{prefix}:{identifier}"
    
    def get(self, key: str) -> Optional[Any]:
        """キャッシュから値を取得
        
        Args:
            key: キャッシュキー
            
        Returns:
            Optional[Any]: デシリアライズされた値、存在しない場合はNone
        """
        try:
            data = self.client.get(key)
            if data is None:
                return None
            return pickle.loads(data)
        except RedisError as e:
            logger.error(f"キャッシュ取得エラー (key={key}): {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """キャッシュに値を保存
        
        Args:
            key: キャッシュキー
            value: 保存する値（pickleでシリアライズ可能な任意のオブジェクト）
            ttl: Time To Live（秒）、デフォルト: 3600秒（1時間）
            
        Returns:
            bool: 保存成功時True、失敗時False
        """
        try:
            serialized = pickle.dumps(value)
            self.client.setex(key, ttl, serialized)
            logger.debug(f"キャッシュ保存成功 (key={key}, ttl={ttl})")
            return True
        except RedisError as e:
            logger.error(f"キャッシュ保存エラー (key={key}): {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """キャッシュキーが存在するか確認
        
        Args:
            key: キャッシュキー
            
        Returns:
            bool: 存在する場合True
        """
        try:
            return self.client.exists(key) > 0
        except RedisError as e:
            logger.error(f"キャッシュ存在確認エラー (key={key}): {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """キャッシュから値を削除
        
        Args:
            key: キャッシュキー
            
        Returns:
            bool: 削除成功時True
        """
        try:
            self.client.delete(key)
            return True
        except RedisError as e:
            logger.error(f"キャッシュ削除エラー (key={key}): {e}")
            return False
    
    def close(self):
        """Redis接続をクローズ"""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Redis接続をクローズしました")


# グローバルキャッシュマネージャーインスタンス
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """グローバルキャッシュマネージャーを取得
    
    Returns:
        CacheManager: キャッシュマネージャーインスタンス
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
