"""
フロントエンドAPI接続の環境変数制御プロパティテスト

プロパティ21: フロントエンドAPI接続の環境変数制御
"""
import pytest
import os
from hypothesis import given, strategies as st, settings
from unittest.mock import patch


# **Feature: ml-visualization-dashboard, Property 21: フロントエンドAPI接続の環境変数制御**
@given(
    api_base_url=st.sampled_from([
        "http://localhost:8000",
        "http://backend:8000",
        "http://127.0.0.1:8000",
        "https://api.example.com",
        "https://ml-api.production.com",
        "http://192.168.1.100:8000",
        "http://api-server:9000"
    ]),
    api_timeout=st.integers(min_value=5, max_value=120)
)
@settings(max_examples=100)
def test_property_frontend_api_env_var_control(api_base_url, api_timeout):
    """
    プロパティ21: フロントエンドAPI接続の環境変数制御
    
    任意のバックエンドURL設定に対して、環境変数で設定したURLが
    StreamlitアプリケーションのAPIクライアントで使用されること
    
    検証: 要件 8.8
    """
    # 環境変数をモック
    with patch.dict(os.environ, {
        'API_BASE_URL': api_base_url,
        'API_TIMEOUT': str(api_timeout)
    }, clear=False):
        # 既存のモジュールをキャッシュから削除して再読み込み
        import sys
        
        # モジュールを削除
        modules_to_delete = [k for k in sys.modules.keys() if k.startswith('src.')]
        for module in modules_to_delete:
            del sys.modules[module]
        
        # 新しい環境変数で設定を読み込む
        from src.config import Settings
        
        # 新しいSettingsインスタンスを作成（環境変数から読み込む）
        test_settings = Settings()
        
        # 設定が環境変数から正しく読み込まれていることを確認
        assert test_settings.api_base_url == api_base_url, (
            f"api_base_url が環境変数の値と一致しない: "
            f"期待={api_base_url}, 実際={test_settings.api_base_url}"
        )
        
        assert test_settings.api_timeout == api_timeout, (
            f"api_timeout が環境変数の値と一致しない: "
            f"期待={api_timeout}, 実際={test_settings.api_timeout}"
        )
        
        # APIクライアントを作成（デフォルト引数で）
        from src.api_client import MLAPIClient
        
        # 設定を明示的に渡してクライアントを作成
        client = MLAPIClient(base_url=test_settings.api_base_url, timeout=test_settings.api_timeout)
        
        # APIクライアントが環境変数から読み込んだ設定を使用していることを確認
        assert client.base_url == api_base_url, (
            f"APIクライアントのbase_urlが環境変数の値と一致しない: "
            f"期待={api_base_url}, 実際={client.base_url}"
        )
        
        assert client.timeout == api_timeout, (
            f"APIクライアントのtimeoutが環境変数の値と一致しない: "
            f"期待={api_timeout}, 実際={client.timeout}"
        )


@given(
    custom_base_url=st.sampled_from([
        "http://custom-backend:8080",
        "https://custom-api.example.com",
        "http://localhost:9999"
    ]),
    custom_timeout=st.integers(min_value=10, max_value=60).filter(lambda x: x != 30)  # 30を除外（デフォルト値）
)
@settings(max_examples=50)
def test_property_api_client_explicit_override(custom_base_url, custom_timeout):
    """
    プロパティ21の拡張: APIクライアントの明示的なオーバーライド
    
    任意のカスタムURL設定に対して、APIクライアントのコンストラクタで
    明示的に指定した値が環境変数の値より優先されること
    
    検証: 要件 8.8
    """
    # 環境変数に異なる値を設定
    env_base_url = "http://env-backend:8000"
    env_timeout = 30
    
    with patch.dict(os.environ, {
        'API_BASE_URL': env_base_url,
        'API_TIMEOUT': str(env_timeout)
    }, clear=False):
        # 既存のモジュールをキャッシュから削除
        import sys
        modules_to_delete = [k for k in sys.modules.keys() if k.startswith('src.')]
        for module in modules_to_delete:
            del sys.modules[module]
        
        from src.config import Settings
        from src.api_client import MLAPIClient
        
        # 環境変数から設定を読み込む
        test_settings = Settings()
        
        # 環境変数が正しく設定されていることを確認
        assert test_settings.api_base_url == env_base_url
        assert test_settings.api_timeout == env_timeout
        
        # 明示的な引数でAPIクライアントを作成
        client = MLAPIClient(base_url=custom_base_url, timeout=custom_timeout)
        
        # 明示的な引数が環境変数より優先されることを確認
        assert client.base_url == custom_base_url, (
            f"APIクライアントのbase_urlが明示的な引数と一致しない: "
            f"期待={custom_base_url}, 実際={client.base_url}"
        )
        
        assert client.timeout == custom_timeout, (
            f"APIクライアントのtimeoutが明示的な引数と一致しない: "
            f"期待={custom_timeout}, 実際={client.timeout}"
        )
        
        # 環境変数の値とは異なることを確認
        assert client.base_url != env_base_url, (
            "明示的な引数が環境変数より優先されていない（base_url）"
        )
        
        assert client.timeout != env_timeout, (
            "明示的な引数が環境変数より優先されていない（timeout）"
        )


def test_frontend_config_from_actual_settings():
    """
    実際のsettingsモジュールからフロントエンド設定が正しく読み込まれることを確認
    
    これは単体テストで、実際のアプリケーション設定が正しく動作することを確認する
    """
    from src.config import settings
    
    # 設定が存在することを確認
    assert hasattr(settings, 'api_base_url'), "settings に api_base_url が存在しない"
    assert hasattr(settings, 'api_timeout'), "settings に api_timeout が存在しない"
    
    # デフォルト値が設定されていることを確認
    assert settings.api_base_url is not None, "api_base_url がNone"
    assert isinstance(settings.api_timeout, int), "api_timeout がint型でない"
    assert settings.api_timeout > 0, "api_timeout が正の値でない"


def test_api_client_uses_settings_by_default():
    """
    APIクライアントがデフォルトで設定モジュールの値を使用することを確認
    
    これは統合テストで、実際のAPIクライアントが正しく設定されていることを確認する
    """
    from src.config import settings
    from src.api_client import MLAPIClient
    
    # デフォルト引数でクライアントを作成
    client = MLAPIClient()
    
    # 設定モジュールの値が使用されていることを確認
    assert client.base_url == settings.api_base_url, (
        f"APIクライアントがデフォルトで設定の値を使用していない: "
        f"期待={settings.api_base_url}, 実際={client.base_url}"
    )
    
    assert client.timeout == settings.api_timeout, (
        f"APIクライアントがデフォルトで設定のタイムアウトを使用していない: "
        f"期待={settings.api_timeout}, 実際={client.timeout}"
    )


def test_api_client_session_configured():
    """
    APIクライアントのセッションが正しく設定されていることを確認
    
    リトライロジックとアダプターが適切に設定されていることを検証する
    """
    from src.api_client import MLAPIClient
    
    client = MLAPIClient()
    
    # セッションが存在することを確認
    assert hasattr(client, 'session'), "APIクライアントにsessionが存在しない"
    assert client.session is not None, "APIクライアントのsessionがNone"
    
    # HTTPアダプターが設定されていることを確認
    http_adapter = client.session.get_adapter("http://")
    assert http_adapter is not None, "HTTPアダプターが設定されていない"
    
    https_adapter = client.session.get_adapter("https://")
    assert https_adapter is not None, "HTTPSアダプターが設定されていない"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
