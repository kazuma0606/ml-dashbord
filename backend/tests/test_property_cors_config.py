"""
CORS設定の環境変数制御プロパティテスト

プロパティ20: CORS設定の環境変数制御
"""
import pytest
import os
from hypothesis import given, strategies as st, settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from unittest.mock import patch


def create_test_app_with_cors(cors_origins: str, cors_allow_credentials: bool, 
                                cors_allow_methods: str, cors_allow_headers: str) -> FastAPI:
    """
    指定されたCORS設定でFastAPIアプリケーションを作成
    
    Args:
        cors_origins: 許可するオリジン（カンマ区切り、または"*"）
        cors_allow_credentials: 認証情報を許可するか
        cors_allow_methods: 許可するHTTPメソッド（カンマ区切り、または"*"）
        cors_allow_headers: 許可するヘッダー（カンマ区切り、または"*"）
    
    Returns:
        FastAPI: CORS設定済みのFastAPIアプリケーション
    """
    app = FastAPI()
    
    # CORS設定を環境変数から読み込む形式をシミュレート
    origins = cors_origins.split(",") if cors_origins != "*" else ["*"]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=cors_allow_credentials,
        allow_methods=cors_allow_methods.split(",") if cors_allow_methods != "*" else ["*"],
        allow_headers=cors_allow_headers.split(",") if cors_allow_headers != "*" else ["*"],
    )
    
    return app


def get_cors_middleware(app: FastAPI) -> CORSMiddleware:
    """
    FastAPIアプリケーションからCORSミドルウェアを取得
    
    Args:
        app: FastAPIアプリケーション
    
    Returns:
        CORSMiddleware: CORSミドルウェアインスタンス、またはkwargsを含むMiddlewareオブジェクト
    """
    for middleware in app.user_middleware:
        if middleware.cls == CORSMiddleware:
            return middleware
    return None


# **Feature: ml-visualization-dashboard, Property 20: CORS設定の環境変数制御**
@given(
    cors_origins=st.sampled_from([
        "*",
        "http://localhost:3000",
        "http://localhost:3000,http://localhost:8501",
        "https://example.com",
        "https://example.com,https://api.example.com,https://app.example.com"
    ]),
    cors_allow_credentials=st.booleans(),
    cors_allow_methods=st.sampled_from([
        "*",
        "GET,POST",
        "GET,POST,PUT,DELETE",
        "GET,POST,PUT,DELETE,PATCH,OPTIONS"
    ]),
    cors_allow_headers=st.sampled_from([
        "*",
        "Content-Type",
        "Content-Type,Authorization",
        "Content-Type,Authorization,X-Requested-With"
    ])
)
@settings(max_examples=100)
def test_property_cors_env_var_control(cors_origins, cors_allow_credentials, 
                                        cors_allow_methods, cors_allow_headers):
    """
    プロパティ20: CORS設定の環境変数制御
    
    任意のCORS設定値に対して、環境変数で設定した値が
    FastAPIアプリケーションのCORSミドルウェアに正しく適用されること
    
    検証: 要件 8.7
    """
    # 環境変数をモック
    with patch.dict(os.environ, {
        'CORS_ORIGINS': cors_origins,
        'CORS_ALLOW_CREDENTIALS': str(cors_allow_credentials),
        'CORS_ALLOW_METHODS': cors_allow_methods,
        'CORS_ALLOW_HEADERS': cors_allow_headers
    }):
        # Settingsクラスを再インポートして環境変数を読み込む
        from importlib import reload
        from src import config
        reload(config)
        
        # 新しい設定でアプリケーションを作成
        app = create_test_app_with_cors(
            cors_origins=config.settings.cors_origins,
            cors_allow_credentials=config.settings.cors_allow_credentials,
            cors_allow_methods=config.settings.cors_allow_methods,
            cors_allow_headers=config.settings.cors_allow_headers
        )
        
        # CORSミドルウェアが追加されていることを確認
        cors_middleware = get_cors_middleware(app)
        assert cors_middleware is not None, (
            "CORSミドルウェアがアプリケーションに追加されていない"
        )
        
        # ミドルウェアのkwargsを取得（FastAPIのMiddlewareオブジェクトから）
        middleware_kwargs = cors_middleware.kwargs
        
        # allow_origins の確認
        expected_origins = cors_origins.split(",") if cors_origins != "*" else ["*"]
        actual_origins = middleware_kwargs.get("allow_origins", [])
        assert actual_origins == expected_origins, (
            f"allow_origins が環境変数の値と一致しない: "
            f"期待={expected_origins}, 実際={actual_origins}"
        )
        
        # allow_credentials の確認
        actual_credentials = middleware_kwargs.get("allow_credentials", False)
        assert actual_credentials == cors_allow_credentials, (
            f"allow_credentials が環境変数の値と一致しない: "
            f"期待={cors_allow_credentials}, 実際={actual_credentials}"
        )
        
        # allow_methods の確認
        expected_methods = cors_allow_methods.split(",") if cors_allow_methods != "*" else ["*"]
        actual_methods = middleware_kwargs.get("allow_methods", [])
        assert actual_methods == expected_methods, (
            f"allow_methods が環境変数の値と一致しない: "
            f"期待={expected_methods}, 実際={actual_methods}"
        )
        
        # allow_headers の確認
        expected_headers = cors_allow_headers.split(",") if cors_allow_headers != "*" else ["*"]
        actual_headers = middleware_kwargs.get("allow_headers", [])
        assert actual_headers == expected_headers, (
            f"allow_headers が環境変数の値と一致しない: "
            f"期待={expected_headers}, 実際={actual_headers}"
        )


def test_cors_config_from_actual_settings():
    """
    実際のsettingsモジュールからCORS設定が正しく読み込まれることを確認
    
    これは単体テストで、実際のアプリケーション設定が正しく動作することを確認する
    """
    from src.config import settings
    
    # 設定が存在することを確認
    assert hasattr(settings, 'cors_origins'), "settings に cors_origins が存在しない"
    assert hasattr(settings, 'cors_allow_credentials'), "settings に cors_allow_credentials が存在しない"
    assert hasattr(settings, 'cors_allow_methods'), "settings に cors_allow_methods が存在しない"
    assert hasattr(settings, 'cors_allow_headers'), "settings に cors_allow_headers が存在しない"
    
    # デフォルト値が設定されていることを確認
    assert settings.cors_origins is not None, "cors_origins がNone"
    assert isinstance(settings.cors_allow_credentials, bool), "cors_allow_credentials がbool型でない"
    assert settings.cors_allow_methods is not None, "cors_allow_methods がNone"
    assert settings.cors_allow_headers is not None, "cors_allow_headers がNone"


def test_cors_middleware_applied_to_main_app():
    """
    メインアプリケーションにCORSミドルウェアが適用されていることを確認
    
    これは統合テストで、実際のmain.pyのアプリケーションが正しく設定されていることを確認する
    """
    from src.main import app
    
    # CORSミドルウェアが追加されていることを確認
    cors_middleware = get_cors_middleware(app)
    assert cors_middleware is not None, (
        "メインアプリケーションにCORSミドルウェアが追加されていない"
    )
    
    # ミドルウェアのkwargsが設定されていることを確認
    middleware_kwargs = cors_middleware.kwargs
    assert "allow_origins" in middleware_kwargs, "allow_origins が設定されていない"
    assert "allow_credentials" in middleware_kwargs, "allow_credentials が設定されていない"
    assert "allow_methods" in middleware_kwargs, "allow_methods が設定されていない"
    assert "allow_headers" in middleware_kwargs, "allow_headers が設定されていない"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
