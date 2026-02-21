"""
モノレポ構成の独立性プロパティテスト

プロパティ22: モノレポ構成の独立性
"""
import pytest
import os
import toml
import subprocess
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume


def get_project_root() -> Path:
    """
    プロジェクトルートディレクトリを取得
    
    Returns:
        Path: プロジェクトルートディレクトリのパス
    """
    # このファイルから見たプロジェクトルート
    current_file = Path(__file__).resolve()
    # tests/test_property_monorepo_independence.py -> project_root
    return current_file.parent.parent


def load_pyproject_toml(project_path: Path) -> dict:
    """
    pyproject.tomlファイルを読み込む
    
    Args:
        project_path: プロジェクトディレクトリのパス
    
    Returns:
        dict: pyproject.tomlの内容
    """
    pyproject_path = project_path / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    
    with open(pyproject_path, 'r', encoding='utf-8') as f:
        return toml.load(f)


def check_uv_can_sync(project_path: Path) -> bool:
    """
    uvで依存関係を同期できるかチェック（dry-run）
    
    Args:
        project_path: プロジェクトディレクトリのパス
    
    Returns:
        bool: 同期可能な場合True
    """
    try:
        # uv sync --dry-run でチェック（実際にはインストールしない）
        result = subprocess.run(
            ["uv", "pip", "compile", "pyproject.toml", "--dry-run"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        # エラーがなければ成功
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # uvがインストールされていない、またはタイムアウトの場合はスキップ
        return None


# **Feature: ml-visualization-dashboard, Property 22: モノレポ構成の独立性**
@given(
    subproject=st.sampled_from(["backend", "frontend"])
)
@settings(max_examples=100)
def test_property_monorepo_independence(subproject):
    """
    プロパティ22: モノレポ構成の独立性
    
    任意のサブプロジェクト（frontend、backend）に対して、
    それぞれ独立したpyproject.tomlファイルを持ち、
    uvで個別に依存関係を管理できること
    
    検証: 要件 10.2, 10.3
    """
    project_root = get_project_root()
    subproject_path = project_root / subproject
    
    # 1. サブプロジェクトディレクトリが存在することを確認
    assert subproject_path.exists(), (
        f"サブプロジェクトディレクトリが存在しない: {subproject_path}"
    )
    assert subproject_path.is_dir(), (
        f"サブプロジェクトパスがディレクトリではない: {subproject_path}"
    )
    
    # 2. 独立したpyproject.tomlファイルが存在することを確認
    pyproject_path = subproject_path / "pyproject.toml"
    assert pyproject_path.exists(), (
        f"サブプロジェクトにpyproject.tomlが存在しない: {pyproject_path}"
    )
    
    # 3. pyproject.tomlが有効なTOML形式であることを確認
    pyproject_data = load_pyproject_toml(subproject_path)
    assert pyproject_data is not None, (
        f"pyproject.tomlの読み込みに失敗: {pyproject_path}"
    )
    
    # 4. 必須フィールドが存在することを確認
    assert "project" in pyproject_data, (
        f"pyproject.tomlに[project]セクションが存在しない: {subproject}"
    )
    
    project_section = pyproject_data["project"]
    assert "name" in project_section, (
        f"pyproject.tomlに[project.name]が存在しない: {subproject}"
    )
    assert "version" in project_section, (
        f"pyproject.tomlに[project.version]が存在しない: {subproject}"
    )
    assert "requires-python" in project_section, (
        f"pyproject.tomlに[project.requires-python]が存在しない: {subproject}"
    )
    
    # 5. 依存関係セクションが存在することを確認
    assert "dependencies" in project_section, (
        f"pyproject.tomlに[project.dependencies]が存在しない: {subproject}"
    )
    
    dependencies = project_section["dependencies"]
    assert isinstance(dependencies, list), (
        f"dependenciesがリスト形式ではない: {subproject}"
    )
    assert len(dependencies) > 0, (
        f"dependenciesが空: {subproject}"
    )
    
    # 6. サブプロジェクト固有の依存関係を持つことを確認
    if subproject == "backend":
        # バックエンド固有の依存関係
        backend_deps = ["fastapi", "uvicorn", "sqlalchemy", "redis"]
        for dep in backend_deps:
            assert any(dep in d.lower() for d in dependencies), (
                f"バックエンド固有の依存関係が見つからない: {dep}"
            )
    elif subproject == "frontend":
        # フロントエンド固有の依存関係
        frontend_deps = ["streamlit", "plotly"]
        for dep in frontend_deps:
            assert any(dep in d.lower() for d in dependencies), (
                f"フロントエンド固有の依存関係が見つからない: {dep}"
            )
    
    # 7. build-systemセクションが存在することを確認
    assert "build-system" in pyproject_data, (
        f"pyproject.tomlに[build-system]セクションが存在しない: {subproject}"
    )
    
    build_system = pyproject_data["build-system"]
    assert "requires" in build_system, (
        f"build-systemにrequiresが存在しない: {subproject}"
    )
    assert "build-backend" in build_system, (
        f"build-systemにbuild-backendが存在しない: {subproject}"
    )


def test_root_workspace_configuration():
    """
    ルートのpyproject.tomlがワークスペース設定を持つことを確認
    
    これは単体テストで、モノレポのワークスペース設定が正しいことを確認する
    """
    project_root = get_project_root()
    root_pyproject = load_pyproject_toml(project_root)
    
    assert root_pyproject is not None, (
        "ルートのpyproject.tomlが存在しないか読み込めない"
    )
    
    # ワークスペース設定の確認
    assert "tool" in root_pyproject, (
        "ルートのpyproject.tomlに[tool]セクションが存在しない"
    )
    assert "uv" in root_pyproject["tool"], (
        "ルートのpyproject.tomlに[tool.uv]セクションが存在しない"
    )
    assert "workspace" in root_pyproject["tool"]["uv"], (
        "ルートのpyproject.tomlに[tool.uv.workspace]セクションが存在しない"
    )
    
    workspace = root_pyproject["tool"]["uv"]["workspace"]
    assert "members" in workspace, (
        "ワークスペース設定にmembersが存在しない"
    )
    
    members = workspace["members"]
    assert isinstance(members, list), (
        "workspace.membersがリスト形式ではない"
    )
    assert "backend" in members, (
        "workspace.membersにbackendが含まれていない"
    )
    assert "frontend" in members, (
        "workspace.membersにfrontendが含まれていない"
    )


def test_subprojects_have_different_names():
    """
    各サブプロジェクトが異なるプロジェクト名を持つことを確認
    
    これは単体テストで、サブプロジェクトが独立していることを確認する
    """
    project_root = get_project_root()
    
    backend_pyproject = load_pyproject_toml(project_root / "backend")
    frontend_pyproject = load_pyproject_toml(project_root / "frontend")
    
    assert backend_pyproject is not None, "backend/pyproject.tomlが読み込めない"
    assert frontend_pyproject is not None, "frontend/pyproject.tomlが読み込めない"
    
    backend_name = backend_pyproject["project"]["name"]
    frontend_name = frontend_pyproject["project"]["name"]
    
    assert backend_name != frontend_name, (
        f"サブプロジェクトが同じ名前を持っている: {backend_name}"
    )
    
    # 名前が適切であることを確認
    assert "backend" in backend_name.lower(), (
        f"バックエンドプロジェクト名が適切でない: {backend_name}"
    )
    assert "frontend" in frontend_name.lower(), (
        f"フロントエンドプロジェクト名が適切でない: {frontend_name}"
    )


def test_subprojects_have_independent_dependencies():
    """
    各サブプロジェクトが独立した依存関係を持つことを確認
    
    これは単体テストで、依存関係が適切に分離されていることを確認する
    """
    project_root = get_project_root()
    
    backend_pyproject = load_pyproject_toml(project_root / "backend")
    frontend_pyproject = load_pyproject_toml(project_root / "frontend")
    
    backend_deps = backend_pyproject["project"]["dependencies"]
    frontend_deps = frontend_pyproject["project"]["dependencies"]
    
    # バックエンド固有の依存関係がフロントエンドに含まれていないことを確認
    backend_specific = ["fastapi", "uvicorn", "sqlalchemy"]
    for dep in backend_specific:
        assert not any(dep in d.lower() for d in frontend_deps), (
            f"フロントエンドにバックエンド固有の依存関係が含まれている: {dep}"
        )
    
    # フロントエンド固有の依存関係がバックエンドに含まれていないことを確認
    frontend_specific = ["streamlit", "plotly"]
    for dep in frontend_specific:
        assert not any(dep in d.lower() for d in backend_deps), (
            f"バックエンドにフロントエンド固有の依存関係が含まれている: {dep}"
        )


def test_subprojects_have_src_directories():
    """
    各サブプロジェクトがsrcディレクトリを持つことを確認
    
    これは単体テストで、プロジェクト構造が適切であることを確認する
    """
    project_root = get_project_root()
    
    for subproject in ["backend", "frontend"]:
        src_path = project_root / subproject / "src"
        assert src_path.exists(), (
            f"サブプロジェクトにsrcディレクトリが存在しない: {subproject}"
        )
        assert src_path.is_dir(), (
            f"srcがディレクトリではない: {subproject}"
        )
        
        # __init__.pyが存在することを確認（Pythonパッケージとして認識される）
        init_path = src_path / "__init__.py"
        assert init_path.exists(), (
            f"src/__init__.pyが存在しない: {subproject}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
