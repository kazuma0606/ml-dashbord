"""
データサービスの基本テスト

CacheManager, DatasetLoader, DataPreprocessorの基本機能をテスト
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings
from src.services import CacheManager, DatasetLoader, DataPreprocessor, Dataset


class TestCacheManager:
    """CacheManagerの基本テスト"""
    
    def test_generate_cache_key(self):
        """キャッシュキー生成のテスト"""
        key = CacheManager.generate_cache_key("dataset", "iris")
        assert key == "dataset:iris"
    
    def test_generate_cache_key_with_special_chars(self):
        """特殊文字を含むキャッシュキー生成のテスト"""
        key = CacheManager.generate_cache_key("model", "random_forest_v2")
        assert key == "model:random_forest_v2"


class TestDatasetLoader:
    """DatasetLoaderの基本テスト"""
    
    def test_get_available_datasets(self):
        """利用可能なデータセット一覧取得のテスト"""
        # Redisモックを使用
        mock_cache = Mock(spec=CacheManager)
        loader = DatasetLoader(cache_manager=mock_cache)
        
        datasets = loader.get_available_datasets()
        
        assert len(datasets) > 0
        assert all("name" in ds and "description" in ds for ds in datasets)
        
        # 期待されるデータセットが含まれているか確認
        dataset_names = [ds["name"] for ds in datasets]
        assert "iris" in dataset_names
        assert "wine" in dataset_names
        assert "breast_cancer" in dataset_names
        assert "digits" in dataset_names
    
    # **Feature: ml-visualization-dashboard, Property 1: データセット読み込みの完全性**
    @given(
        dataset_name=st.sampled_from(["iris", "wine", "breast_cancer", "digits"])
    )
    @settings(max_examples=100)
    def test_property_dataset_loading_completeness(self, dataset_name):
        """
        プロパティ1: データセット読み込みの完全性
        
        任意の有効なデータセット名に対して、データセットを読み込んだ後、
        メタデータ（データセット名、サンプル数、特徴量数）がすべて存在し、
        データとターゲットが空でないこと
        
        検証: 要件 1.1, 1.5
        """
        # Redisモックを使用（キャッシュなしで毎回読み込み）
        mock_cache = Mock(spec=CacheManager)
        mock_cache.generate_cache_key.return_value = f"dataset:{dataset_name}"
        mock_cache.get.return_value = None  # キャッシュなし
        mock_cache.set.return_value = True
        
        loader = DatasetLoader(cache_manager=mock_cache)
        
        # データセットを読み込む
        dataset = loader.load_dataset(dataset_name)
        
        # メタデータの存在確認
        assert dataset.name is not None, "データセット名が存在しない"
        assert dataset.name == dataset_name, "データセット名が一致しない"
        
        # データとターゲットが空でないことを確認
        assert dataset.data is not None, "データが存在しない"
        assert dataset.target is not None, "ターゲットが存在しない"
        assert len(dataset.data) > 0, "データが空"
        assert len(dataset.target) > 0, "ターゲットが空"
        
        # サンプル数の確認
        n_samples = len(dataset.data)
        assert n_samples > 0, "サンプル数が0"
        assert len(dataset.target) == n_samples, "データとターゲットのサンプル数が一致しない"
        
        # 特徴量数の確認
        n_features = dataset.data.shape[1]
        assert n_features > 0, "特徴量数が0"
        
        # メタデータの完全性確認
        metadata = loader.get_dataset_metadata(dataset_name)
        assert "name" in metadata, "メタデータに名前が含まれていない"
        assert "n_samples" in metadata, "メタデータにサンプル数が含まれていない"
        assert "n_features" in metadata, "メタデータに特徴量数が含まれていない"
        assert metadata["name"] == dataset_name, "メタデータの名前が一致しない"
        assert metadata["n_samples"] == n_samples, "メタデータのサンプル数が一致しない"
        assert metadata["n_features"] == n_features, "メタデータの特徴量数が一致しない"
        
        # 特徴量名とターゲット名の存在確認
        assert dataset.feature_names is not None, "特徴量名が存在しない"
        assert dataset.target_names is not None, "ターゲット名が存在しない"
        assert len(dataset.feature_names) == n_features, "特徴量名の数が一致しない"
        assert len(dataset.target_names) > 0, "ターゲット名が空"


    # **Feature: ml-visualization-dashboard, Property 17: データセットキャッシュの保存**
    @given(
        dataset_name=st.sampled_from(["iris", "wine", "breast_cancer", "digits"])
    )
    @settings(max_examples=100)
    def test_property_dataset_cache_storage(self, test_cache_manager, dataset_name):
        """
        プロパティ17: データセットキャッシュの保存
        
        任意のデータセット名に対して、初回リクエスト後、
        Redisキャッシュにそのデータセットのキーが存在すること
        
        検証: 要件 7.1
        """
        # テスト用キャッシュマネージャーを使用
        cache_manager = test_cache_manager
        
        # キャッシュキーを生成
        cache_key = cache_manager.generate_cache_key("dataset", dataset_name)
        
        # テスト前にキャッシュをクリア（クリーンな状態から開始）
        cache_manager.delete(cache_key)
        
        # キャッシュが存在しないことを確認
        assert not cache_manager.exists(cache_key), (
            f"テスト開始前にキャッシュが存在している: {cache_key}"
        )
        
        # DatasetLoaderを作成してデータセットを読み込む
        loader = DatasetLoader(cache_manager=cache_manager)
        dataset = loader.load_dataset(dataset_name)
        
        # データセットが正常に読み込まれたことを確認
        assert dataset is not None, "データセットの読み込みに失敗"
        assert dataset.name == dataset_name, "データセット名が一致しない"
        
        # 初回リクエスト後、Redisキャッシュにキーが存在することを確認
        assert cache_manager.exists(cache_key), (
            f"初回リクエスト後にキャッシュキーが存在しない: {cache_key}"
        )
        
        # キャッシュから取得できることを確認
        cached_dataset = cache_manager.get(cache_key)
        assert cached_dataset is not None, (
            "キャッシュキーは存在するが、値を取得できない"
        )
        
        # キャッシュされたデータセットが元のデータセットと一致することを確認
        assert cached_dataset.name == dataset.name, (
            "キャッシュされたデータセット名が一致しない"
        )
        assert np.array_equal(cached_dataset.data, dataset.data), (
            "キャッシュされたデータが一致しない"
        )
        assert np.array_equal(cached_dataset.target, dataset.target), (
            "キャッシュされたターゲットが一致しない"
        )
        
        # テスト後のクリーンアップ
        cache_manager.delete(cache_key)

    # **Feature: ml-visualization-dashboard, Property 18: キャッシュからの取得**
    @given(
        dataset_name=st.sampled_from(["iris", "wine", "breast_cancer", "digits"])
    )
    @settings(max_examples=100)
    def test_property_cache_retrieval(self, test_cache_manager, dataset_name):
        """
        プロパティ18: キャッシュからの取得
        
        任意のキャッシュ済みデータセットに対して、2回目のリクエストで取得される
        データが、1回目のリクエストで取得されたデータと等しいこと
        
        検証: 要件 7.2
        """
        # テスト用キャッシュマネージャーを使用
        cache_manager = test_cache_manager
        
        # キャッシュキーを生成
        cache_key = cache_manager.generate_cache_key("dataset", dataset_name)
        
        # テスト前にキャッシュをクリア（クリーンな状態から開始）
        cache_manager.delete(cache_key)
        
        # DatasetLoaderを作成
        loader = DatasetLoader(cache_manager=cache_manager)
        
        # 1回目のリクエスト: scikit-learnから読み込み、キャッシュに保存
        dataset_first = loader.load_dataset(dataset_name)
        
        # データセットが正常に読み込まれたことを確認
        assert dataset_first is not None, "1回目のデータセット読み込みに失敗"
        assert dataset_first.name == dataset_name, "1回目のデータセット名が一致しない"
        
        # キャッシュに保存されたことを確認
        assert cache_manager.exists(cache_key), (
            "1回目のリクエスト後にキャッシュが存在しない"
        )
        
        # 2回目のリクエスト: キャッシュから取得
        dataset_second = loader.load_dataset(dataset_name)
        
        # データセットが正常に取得されたことを確認
        assert dataset_second is not None, "2回目のデータセット取得に失敗"
        assert dataset_second.name == dataset_name, "2回目のデータセット名が一致しない"
        
        # 1回目と2回目のデータが完全に一致することを確認
        assert dataset_first.name == dataset_second.name, (
            "1回目と2回目でデータセット名が一致しない"
        )
        
        assert np.array_equal(dataset_first.data, dataset_second.data), (
            "1回目と2回目でデータが一致しない"
        )
        
        assert np.array_equal(dataset_first.target, dataset_second.target), (
            "1回目と2回目でターゲットが一致しない"
        )
        
        assert dataset_first.feature_names == dataset_second.feature_names, (
            "1回目と2回目で特徴量名が一致しない"
        )
        
        assert dataset_first.target_names == dataset_second.target_names, (
            "1回目と2回目でターゲット名が一致しない"
        )
        
        assert dataset_first.DESCR == dataset_second.DESCR, (
            "1回目と2回目で説明文が一致しない"
        )
        
        # データの形状も確認
        assert dataset_first.data.shape == dataset_second.data.shape, (
            "1回目と2回目でデータの形状が一致しない"
        )
        
        assert dataset_first.target.shape == dataset_second.target.shape, (
            "1回目と2回目でターゲットの形状が一致しない"
        )
        
        # テスト後のクリーンアップ
        cache_manager.delete(cache_key)


class TestDataPreprocessor:
    """DataPreprocessorの基本テスト"""
    
    def test_split_data_basic(self):
        """基本的なデータ分割のテスト"""
        # テストデータ生成（十分なサンプル数）
        np.random.seed(42)
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)
        
        X_train, X_test, y_train, y_test = DataPreprocessor.split_data(
            X, y, test_size=0.3, random_state=42
        )
        
        # 分割が正しく行われたか確認
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
        # テストサイズが期待値に近いか確認（±1%）
        expected_test_size = int(len(X) * 0.3)
        assert abs(len(X_test) - expected_test_size) <= 1
    
    def test_split_data_reproducibility(self):
        """同じシードで再現性があるかテスト"""
        np.random.seed(42)
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)
        
        # 同じシードで2回分割
        X_train1, X_test1, y_train1, y_test1 = DataPreprocessor.split_data(
            X, y, test_size=0.3, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = DataPreprocessor.split_data(
            X, y, test_size=0.3, random_state=42
        )
        
        # 結果が一致するか確認
        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(X_test1, X_test2)
        assert np.array_equal(y_train1, y_train2)
        assert np.array_equal(y_test1, y_test2)
    
    def test_prepare_preview(self):
        """データプレビュー生成のテスト"""
        # テストデータセット作成
        dataset = Dataset(
            data=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
            target=np.array([0, 1, 0, 1, 0]),
            feature_names=["feature_0", "feature_1"],
            target_names=["class_0", "class_1"],
            DESCR="Test dataset",
            name="test"
        )
        
        preview = DataPreprocessor.prepare_preview(dataset, n_rows=3)
        
        assert "data" in preview
        assert "columns" in preview
        assert "n_rows" in preview
        assert preview["n_rows"] == 3
        assert len(preview["data"]) == 3
        assert "target" in preview["columns"]
    
    def test_prepare_preview_exceeds_dataset_size(self):
        """プレビュー行数がデータセットサイズを超える場合のテスト"""
        dataset = Dataset(
            data=np.array([[1, 2], [3, 4]]),
            target=np.array([0, 1]),
            feature_names=["feature_0", "feature_1"],
            target_names=["class_0", "class_1"],
            DESCR="Small test dataset",
            name="test_small"
        )
        
        preview = DataPreprocessor.prepare_preview(dataset, n_rows=10)
        
        # データセットサイズを超えない
        assert preview["n_rows"] == 2
        assert len(preview["data"]) == 2
    
    def test_get_split_info(self):
        """分割情報取得のテスト"""
        X_train = np.random.rand(70, 4)
        X_test = np.random.rand(30, 4)
        y_train = np.random.randint(0, 3, 70)
        y_test = np.random.randint(0, 3, 30)
        
        info = DataPreprocessor.get_split_info(X_train, X_test, y_train, y_test)
        
        assert info["train_samples"] == 70
        assert info["test_samples"] == 30
        assert info["total_samples"] == 100
        assert abs(info["actual_test_ratio"] - 0.3) < 0.01
        assert info["n_features"] == 4
    
    # **Feature: ml-visualization-dashboard, Property 2: データ分割比率の正確性**
    @given(
        n_samples=st.integers(min_value=100, max_value=1000),
        n_features=st.integers(min_value=2, max_value=20),
        test_size=st.floats(min_value=0.1, max_value=0.5),
        random_state=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_property_data_split_ratio_accuracy(self, n_samples, n_features, test_size, random_state):
        """
        プロパティ2: データ分割比率の正確性
        
        任意の10%から50%の範囲のテスト分割比率に対して、
        実際のテストセットサイズが指定された比率の±1%以内であること
        
        検証: 要件 1.2
        """
        # テストデータ生成
        np.random.seed(random_state)
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)
        
        # データ分割
        X_train, X_test, y_train, y_test = DataPreprocessor.split_data(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 実際のテスト比率を計算
        total_samples = len(X_train) + len(X_test)
        actual_test_ratio = len(X_test) / total_samples
        
        # 指定された比率の±1%以内であることを確認
        tolerance = 0.01
        assert abs(actual_test_ratio - test_size) <= tolerance, (
            f"テスト分割比率が許容範囲外: "
            f"期待={test_size:.4f}, 実際={actual_test_ratio:.4f}, "
            f"差={abs(actual_test_ratio - test_size):.4f}, 許容={tolerance}"
        )
        
        # 追加の整合性チェック
        assert len(X_train) + len(X_test) == n_samples, "分割後のサンプル数が元のサンプル数と一致しない"
        assert len(y_train) + len(y_test) == n_samples, "分割後のターゲット数が元のターゲット数と一致しない"
        assert len(X_train) == len(y_train), "訓練データとターゲットのサイズが一致しない"
        assert len(X_test) == len(y_test), "テストデータとターゲットのサイズが一致しない"
    
    # **Feature: ml-visualization-dashboard, Property 3: 乱数シードによる再現性**
    @given(
        n_samples=st.integers(min_value=100, max_value=1000),
        n_features=st.integers(min_value=2, max_value=20),
        test_size=st.floats(min_value=0.1, max_value=0.5),
        random_state=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_property_random_seed_reproducibility(self, n_samples, n_features, test_size, random_state):
        """
        プロパティ3: 乱数シードによる再現性
        
        任意の乱数シード値に対して、同じシードで2回データ分割を実行した場合、
        train/testセットが完全に一致すること
        
        検証: 要件 1.3
        """
        # テストデータ生成
        np.random.seed(random_state)
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)
        
        # 同じシードで2回データ分割を実行
        X_train1, X_test1, y_train1, y_test1 = DataPreprocessor.split_data(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train2, X_test2, y_train2, y_test2 = DataPreprocessor.split_data(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 訓練セットが完全に一致することを確認
        assert np.array_equal(X_train1, X_train2), (
            "同じシードで分割した訓練特徴量が一致しない"
        )
        assert np.array_equal(y_train1, y_train2), (
            "同じシードで分割した訓練ターゲットが一致しない"
        )
        
        # テストセットが完全に一致することを確認
        assert np.array_equal(X_test1, X_test2), (
            "同じシードで分割したテスト特徴量が一致しない"
        )
        assert np.array_equal(y_test1, y_test2), (
            "同じシードで分割したテストターゲットが一致しない"
        )
        
        # 追加の整合性チェック: 分割サイズも一致することを確認
        assert len(X_train1) == len(X_train2), "訓練セットのサイズが一致しない"
        assert len(X_test1) == len(X_test2), "テストセットのサイズが一致しない"
    
    # **Feature: ml-visualization-dashboard, Property 4: データプレビューの正確性**
    @given(
        dataset_name=st.sampled_from(["iris", "wine", "breast_cancer", "digits"]),
        n_rows=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100)
    def test_property_data_preview_accuracy(self, dataset_name, n_rows):
        """
        プロパティ4: データプレビューの正確性
        
        任意のデータセットとプレビュー行数Nに対して、プレビューテーブルが
        正確にN行（またはデータセットの全行数がN未満の場合はその行数）を含み、
        すべての特徴量列とラベル列が存在すること
        
        検証: 要件 1.4
        """
        # データセットを読み込む
        mock_cache = Mock(spec=CacheManager)
        mock_cache.generate_cache_key.return_value = f"dataset:{dataset_name}"
        mock_cache.get.return_value = None  # キャッシュなし
        mock_cache.set.return_value = True
        
        loader = DatasetLoader(cache_manager=mock_cache)
        dataset = loader.load_dataset(dataset_name)
        
        # プレビューを生成
        preview = DataPreprocessor.prepare_preview(dataset, n_rows=n_rows)
        
        # プレビューの構造を確認
        assert "data" in preview, "プレビューに'data'キーが存在しない"
        assert "columns" in preview, "プレビューに'columns'キーが存在しない"
        assert "n_rows" in preview, "プレビューに'n_rows'キーが存在しない"
        
        # 実際の行数を確認（データセットサイズを超えない）
        expected_rows = min(n_rows, len(dataset.data))
        assert preview["n_rows"] == expected_rows, (
            f"プレビュー行数が期待値と一致しない: "
            f"期待={expected_rows}, 実際={preview['n_rows']}"
        )
        
        # データの行数を確認
        assert len(preview["data"]) == expected_rows, (
            f"プレビューデータの行数が期待値と一致しない: "
            f"期待={expected_rows}, 実際={len(preview['data'])}"
        )
        
        # すべての特徴量列が存在することを確認
        for feature_name in dataset.feature_names:
            assert feature_name in preview["columns"], (
                f"特徴量列 '{feature_name}' がプレビューに存在しない"
            )
        
        # ターゲット列が存在することを確認
        assert "target" in preview["columns"], (
            "ターゲット列がプレビューに存在しない"
        )
        
        # 各行にすべての列が含まれていることを確認
        for i, row in enumerate(preview["data"]):
            # すべての特徴量列が各行に存在することを確認
            for feature_name in dataset.feature_names:
                assert feature_name in row, (
                    f"行{i}に特徴量列 '{feature_name}' が存在しない"
                )
            
            # ターゲット列が各行に存在することを確認
            assert "target" in row, (
                f"行{i}にターゲット列が存在しない"
            )
        
        # カラム数の確認（特徴量 + ターゲット）
        expected_columns = len(dataset.feature_names) + 1  # +1 for target
        assert len(preview["columns"]) == expected_columns, (
            f"カラム数が期待値と一致しない: "
            f"期待={expected_columns}, 実際={len(preview['columns'])}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
