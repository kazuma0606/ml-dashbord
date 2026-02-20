"""
プロパティベーステスト: 実験記録リポジトリ

**Feature: ml-visualization-dashboard, Property 11: パラメータ保存のラウンドトリップ**
**検証: 要件 5.1**
"""
import pytest
from datetime import datetime
from hypothesis import given, strategies as st, settings
from sqlalchemy.orm import Session

from src.repositories.experiment_repository import ExperimentRepository
from src.models.schemas import ExperimentRecord


# ハイパーパラメータ生成戦略
@st.composite
def hyperparameters_strategy(draw):
    """様々なハイパーパラメータの組み合わせを生成"""
    model_type = draw(st.sampled_from([
        "random_forest",
        "gradient_boosting",
        "svm",
        "logistic_regression",
        "knn"
    ]))
    
    params = {}
    
    if model_type in ["random_forest", "gradient_boosting"]:
        params["n_estimators"] = draw(st.integers(min_value=10, max_value=200))
        params["max_depth"] = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=50)))
        params["min_samples_split"] = draw(st.integers(min_value=2, max_value=20))
        
        if model_type == "gradient_boosting":
            params["learning_rate"] = draw(st.floats(min_value=0.01, max_value=1.0))
    
    elif model_type in ["svm", "logistic_regression"]:
        params["C"] = draw(st.floats(min_value=0.01, max_value=100.0))
        
        if model_type == "svm":
            params["kernel"] = draw(st.sampled_from(["linear", "rbf", "poly"]))
    
    elif model_type == "knn":
        params["n_neighbors"] = draw(st.integers(min_value=1, max_value=20))
        params["weights"] = draw(st.sampled_from(["uniform", "distance"]))
    
    return params


# 実験記録生成戦略
@st.composite
def experiment_record_strategy(draw):
    """ランダムな実験記録を生成"""
    dataset_name = draw(st.sampled_from(["iris", "wine", "breast_cancer", "digits"]))
    model_type = draw(st.sampled_from([
        "random_forest",
        "gradient_boosting", 
        "svm",
        "logistic_regression",
        "knn"
    ]))
    accuracy = draw(st.floats(min_value=0.0, max_value=1.0))
    f1_score = draw(st.floats(min_value=0.0, max_value=1.0))
    hyperparameters = draw(hyperparameters_strategy())
    training_time = draw(st.floats(min_value=0.01, max_value=100.0))
    
    return ExperimentRecord(
        dataset_name=dataset_name,
        model_type=model_type,
        accuracy=accuracy,
        f1_score=f1_score,
        hyperparameters=hyperparameters,
        training_time=training_time,
        timestamp=datetime.now()
    )


class TestHyperparameterRoundtrip:
    """パラメータ保存のラウンドトリップテスト"""
    
    # **Feature: ml-visualization-dashboard, Property 11: パラメータ保存のラウンドトリップ**
    @given(experiment=experiment_record_strategy())
    @settings(max_examples=100, deadline=None)
    def test_hyperparameter_save_roundtrip(self, test_engine, experiment: ExperimentRecord):
        """
        プロパティ11: パラメータ保存のラウンドトリップ
        
        任意のハイパーパラメータ設定に対して、データベースに保存してから取得した設定が、
        元の設定と等しいこと
        
        **検証: 要件 5.1**
        """
        # テスト用セッションを作成
        from sqlalchemy.orm import sessionmaker
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        db_session = TestSessionLocal()
        
        try:
            # リポジトリ作成
            repository = ExperimentRepository(db_session)
            
            # 元のハイパーパラメータを保存
            original_hyperparameters = experiment.hyperparameters.copy()
            
            # データベースに保存
            experiment_id = repository.save(experiment)
            
            # データベースから取得
            retrieved_experiments = repository.get_all()
            
            # 保存した実験が取得できることを確認
            assert len(retrieved_experiments) > 0
            
            # 保存した実験を見つける
            retrieved_experiment = next(
                (exp for exp in retrieved_experiments if exp.id == experiment_id),
                None
            )
            
            assert retrieved_experiment is not None, "保存した実験が取得できませんでした"
            
            # ハイパーパラメータが完全に一致することを確認（ラウンドトリップ）
            assert retrieved_experiment.hyperparameters == original_hyperparameters, (
                f"ハイパーパラメータが一致しません。\n"
                f"元の値: {original_hyperparameters}\n"
                f"取得した値: {retrieved_experiment.hyperparameters}"
            )
            
            # その他のフィールドも確認
            assert retrieved_experiment.dataset_name == experiment.dataset_name
            assert retrieved_experiment.model_type == experiment.model_type
            assert abs(retrieved_experiment.accuracy - experiment.accuracy) < 1e-6
            assert abs(retrieved_experiment.f1_score - experiment.f1_score) < 1e-6
            assert abs(retrieved_experiment.training_time - experiment.training_time) < 1e-6
            
            # テスト後にクリーンアップ
            repository.clear()
            
        finally:
            db_session.close()


class TestExperimentRecordCompleteness:
    """実験記録の完全性テスト"""
    
    # **Feature: ml-visualization-dashboard, Property 12: 実験記録の完全性**
    @given(experiment=experiment_record_strategy())
    @settings(max_examples=100, deadline=None)
    def test_experiment_record_completeness(self, test_engine, experiment: ExperimentRecord):
        """
        プロパティ12: 実験記録の完全性
        
        任意の学習実行完了後、データベースに保存された実験記録が、
        モデルタイプ、データセット名、accuracy、f1_score、hyperparameters、timestampの
        すべてのフィールドを含むこと
        
        **検証: 要件 5.2**
        """
        # テスト用セッションを作成
        from sqlalchemy.orm import sessionmaker
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        db_session = TestSessionLocal()
        
        try:
            # リポジトリ作成
            repository = ExperimentRepository(db_session)
            
            # データベースに保存
            experiment_id = repository.save(experiment)
            
            # データベースから取得
            retrieved_experiments = repository.get_all()
            
            # 保存した実験を見つける
            retrieved_experiment = next(
                (exp for exp in retrieved_experiments if exp.id == experiment_id),
                None
            )
            
            assert retrieved_experiment is not None, "保存した実験が取得できませんでした"
            
            # すべての必須フィールドが存在し、None以外の値を持つことを確認
            assert retrieved_experiment.model_type is not None, "model_typeフィールドが存在しません"
            assert retrieved_experiment.model_type != "", "model_typeが空文字列です"
            
            assert retrieved_experiment.dataset_name is not None, "dataset_nameフィールドが存在しません"
            assert retrieved_experiment.dataset_name != "", "dataset_nameが空文字列です"
            
            assert retrieved_experiment.accuracy is not None, "accuracyフィールドが存在しません"
            assert isinstance(retrieved_experiment.accuracy, (int, float)), "accuracyが数値ではありません"
            
            assert retrieved_experiment.f1_score is not None, "f1_scoreフィールドが存在しません"
            assert isinstance(retrieved_experiment.f1_score, (int, float)), "f1_scoreが数値ではありません"
            
            assert retrieved_experiment.hyperparameters is not None, "hyperparametersフィールドが存在しません"
            assert isinstance(retrieved_experiment.hyperparameters, dict), "hyperparametersが辞書ではありません"
            
            assert retrieved_experiment.timestamp is not None, "timestampフィールドが存在しません"
            assert isinstance(retrieved_experiment.timestamp, datetime), "timestampがdatetimeではありません"
            
            # training_timeも確認（要件には明示されていないが、ExperimentRecordの一部）
            assert retrieved_experiment.training_time is not None, "training_timeフィールドが存在しません"
            assert isinstance(retrieved_experiment.training_time, (int, float)), "training_timeが数値ではありません"
            
            # テスト後にクリーンアップ
            repository.clear()
            
        finally:
            db_session.close()


class TestExperimentHistoryChronologicalOrder:
    """実験履歴の時系列順序テスト"""
    
    # **Feature: ml-visualization-dashboard, Property 13: 実験履歴の時系列順序**
    @given(
        num_experiments=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_experiment_history_chronological_order(self, test_engine, num_experiments: int):
        """
        プロパティ13: 実験履歴の時系列順序
        
        任意の実験記録セットに対して、取得した履歴リストがtimestampの降順
        （最新が先頭）にソートされていること
        
        **検証: 要件 5.3**
        """
        # テスト用セッションを作成
        from sqlalchemy.orm import sessionmaker
        from datetime import timedelta
        import time
        
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        db_session = TestSessionLocal()
        
        try:
            # リポジトリ作成
            repository = ExperimentRepository(db_session)
            
            # 複数の実験記録を異なるタイムスタンプで作成
            base_time = datetime.now()
            saved_experiments = []
            
            for i in range(num_experiments):
                # 各実験に異なるタイムスタンプを設定（古い順に作成）
                experiment = ExperimentRecord(
                    dataset_name="iris",
                    model_type="random_forest",
                    accuracy=0.9 + (i * 0.01),
                    f1_score=0.85 + (i * 0.01),
                    hyperparameters={"n_estimators": 100 + i},
                    training_time=1.5 + (i * 0.1),
                    timestamp=base_time + timedelta(seconds=i)
                )
                
                # データベースに保存
                experiment_id = repository.save(experiment)
                saved_experiments.append((experiment_id, experiment.timestamp))
                
                # タイムスタンプが確実に異なるように少し待機
                time.sleep(0.01)
            
            # データベースから取得
            retrieved_experiments = repository.get_all()
            
            # 取得した実験数が保存した数と一致することを確認
            assert len(retrieved_experiments) >= num_experiments, (
                f"保存した実験数（{num_experiments}）と取得した実験数（{len(retrieved_experiments)}）が一致しません"
            )
            
            # タイムスタンプが降順（最新が先頭）にソートされていることを確認
            for i in range(len(retrieved_experiments) - 1):
                current_timestamp = retrieved_experiments[i].timestamp
                next_timestamp = retrieved_experiments[i + 1].timestamp
                
                assert current_timestamp >= next_timestamp, (
                    f"実験履歴が時系列降順にソートされていません。\n"
                    f"インデックス {i} のタイムスタンプ: {current_timestamp}\n"
                    f"インデックス {i+1} のタイムスタンプ: {next_timestamp}\n"
                    f"最新の実験が先頭に来る必要があります。"
                )
            
            # 最初の要素が最新のタイムスタンプを持つことを確認
            if len(retrieved_experiments) > 0:
                max_timestamp = max(exp.timestamp for exp in retrieved_experiments)
                assert retrieved_experiments[0].timestamp == max_timestamp, (
                    "最初の要素が最新のタイムスタンプを持っていません"
                )
            
            # テスト後にクリーンアップ
            repository.clear()
            
        finally:
            db_session.close()



class TestHistoryClearCompleteness:
    """履歴クリアの完全性テスト"""
    
    # **Feature: ml-visualization-dashboard, Property 15: 履歴クリアの完全性**
    @given(
        num_experiments=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_history_clear_completeness(self, test_engine, num_experiments: int):
        """
        プロパティ15: 履歴クリアの完全性
        
        任意の実験記録セットに対して、履歴リセットを実行した後、
        データベースから取得される実験記録が空リストであること
        
        **検証: 要件 5.5**
        """
        # テスト用セッションを作成
        from sqlalchemy.orm import sessionmaker
        
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        db_session = TestSessionLocal()
        
        try:
            # リポジトリ作成
            repository = ExperimentRepository(db_session)
            
            # 複数の実験記録を作成して保存
            saved_ids = []
            for i in range(num_experiments):
                experiment = ExperimentRecord(
                    dataset_name=f"dataset_{i}",
                    model_type="random_forest",
                    accuracy=0.8 + (i * 0.01),
                    f1_score=0.75 + (i * 0.01),
                    hyperparameters={"n_estimators": 100 + i, "max_depth": 10 + i},
                    training_time=1.0 + (i * 0.1),
                    timestamp=datetime.now()
                )
                experiment_id = repository.save(experiment)
                saved_ids.append(experiment_id)
            
            # 保存前の確認：実験記録が存在することを確認
            experiments_before_clear = repository.get_all()
            assert len(experiments_before_clear) >= num_experiments, (
                f"保存した実験数（{num_experiments}）が取得できませんでした。"
                f"取得した数: {len(experiments_before_clear)}"
            )
            
            # 履歴をクリア
            clear_result = repository.clear()
            assert clear_result is True, "clear()メソッドがTrueを返しませんでした"
            
            # クリア後の確認：実験記録が空であることを確認
            experiments_after_clear = repository.get_all()
            assert len(experiments_after_clear) == 0, (
                f"履歴クリア後に実験記録が残っています。\n"
                f"残っている実験数: {len(experiments_after_clear)}\n"
                f"期待値: 0\n"
                f"履歴クリアは完全にすべての記録を削除する必要があります。"
            )
            
            # 念のため、保存したIDで個別に確認（すべて存在しないはず）
            all_experiments = repository.get_all()
            for saved_id in saved_ids:
                found = any(exp.id == saved_id for exp in all_experiments)
                assert not found, (
                    f"クリア後にID {saved_id} の実験記録が見つかりました。"
                    f"すべての記録が削除されるべきです。"
                )
            
        finally:
            db_session.close()
