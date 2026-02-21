"""Visualization components for ML Dashboard."""
from typing import Optional, List, Dict, Any
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render_data_preview(
    preview_data: Optional[Dict[str, Any]] = None,
    n_rows: int = 10
) -> None:
    """Render data preview table with colored labels.
    
    Args:
        preview_data: Dictionary containing 'features', 'target', 'feature_names', 'target_names'
        n_rows: Number of rows to display
    """
    st.subheader("📋 Data Preview")
    
    if preview_data is None:
        st.info("Load a dataset to see preview")
        return
    
    try:
        # Create DataFrame from preview data
        features = preview_data.get("features", [])
        target = preview_data.get("target", [])
        feature_names = preview_data.get("feature_names", [])
        target_names = preview_data.get("target_names", [])
        
        if not features or not target:
            st.warning("No preview data available")
            return
        
        # Build DataFrame
        df = pd.DataFrame(features, columns=feature_names)
        
        # Add target column with names if available
        if target_names:
            df["Target"] = [target_names[int(t)] if int(t) < len(target_names) else str(t) for t in target]
        else:
            df["Target"] = target
        
        # Display with styling
        st.dataframe(
            df.head(n_rows),
            use_container_width=True,
            hide_index=False
        )
        
    except Exception as e:
        st.error(f"Failed to render preview: {str(e)}")


def render_confusion_matrix(
    confusion_matrix: Optional[List[List[int]]] = None,
    target_names: Optional[List[str]] = None
) -> None:
    """Render confusion matrix heatmap.
    
    Args:
        confusion_matrix: 2D list representing confusion matrix
        target_names: Names of target classes
    """
    st.subheader("🔥 Confusion Matrix")
    
    if confusion_matrix is None:
        st.info("Train a model to see confusion matrix")
        return
    
    try:
        # Create labels
        if target_names:
            labels = target_names
        else:
            n_classes = len(confusion_matrix)
            labels = [f"Class {i}" for i in range(n_classes)]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
            width=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to render confusion matrix: {str(e)}")


def render_feature_importance(
    feature_importances: Optional[List[float]] = None,
    feature_names: Optional[List[str]] = None
) -> None:
    """Render feature importance horizontal bar chart.
    
    Args:
        feature_importances: List of importance values (sorted descending)
        feature_names: Names of features
    """
    st.subheader("📊 Feature Importance")
    
    if feature_importances is None:
        st.info("Train a tree-based model to see feature importance")
        return
    
    try:
        # Create DataFrame
        if feature_names and len(feature_names) == len(feature_importances):
            df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": feature_importances
            })
        else:
            df = pd.DataFrame({
                "Feature": [f"Feature {i}" for i in range(len(feature_importances))],
                "Importance": feature_importances
            })
        
        # Sort by importance (descending)
        df = df.sort_values("Importance", ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(
            df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance (Descending Order)",
            labels={"Importance": "Importance Score", "Feature": "Feature Name"}
        )
        
        fig.update_layout(height=max(300, len(df) * 25))
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to render feature importance: {str(e)}")


def render_experiment_history(
    experiments: Optional[List[Dict[str, Any]]] = None
) -> None:
    """Render experiment history table.
    
    Args:
        experiments: List of experiment records
    """
    st.subheader("📜 Experiment History")
    
    if not experiments:
        st.info("No experiments recorded yet")
        return
    
    try:
        # Create DataFrame
        df = pd.DataFrame(experiments)
        
        # Format columns
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        
        if "accuracy" in df.columns:
            df["accuracy"] = df["accuracy"].apply(lambda x: f"{x:.4f}")
        
        if "f1_score" in df.columns:
            df["f1_score"] = df["f1_score"].apply(lambda x: f"{x:.4f}")
        
        if "model_type" in df.columns:
            df["model_type"] = df["model_type"].apply(lambda x: x.replace("_", " ").title())
        
        # Select columns to display
        display_columns = ["timestamp", "dataset_name", "model_type", "accuracy", "f1_score"]
        display_columns = [col for col in display_columns if col in df.columns]
        
        st.dataframe(
            df[display_columns],
            use_container_width=True,
            hide_index=True
        )
        
    except Exception as e:
        st.error(f"Failed to render experiment history: {str(e)}")


def render_accuracy_trend(
    experiments: Optional[List[Dict[str, Any]]] = None
) -> None:
    """Render accuracy trend bar chart with best model highlighted.
    
    Args:
        experiments: List of experiment records
    """
    st.subheader("📈 Accuracy Trend")
    
    if not experiments:
        st.info("No experiments to visualize")
        return
    
    try:
        # Create DataFrame
        df = pd.DataFrame(experiments)
        
        if "accuracy" not in df.columns:
            st.warning("No accuracy data available")
            return
        
        # Find best model
        best_idx = df["accuracy"].idxmax()
        df["is_best"] = False
        df.loc[best_idx, "is_best"] = True
        
        # Create labels
        df["label"] = df.apply(
            lambda row: f"{row.get('model_type', 'Model').replace('_', ' ').title()} ({row.get('dataset_name', 'Dataset')})",
            axis=1
        )
        
        # Create bar chart
        fig = go.Figure()
        
        # Regular bars
        regular_df = df[~df["is_best"]]
        if not regular_df.empty:
            fig.add_trace(go.Bar(
                x=regular_df.index,
                y=regular_df["accuracy"],
                name="Experiments",
                marker_color="lightblue",
                text=regular_df["accuracy"].apply(lambda x: f"{x:.4f}"),
                textposition="outside",
                hovertext=regular_df["label"],
                hoverinfo="text+y"
            ))
        
        # Best model bar
        best_df = df[df["is_best"]]
        if not best_df.empty:
            fig.add_trace(go.Bar(
                x=best_df.index,
                y=best_df["accuracy"],
                name="Best Model",
                marker_color="gold",
                text=best_df["accuracy"].apply(lambda x: f"{x:.4f}"),
                textposition="outside",
                hovertext=best_df["label"],
                hoverinfo="text+y"
            ))
        
        fig.update_layout(
            title="Accuracy Across Experiments (Best Model Highlighted)",
            xaxis_title="Experiment #",
            yaxis_title="Accuracy",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to render accuracy trend: {str(e)}")


def render_classification_report(
    classification_report: Optional[Dict[str, Any]] = None
) -> None:
    """Render classification report in expandable panel.
    
    Args:
        classification_report: Dictionary containing precision, recall, f1-score per class
    """
    with st.expander("📄 Classification Report", expanded=False):
        if classification_report is None:
            st.info("Train a model to see classification report")
            return
        
        try:
            # Convert to DataFrame for better display
            # Remove 'accuracy' key if present (it's a single value, not per-class)
            report_dict = {k: v for k, v in classification_report.items() 
                          if isinstance(v, dict)}
            
            if report_dict:
                df = pd.DataFrame(report_dict).T
                
                # Format numeric columns
                for col in df.columns:
                    if df[col].dtype in ['float64', 'float32']:
                        df[col] = df[col].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(df, use_container_width=True)
            else:
                st.json(classification_report)
                
        except Exception as e:
            st.error(f"Failed to render classification report: {str(e)}")
