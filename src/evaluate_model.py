# src/evaluate_model.py
import json
import os
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def load_trained_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    return joblib.load(model_path)


def load_labeled_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"评估数据文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("评估数据必须包含 'label' 列 (0/1)")
    if "customer_id" not in df.columns:
        raise ValueError("评估数据必须包含 'customer_id' 列")
    return df


def evaluate_on_dataframe(model, df: pd.DataFrame) -> Dict[str, Any]:
    """
    使用已训练的模型在带标签的数据集上计算准确率/召回率等指标。
    """
    feature_cols = [c for c in df.columns if c not in ("customer_id", "label")]
    X = df[feature_cols]
    y_true = df["label"].values

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    except Exception:
        metrics["roc_auc"] = None

    return metrics


def evaluate_model(
    model_path: str = os.path.join("models", "buy_model.pkl"),
    data_path: str = os.path.join("data", "crm_training_data.csv"),
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    model = load_trained_model(model_path)
    df = load_labeled_data(data_path)

    metrics = evaluate_on_dataframe(model, df)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


if __name__ == "__main__":
    results = evaluate_model()
    print("=== 模型评估结果 ===")
    for k, v in results.items():
        if k == "confusion_matrix":
            print(f"{k}: {v} (格式: [[TN, FP], [FN, TP]])")
        elif k == "classification_report":
            print("\n分类报告:\n")
            print(v)
        else:
            print(f"{k}: {v}")
