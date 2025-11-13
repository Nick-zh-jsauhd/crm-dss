# src/predict.py
import os
import joblib
import pandas as pd


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    return joblib.load(model_path)


def predict_new_customers(
    model_path: str = os.path.join("models", "buy_model.pkl"),
    input_csv: str = os.path.join("data", "crm_new_customers.csv"),
    output_csv: str = os.path.join("data", "crm_new_with_predictions.csv"),
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"待预测数据文件不存在: {input_csv}")

    df = pd.read_csv(input_csv)
    if "customer_id" not in df.columns:
        raise ValueError("待预测数据中必须包含 'customer_id' 列")

    feature_cols = [c for c in df.columns if c != "customer_id"]
    X = df[feature_cols]

    model = load_model(model_path)

    # 预测概率
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    df["prob"] = proba
    df["pred_label"] = pred

    df.to_csv(output_csv, index=False)
    print(f"预测结果已保存到: {output_csv}")


if __name__ == "__main__":
    predict_new_customers()
