# src/train_model.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def load_training_data(csv_path: str) -> pd.DataFrame:
    """
    加载训练数据，要求至少包含：
    - customer_id：客户ID（不会作为特征）
    - label：0/1 标签
    - 其余为特征列（数值或类别均可）
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"训练数据文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("训练数据中必须包含 'label' 列（0/1 标签）")
    if "customer_id" not in df.columns:
        raise ValueError("训练数据中必须包含 'customer_id' 列")
    return df


def build_model_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    根据特征自动构建预处理 + 模型 Pipeline：
    - 数值特征：缺失值填充（中位数）
    - 类别特征：缺失值填充 + OneHot
    - 模型：RandomForestClassifier（你之后可以换成 XGBoost/LightGBM）
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return model


def train_and_save_model(
    csv_path: str = os.path.join("data", "crm_training_data.csv"),
    model_path: str = os.path.join("models", "buy_model.pkl"),
):
    # 1. 加载数据
    df = load_training_data(csv_path)

    # 2. 拆分特征和标签
    feature_cols = [c for c in df.columns if c not in ("customer_id", "label")]
    X = df[feature_cols]
    y = df["label"]

    # 3. 切分训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. 构建 Pipeline
    model = build_model_pipeline(X_train)

    # 5. 训练
    print("开始训练模型...")
    model.fit(X_train, y_train)

    # 6. 在测试集上评估
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== 分类报告 ===")
    print(classification_report(y_test, y_pred))

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"AUC: {auc:.4f}")
    except Exception:
        pass

    # 7. 保存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n模型已保存到: {model_path}")


if __name__ == "__main__":
    train_and_save_model()
