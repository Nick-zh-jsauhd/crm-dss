# app/app.py
import os
import sys

import pandas as pd
import streamlit as st
import joblib

# ====== 路径配置 ======
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

from evaluate_model import evaluate_on_dataframe, load_labeled_data
from rules import classify_customer
from llm_agent import (
    generate_advice_template,
    generate_advice_with_llm,
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "buy_model.pkl")
SAMPLE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "crm_test_data.csv")
TRAINING_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "crm_training_data.csv")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("模型文件不存在，请先在项目根目录运行：python -m src.train_model")
        st.stop()
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(page_title="CRM 决策支持系统", layout="wide")
    st.title("📊 CRM 决策支持系统 - Yannick")
    st.markdown(
        "该系统基于历史数据训练的模型，为客户成交概率预测提供支持，并生成跟进建议。"
    )

    model = load_model()

    # ============= LLM 设置 =============
    st.sidebar.header("🔑 LLM 设置（DeepSeek）")

    # 1) 尝试读取环境变量（Streamlit Cloud secrets）
    default_key = os.getenv("DEEPSEEK_API_KEY", "")

    api_key_input = st.sidebar.text_input(
        "输入你的 DeepSeek API Key（可选）",
        type="password",
        help="不填写则使用服务器默认配置（如果已配置）；本地运行时可手动填写。",
        value=""  # 不在界面预填，避免泄露
    )

    # 初始化 session_state
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    # 优先使用用户输入，其次使用环境变量
    if api_key_input:
        st.session_state["api_key"] = api_key_input
    else:
        st.session_state["api_key"] = default_key

    # ============= 数据来源选择 =============
    st.sidebar.header("📂 数据与预测")

    data_source = st.sidebar.radio(
        "选择数据来源",
        ("使用示例数据集", "上传自定义CSV"),
    )

    df = None

    if data_source.startswith("使用示例数据集"):
        # 使用仓库内自带的示例数据
        if not os.path.exists(SAMPLE_DATA_PATH):
            st.error(
                "示例数据集 data/crm_test_data.csv 不存在，请在项目根目录下创建 data 文件夹并放入该 CSV 后重新部署。"
            )
            return

        df = pd.read_csv(SAMPLE_DATA_PATH)
        st.sidebar.success("已加载示例数据集")
    else:
        # 用户自定义上传
        uploaded_file = st.sidebar.file_uploader("上传客户特征数据（CSV）", type=["csv"])
        if uploaded_file is None:
            st.info("请在左侧上传待预测的客户数据，或选择“使用示例数据集”。")
            return
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("已加载自定义数据集。")

    # 到这里 df 一定已经有值
    if "customer_id" not in df.columns:
        st.error("数据中必须包含 'customer_id' 列。")
        return

    feature_cols = [c for c in df.columns if c != "customer_id"]
    X = df[feature_cols]

    # ====== 预测结果缓存 ======
    if "pred_df" not in st.session_state:
        st.session_state["pred_df"] = None

    # ====== 点击按钮运行预测 ======
    if st.sidebar.button("运行预测"):
        with st.spinner("正在运行模型预测..."):
            proba = model.predict_proba(X)[:, 1]
            pred = (proba >= 0.5).astype(int)

            df_pred = df.copy()
            df_pred["prob"] = proba
            df_pred["pred_label"] = pred

            df_pred["rule_advice"] = [
                classify_customer(row, p) for (_, row), p in zip(df_pred.iterrows(), proba)
            ]

        st.session_state["pred_df"] = df_pred
        st.success("预测完成！")

    # ================= 展示预测结果 =================
    if st.session_state["pred_df"] is not None:
        df_pred = st.session_state["pred_df"]

        st.subheader("📋 客户预测结果一览")
        st.dataframe(
            df_pred[["customer_id", "prob", "pred_label", "rule_advice"] + feature_cols].head(50),
            use_container_width=True,
        )

        # ========= 单客户详情 =========
        st.subheader("🔍 单客户详情与跟进建议")

        customer_ids = df_pred["customer_id"].tolist()
        selected_id = st.selectbox("选择一个客户ID查看详情", options=customer_ids)

        selected_row = df_pred[df_pred["customer_id"] == selected_id].iloc[0]
        selected_prob = float(selected_row["prob"])
        selected_rule_advice = selected_row["rule_advice"]

        st.markdown("**基础信息与模型结果**")
        info_cols = ["customer_id"] + [c for c in feature_cols if c in selected_row.index]
        st.table(selected_row[info_cols].to_frame("值"))

        st.markdown(
            f"**模型预测成交概率：** `{selected_prob:.2%}`  \n"
            f"**规则引擎建议：** {selected_rule_advice}"
        )

        # ========= LLM 建议 按钮 =========
        if st.button("生成文字版跟进建议（DeepSeek）"):
            api_key = st.session_state.get("api_key", "")

            if api_key:
                with st.spinner("正在调用 DeepSeek 生成建议..."):
                    advice = generate_advice_with_llm(
                        selected_row,
                        selected_prob,
                        selected_rule_advice,
                        api_key=api_key,
                    )
            else:
                st.info("未输入 DeepSeek API Key，已使用模板版建议（非大模型）生成结果。")
                advice = generate_advice_template(
                    selected_row,
                    selected_prob,
                    selected_rule_advice,
                )

            st.markdown("### 🧠 建议文本")
            st.write(advice)

    # ================= 模型准确性评估 =================
    st.divider()
    st.header("📈 模型准确性评估")
    st.markdown(
        "上传或选用带 `label` (0/1) 的数据集，计算准确率、召回率、F1、ROC AUC 等指标，"
        "便于监控模型效果。"
    )

    eval_source = st.radio(
        "选择评估数据来源",
        ("使用本地训练集 (data/crm_training_data.csv)", "上传自定义带 label 的 CSV"),
        key="eval_source",
    )

    eval_df = None
    if eval_source.startswith("使用本地训练集"):
        if os.path.exists(TRAINING_DATA_PATH):
            try:
                eval_df = load_labeled_data(TRAINING_DATA_PATH)
                st.caption(f"已载入: {TRAINING_DATA_PATH}")
            except Exception as e:
                st.error(f"加载本地训练集失败：{e}")
        else:
            st.error("未找到 data/crm_training_data.csv，请先准备带 label 的评估数据。")
    else:
        eval_file = st.file_uploader("上传带 label 的 CSV (须包含 customer_id, label 列)", type=["csv"])
        if eval_file is not None:
            eval_df = pd.read_csv(eval_file)
            missing = [c for c in ("customer_id", "label") if c not in eval_df.columns]
            if missing:
                st.error(f"缺少必需列: {', '.join(missing)}")
                eval_df = None

    if st.button("运行模型评估", key="run_eval"):
        if eval_df is None:
            st.warning("请先选择有效的评估数据。")
        else:
            with st.spinner("正在计算评估指标..."):
                try:
                    metrics = evaluate_on_dataframe(model, eval_df)
                except Exception as e:
                    st.error(f"评估失败：{e}")
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("准确率", f"{metrics['accuracy']:.2%}")
                    col2.metric("精确率", f"{metrics['precision']:.2%}")
                    col3.metric("召回率", f"{metrics['recall']:.2%}")
                    auc_value = metrics.get("roc_auc")
                    if auc_value is not None:
                        col4.metric("ROC AUC", f"{auc_value:.3f}")
                    else:
                        col4.metric("ROC AUC", "N/A")

                    st.metric("F1 分数", f"{metrics['f1']:.2%}")

                    st.markdown("**混淆矩阵 (行=真实值, 列=预测值)**")
                    cm = metrics["confusion_matrix"]
                    cm_df = pd.DataFrame(
                        cm,
                        columns=["预测:负类(0)", "预测:正类(1)"],
                        index=["真实:负类(0)", "真实:正类(1)"],
                    )
                    st.dataframe(cm_df)

                    st.markdown("**分类报告**")
                    st.text(metrics["classification_report"])

if __name__ == "__main__":
    main()
