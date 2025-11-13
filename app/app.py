# app/app.py
import os
import sys

import pandas as pd
import streamlit as st
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

from rules import classify_customer
from llm_agent import (
    generate_advice_template,
    generate_advice_with_llm,
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "buy_model.pkl")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("模型文件不存在，请先在项目根目录运行：python -m src.train_model")
        st.stop()
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(page_title="CRM 决策支持系统 - Yannick", layout="wide")
    st.title("📊 CRM 决策支持系统")
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

    # ============= 数据上传 & 预测 =============
    st.sidebar.header("数据与预测")
    uploaded_file = st.sidebar.file_uploader("上传客户特征数据（CSV）", type=["csv"])

    if uploaded_file is None:
        st.info("请在左侧上传待预测的客户数据（至少包含 customer_id 以及若干特征列）。")
        return

    df = pd.read_csv(uploaded_file)
    if "customer_id" not in df.columns:
        st.error("数据中必须包含 'customer_id' 列。")
        return

    feature_cols = [c for c in df.columns if c != "customer_id"]
    X = df[feature_cols]

    if "pred_df" not in st.session_state:
        st.session_state["pred_df"] = None

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


if __name__ == "__main__":
    main()
