# src/llm_agent.py

from typing import Optional
import textwrap

import pandas as pd


def build_text_profile(row: pd.Series, prob: float, rule_advice: str) -> str:
    """
    把结构化信息转成易读的文本画像，用于展示或喂给 LLM（DeepSeek）。
    """
    customer_id = row.get("customer_id", "未知ID")
    industry = row.get("industry", "未知行业")
    region = row.get("region", "未知地区")
    segment = row.get("segment", "普通客户")
    recency: Optional[float] = row.get("days_since_last_contact", None)

    lines = [
        f"客户ID：{customer_id}",
        f"行业：{industry}；地区：{region}；客户分层：{segment}",
        f"模型预测成交概率：{prob:.2%}",
        f"规则引擎建议：{rule_advice}",
    ]
    if recency is not None:
        lines.append(f"距离最近一次联系：{recency} 天")

    return "\n".join(lines)


def generate_advice_template(row: pd.Series, prob: float, rule_advice: str) -> str:
    """
    不接外部 LLM 时的默认文字版本（保证系统在没有 API Key / 没有依赖时也能跑）。
    """
    profile = build_text_profile(row, prob, rule_advice)
    text = f"""
    以下是系统根据历史数据和模型预测生成的客户分析与建议（非大模型，仅规则模板）：

    {profile}

    建议跟进行动：
    1. 根据客户当前优先级，合理分配销售资源：
       - 极高/高优先级：安排销售专员在 1-3 天内电话或拜访，了解需求进展，推动签单。
       - 中等优先级：可结合营销活动（试用、优惠券）进行触达，观察客户反馈。
       - 低优先级：建议通过短信、邮件等自动化方式进行维护，降低人工成本。

    2. 沟通话术方向：
       - 先回顾历史合作或沟通记录，体现对客户的了解。
       - 结合客户所在行业和地区的典型痛点，给出 1–2 个有针对性的解决方案。
       - 对高价值客户，可适当强调专属服务、长期合作规划等。

    3. 后续数据积累建议：
       - 记录本次沟通的结果（是否有明确意向、预算、时间节点），以便后续模型持续优化。
    """
    return textwrap.dedent(text).strip()


def generate_advice_with_llm(
    row: pd.Series,
    prob: float,
    rule_advice: str,
    api_key: str,
) -> str:
    """
    使用 DeepSeek API 生成建议。

    设计原则：
    - 不在代码中写死 key，由调用方（如 Streamlit app）通过参数传入 api_key；
    - 如果没有 api_key、环境未安装 openai、或调用异常，则自动回退为模板版建议。
    """
    # 1) 没有 key：直接走模板版
    if not api_key:
        return generate_advice_template(row, prob, rule_advice)

    # 2) 尝试导入 openai（DeepSeek 使用 OpenAI 兼容 SDK）
    try:
        from openai import OpenAI  # 需要 pip install openai>=1.0.0
    except ImportError:
        fallback = generate_advice_template(row, prob, rule_advice)
        extra = "\n\n[提示] 当前环境未安装 `openai` 库，已自动回退为模板版建议。"
        return fallback + extra

    profile = build_text_profile(row, prob, rule_advice)

    prompt = f"""
    你是一个 CRM 销售决策助手，请根据以下客户信息生成专业的销售跟进建议。

    【客户画像与模型结果】
    {profile}

    请输出：
    1. 客户优先级评估（简要说明原因）
    2. 建议的具体跟进行动（电话/拜访/线上沟通等）
    3. 推荐的沟通话术方向（不需要完整逐字稿，只要要点）
    4. 需要注意的风险点或潜在异议
    """
    prompt = textwrap.dedent(prompt).strip()

    # 3) 构造 DeepSeek 客户端（OpenAI 兼容接口）
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",  # 如需更换模型，可在此处修改
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful CRM decision support assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )

        content = resp.choices[0].message.content
        return content.strip()
    except Exception as e:
        # 任何异常都回退到模板建议，避免前端崩溃
        fallback = generate_advice_template(row, prob, rule_advice)
        extra = f"\n\n[提示] 调用 DeepSeek API 时出现异常：{e}，已自动回退为模板版建议。"
        return fallback + extra
