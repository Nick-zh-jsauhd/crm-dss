# src/rules.py
from typing import Optional
import pandas as pd


def classify_customer(row: pd.Series, prob: float) -> str:
    """
    根据模型预测概率 + 一些业务字段，输出简短规则建议。
    你可以在训练数据中加入：
    - segment: 客户分层（高价值/普通等）
    - days_since_last_contact: 距离最近一次联系的天数
    如果没有这些字段，会走降级逻辑。
    """
    segment: str = row.get("segment", "普通客户")
    recency: Optional[float] = row.get("days_since_last_contact", None)

    # 优先用概率划分档位
    if prob >= 0.8:
        base = "极高优先级客户"
    elif prob >= 0.6:
        base = "高优先级客户"
    elif prob >= 0.4:
        base = "中等优先级客户"
    else:
        base = "低优先级客户"

    # 叠加补充说明
    notes = []

    if segment == "高价值":
        notes.append("高价值客户，适合重点维护")

    if recency is not None:
        if recency > 30:
            notes.append("长期未联系，存在流失或机会丢失风险")
        elif recency > 14:
            notes.append("近期未连续跟进，可安排一次主动回访")
        else:
            notes.append("近期有跟进，可继续推进")

    if prob < 0.4:
        notes.append("可考虑使用自动化触达（短信/邮件等）")

    detail = "；".join(notes) if notes else "根据销售资源情况安排跟进"
    return f"{base}：{detail}"
