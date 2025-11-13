# project/baselines/rule_based.py
# -*- coding: utf-8 -*-
"""
增强版 rule_based.py：
- 默认行为：跑 test.csv → pred_test.csv
- 扩展行为：当 RUN_ALL=True 时，会自动跑 5/50/100/200 全部子集
- 自动写入：experiments/rule_based/pred_*.csv
"""

from __future__ import annotations
import os, sys, json, glob
import pandas as pd

# ===== 是否自动跑全部子集 =====
RUN_ALL = True   # ←← 只改这里即可！True = 自动生成全部

# ==== 项目根 & import path ====
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root   = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from rules.tool_selector import select_tool
from rules.arg_filler import normalize_tools_field, fill_arguments


# ==== 自动发现文件 ====
def _autofind(basenames, default_rel):
    default = os.path.join(repo_root, default_rel)
    if os.path.exists(default):
        return default
    cands = []
    for bn in basenames:
        cands.extend(glob.glob(os.path.join(repo_root, "**", bn), recursive=True))
    if cands:
        cands.sort(key=len)
        print("🔎 自动找到候选：\n  " + "\n  ".join(cands[:5]))
        return cands[0]
    return default


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ======== 核心函数：处理一个 split =========
def run_one_split(split_path: str, out_path: str, log_path: str):

    print("\n===================================================")
    print(f"📥 读取：{split_path}")
    print(f"📄 输出：{out_path}")
    print("===================================================")

    if not os.path.exists(split_path):
        print(f"❌ 路径不存在：{split_path}")
        return

    df = pd.read_csv(split_path)
    if "id" not in df.columns:
        df["id"] = range(len(df))

    ensure_dir(out_path)
    ensure_dir(log_path)

    rows = []
    with open(log_path, "w", encoding="utf-8") as flog:
        for _, r in df.iterrows():
            q = str(r["query"])
            tools_raw = r["tools"]

            # normalize tool schema
            try:
                tools = normalize_tools_field(tools_raw)
            except Exception as e:
                rows.append({
                    "id": r["id"],
                    "query": q,
                    "tools": tools_raw,
                    "pred_tool": "",
                    "pred_args": json.dumps({}, ensure_ascii=False),
                    "raw_output": json.dumps({"error": f"tools_parse_error:{e}"}, ensure_ascii=False)
                })
                continue

            # tool select
            pred_tool, tool_obj, dbg_sel = select_tool(q, tools)

            # arg fill
            if tool_obj:
                pred_args, dbg_fill = fill_arguments(q, tool_obj)
            else:
                pred_args, dbg_fill = {}, {"per_arg": []}

            rows.append({
                "id": r["id"],
                "query": q,
                "tools": tools_raw,
                "pred_tool": pred_tool,
                "pred_args": json.dumps(pred_args, ensure_ascii=False),
                "raw_output": json.dumps(
                    {"rule_log": {"select": dbg_sel, "fill": dbg_fill}},
                    ensure_ascii=False
                )
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"✅ 完成：{len(out_df)} 条")
    print(f"📄 写入：{out_path}")


# ========= 入口：处理 test 或 全部子集 =========
def main():
    print("🪶 项目根：", repo_root)

    # ===== 默认行为：处理 test.csv =====
    test_path = _autofind(["test.csv"], "splits/test.csv")
    if not RUN_ALL:
        out_path = os.path.join(repo_root, "experiments/rule_based/pred_test.csv")
        log_path = os.path.join(repo_root, "reports/baselines/rule_based_log.jsonl")
        run_one_split(test_path, out_path, log_path)
        return

    # ===== 批处理模式 =====
    TARGETS = [
        ("splits/test.csv",       "pred_test.csv"),
        ("splits/train_5.csv",    "pred_train_5.csv"),
        ("splits/train_50.csv",   "pred_train_50.csv"),
        ("splits/train_100.csv",  "pred_train_100.csv"),
        ("splits/train_200.csv",  "pred_train_200.csv"),
    ]

    for rel_split, out_name in TARGETS:
        sp = os.path.join(repo_root, rel_split)
        op = os.path.join(repo_root, "experiments/rule_based", out_name)
        lp = os.path.join(repo_root, "reports/baselines", out_name.replace(".csv", "_log.jsonl"))

        run_one_split(sp, op, lp)

    print("\n🎉 全部分集的 rule-based 预测已生成完毕！")


if __name__ == "__main__":
    main()
