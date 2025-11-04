# project/baselines/rule_based.py
# -*- coding: utf-8 -*-
"""
直接运行版（免参数）：
- 自动定位项目根
- 自动优先使用 data/splits/test.csv；找不到就全仓库递归搜 test.csv
- 产出 experiments/rule_based/pred_test.csv 与 reports/baselines/rule_based_log.jsonl
"""
from __future__ import annotations
import os, sys, json, glob
import pandas as pd

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

SPLIT_PATH = _autofind(["test.csv"], "data/splits/test.csv")
OUT_PATH   = os.path.join(repo_root, "experiments/rule_based/pred_test.csv")
LOG_PATH   = os.path.join(repo_root, "reports/baselines/rule_based_log.jsonl")
LAMBDA_ARG = 0.2

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    print("🪶 项目根：", repo_root)
    print("📥 读取：", SPLIT_PATH)
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"找不到测试集：{SPLIT_PATH}")

    df = pd.read_csv(SPLIT_PATH)
    if "id" not in df.columns:
        df["id"] = range(len(df))

    ensure_dir(OUT_PATH)
    ensure_dir(LOG_PATH)

    rows = []
    with open(LOG_PATH, "w", encoding="utf-8") as flog:
        for _, r in df.iterrows():
            q = str(r["query"])
            tools_raw = r["tools"]

            try:
                tools = normalize_tools_field(tools_raw)
            except Exception as e:
                rows.append({
                    "id": r["id"], "query": q, "tools": tools_raw,
                    "pred_tool": "", "pred_args": json.dumps({}, ensure_ascii=False),
                    "raw_output": json.dumps({"error": f"tools_parse_error:{e}"}, ensure_ascii=False)
                })
                flog.write(json.dumps({"id": r["id"], "stage": "tools_parse_error", "err": str(e)}, ensure_ascii=False) + "\n")
                continue

            pred_tool, tool_obj, sel_dbg = select_tool(q, tools, lambda_arg=LAMBDA_ARG)
            if tool_obj:
                pred_args, fill_dbg = fill_arguments(q, tool_obj)
            else:
                pred_args, fill_dbg = {}, {"per_arg": []}

            flog.write(json.dumps({
                "id": r["id"], "query": q, "chosen_tool": pred_tool,
                "select_debug": sel_dbg, "fill_debug": fill_dbg
            }, ensure_ascii=False) + "\n")

            rows.append({
                "id": r["id"],
                "query": q,
                "tools": tools_raw,
                "pred_tool": pred_tool,
                "pred_args": json.dumps(pred_args, ensure_ascii=False),
                "raw_output": json.dumps({"rule_log": {"select": sel_dbg, "fill": fill_dbg}}, ensure_ascii=False)
            })

    out_df = pd.DataFrame(rows, columns=["id","query","tools","pred_tool","pred_args","raw_output"])
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(f"✅ 预测完成：{len(out_df)} 条")
    print("📄 写入：", OUT_PATH)
    print("🪵 日志：", LOG_PATH)

if __name__ == "__main__":
    main()
