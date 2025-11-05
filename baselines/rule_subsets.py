# -*- coding: utf-8 -*-
"""
baselines/rule_subsets.py
直接运行版（免命令行参数）：
- 自动查找 train_5/50/100/200.csv
- 用与 pred_test 同样的规则法跑预测
- 输出 experiments/rule_based/pred_train_{size}.csv
- 同时写 per-subset 的 JSONL 日志到 reports/baselines/rule_based_log_train_{size}.jsonl
"""
from __future__ import annotations
import os, sys, json, glob
from typing import List, Dict, Any
import pandas as pd

# ==== 设定项目根并修复 import 路径 ====
CUR  = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(CUR, ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rules.tool_selector import select_tool
from rules.arg_filler import normalize_tools_field, fill_arguments

# ==== 目标子集规模 ====
SIZES = [5, 50, 100, 200]

# ==== 候选的 splits 目录（包含你们各自机器上的绝对路径与仓库内默认路径）====
CANDIDATE_SPLIT_DIRS = [
    # 仓库内默认
    os.path.join(REPO, "data", "splits"),
    os.path.join(REPO, "splits"),
    # 你们各自机器上的绝对路径（来自你贴的脚本）
    "/Users/yioha_/Desktop/Small-Data/splits",
    "/Users/rundongguo/Desktop/cogs 108/Untitled/Small-Data/splits",
    "/Users/iriswu/Desktop/3001 Small Data/Small-Data/splits",
    "/Users/yinghanding/Desktop/Small-Data/data/splits",
]

OUT_EXP_DIR = os.path.join(REPO, "experiments", "rule_based")
OUT_RPT_DIR = os.path.join(REPO, "reports", "baselines")

def ensure_dir(path: str):
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _autofind_subset(size: int) -> str | None:
    """
    优先使用仓库默认 data/splits/train_{size}.csv；
    若不存在，则在 CANDIDATE_SPLIT_DIRS 里查找；
    再不行，用全仓库递归搜 'train_{size}.csv'
    """
    default = os.path.join(REPO, "data", "splits", f"train_{size}.csv")
    if os.path.exists(default):
        return default
    # 指定候选目录查找
    for d in CANDIDATE_SPLIT_DIRS:
        cand = os.path.join(d, f"train_{size}.csv")
        if os.path.exists(cand):
            return cand
    # 最后递归全仓库搜索
    cands = glob.glob(os.path.join(REPO, "**", f"train_{size}.csv"), recursive=True)
    if cands:
        cands.sort(key=len)
        return cands[0]
    return None

def _read_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 补 id
    if "id" not in df.columns:
        df.insert(0, "id", range(len(df)))
    # 必需列检查
    need = {"query", "tools", "gold_call"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{os.path.basename(path)} 缺少必要列：{miss}")
    return df

def _predict_one_split(split_path: str, out_csv: str, out_log: str, lambda_arg: float = 0.2):
    print(f"📥 读取：{split_path}")
    df = _read_split(split_path)

    ensure_dir(out_csv)
    ensure_dir(out_log)

    rows = []
    with open(out_log, "w", encoding="utf-8") as flog:
        for _, r in df.iterrows():
            q = str(r["query"])
            tools_raw = r["tools"]
            # 解析 tools
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

            # 选工具 + 填参（与 pred_test 完全同逻辑）
            pred_tool, tool_obj, sel_dbg = select_tool(q, tools, lambda_arg=lambda_arg)
            if tool_obj:
                pred_args, fill_dbg = fill_arguments(q, tool_obj)
            else:
                pred_args, fill_dbg = {}, {"per_arg": []}

            # 写日志（便于误差分析）
            flog.write(json.dumps({
                "id": r["id"], "query": q, "chosen_tool": pred_tool,
                "select_debug": sel_dbg, "fill_debug": fill_dbg
            }, ensure_ascii=False) + "\n")

            # 输出一行
            rows.append({
                "id": r["id"],
                "query": q,
                "tools": tools_raw,
                "pred_tool": pred_tool,
                "pred_args": json.dumps(pred_args, ensure_ascii=False),
                "raw_output": json.dumps({"rule_log": {"select": sel_dbg, "fill": fill_dbg}}, ensure_ascii=False)
            })

    out_df = pd.DataFrame(rows, columns=["id", "query", "tools", "pred_tool", "pred_args", "raw_output"])
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ 写出：{out_csv}")
    print(f"🪵 日志：{out_log}")

def main():
    print("🪶 项目根：", REPO)
    ensure_dir(OUT_EXP_DIR)
    ensure_dir(OUT_RPT_DIR)

    missing = []
    for size in SIZES:
        split_path = _autofind_subset(size)
        if not split_path:
            print(f"⚠️ 未找到 train_{size}.csv，将跳过该子集。")
            missing.append(size)
            continue
        out_csv = os.path.join(OUT_EXP_DIR, f"pred_train_{size}.csv")
        out_log = os.path.join(OUT_RPT_DIR, f"rule_based_log_train_{size}.jsonl")
        _predict_one_split(split_path, out_csv, out_log, lambda_arg=0.2)

    if missing:
        print("⚠️ 下列规模未生成，因为没找到对应 CSV：", missing)
    else:
        print("🎉 所有目标子集（5/50/100/200）均已完成。")

if __name__ == "__main__":
    main()
