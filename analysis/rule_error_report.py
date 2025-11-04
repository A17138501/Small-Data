# project/analysis/rule_error_report.py
# -*- coding: utf-8 -*-
"""
直接运行版（免参数）：
- 自动在仓库中递归寻找 test.csv（gold）与 pred_test.csv（pred）
- 统计失败类别并抽样输出 Markdown
- 输出 reports/baselines/rule_error_samples.md
"""
from __future__ import annotations
import os, json, glob, random
from collections import Counter
from typing import Dict, Any, Tuple, List
import pandas as pd

# ==== 项目根 ====
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root   = os.path.abspath(os.path.join(current_dir, ".."))

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

GOLD_PATH = _autofind(["test.csv"], "data/splits/test.csv")
PRED_PATH = _autofind(["pred_test.csv","pred.csv"], "experiments/rule_based/pred_test.csv")
OUT_PATH  = os.path.join(repo_root, "reports/baselines/rule_error_samples.md")
SAMPLE_K  = 100

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _parse_gold_call(s: str) -> Tuple[str, Dict[str, Any]]:
    try:
        obj = json.loads(s)
        tool = obj.get("tool", "")
        args = obj.get("arguments", {}) or {}
        return tool, args
    except Exception:
        return "", {}

def _parse_pred_args(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}

def _diff_one(g_tool: str, g_args: Dict[str, Any], p_tool: str, p_args: Dict[str, Any]) -> str:
    if g_tool != p_tool:
        return "wrong_tool"
    if set(g_args.keys()) != set(p_args.keys()):
        return "arg_keys_mismatch"
    for k in g_args.keys():
        if str(g_args[k]) != str(p_args[k]):
            return "arg_value_mismatch"
    return "match"

def main():
    print("🪶 项目根：", repo_root)
    print("Gold：", GOLD_PATH)
    print("Pred：", PRED_PATH)
    print("Out ：", OUT_PATH)

    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(f"找不到 GOLD：{GOLD_PATH}")
    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError(f"找不到 PRED：{PRED_PATH}")

    ensure_dir(OUT_PATH)

    g = pd.read_csv(GOLD_PATH)
    p = pd.read_csv(PRED_PATH)
    if "id" not in g.columns:
        g["id"] = range(len(g))

    need_cols_gold = {"id","query","tools","gold_call"}
    need_cols_pred = {"id","pred_tool","pred_args"}
    missing_gold = need_cols_gold - set(g.columns)
    missing_pred = need_cols_pred - set(p.columns)
    if missing_gold:
        raise ValueError(f"gold 缺少列：{missing_gold}")
    if missing_pred:
        raise ValueError(f"pred 缺少列：{missing_pred}")

    m = g[["id","query","tools","gold_call"]].merge(
        p[["id","pred_tool","pred_args"]], on="id", how="inner", suffixes=("_gold","_pred")
    )
    if m.empty:
        raise ValueError("gold 与 pred 合并后为空，请检查 id 是否一致。")

    fails: List[Dict[str, Any]] = []
    total = 0
    for _, r in m.iterrows():
        total += 1
        g_tool, g_args = _parse_gold_call(str(r["gold_call"]))
        p_tool = str(r["pred_tool"])
        p_args = _parse_pred_args(str(r["pred_args"]))
        cat = _diff_one(g_tool, g_args, p_tool, p_args)
        if cat != "match":
            fails.append({
                "id": r["id"], "cat": cat, "query": r["query"],
                "g_tool": g_tool, "p_tool": p_tool,
                "g_args": g_args, "p_args": p_args
            })

    cnt = Counter([x["cat"] for x in fails])
    n_fail = len(fails)
    n_ok = total - n_fail
    print(f"总样本: {total} | 成功: {n_ok} | 失败: {n_fail}")

    random.seed(2025)
    sample = fails if n_fail <= SAMPLE_K else random.sample(fails, SAMPLE_K)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Rule-based 错误样本（抽样）\n\n")
        f.write(f"- 总样本：{total}\n- 匹配成功：{n_ok}\n- 失败：{n_fail}\n\n")
        if cnt:
            f.write("## 失败分布\n\n")
            for k, v in cnt.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        f.write(f"## 样本（最多 {SAMPLE_K} 条）\n\n")
        for r in sample:
            f.write(f"### id {r['id']} — {r['cat']}\n")
            f.write(f"- Query: {r['query']}\n")
            f.write(f"- Gold Tool: {r['g_tool']}\n")
            f.write(f"- Pred Tool: {r['p_tool']}\n")
            f.write(f"- Gold Args: `{json.dumps(r['g_args'], ensure_ascii=False)}`\n")
            f.write(f"- Pred Args: `{json.dumps(r['p_args'], ensure_ascii=False)}`\n\n")

    print("✅ 已写入：", OUT_PATH)

if __name__ == "__main__":
    main()
