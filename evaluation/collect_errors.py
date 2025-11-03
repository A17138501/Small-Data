import os
import json
import ast
import pandas as pd

# 路径（从项目根目录执行：python evaluation/collect_errors_v5.py）
REPORT_DIR = "reports/baselines"
PRED_DIR   = "experiments/zero_shot/openai_gpt-4o-mini"
SPLIT_DIR  = "splits"
OUT_CSV    = os.path.join(PRED_DIR, "error_samples.csv")

SPLITS = [
    ("train_5",   "pred_train_5.csv",   "train_5.csv",   "eval_train_5.json"),
    ("train_50",  "pred_train_50.csv",  "train_50.csv",  "eval_train_50.json"),
    ("train_100", "pred_train_100.csv", "train_100.csv", "eval_train_100.json"),
    ("train_200", "pred_train_200.csv", "train_200.csv", "eval_train_200.json"),
    # 如需 test：("test", "pred_test.csv", "test.csv", "eval_test.json"),
]

def safe_parse(x):
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return None
    return None

def normalize_gold(df_gold):
    """
    返回只含三列的 DataFrame：['id','tools','gold_call']。
    - tools：原样保留（字符串或JSON）供下游使用
    - gold_call：若不存在则从 (gold_tool, gold_args) 或 (tool, arguments) 构造
    """
    cols = set(df_gold.columns)
    # tools 原样留存（没有就补空）
    tools_series = df_gold["tools"] if "tools" in cols else pd.Series([None]*len(df_gold))

    # 1) 已有 gold_call
    if "gold_call" in cols:
        return pd.DataFrame({
            "id": df_gold["id"],
            "tools": tools_series,
            "gold_call": df_gold["gold_call"]
        })

    # 2) gold_tool / gold_args
    if ("gold_tool" in cols) or ("gold_args" in cols):
        gold_calls = []
        for _, r in df_gold.iterrows():
            tool = r["gold_tool"] if "gold_tool" in cols else None
            args = r["gold_args"] if "gold_args" in cols else None
            args_obj = safe_parse(args)
            gold_calls.append(json.dumps({"tool": tool, "arguments": args_obj}, sort_keys=True))
        return pd.DataFrame({"id": df_gold["id"], "tools": tools_series, "gold_call": gold_calls})

    # 3) tool / arguments
    if ("tool" in cols) or ("arguments" in cols):
        gold_calls = []
        for _, r in df_gold.iterrows():
            tool = r["tool"] if "tool" in cols else None
            args = r["arguments"] if "arguments" in cols else None
            args_obj = safe_parse(args)
            gold_calls.append(json.dumps({"tool": tool, "arguments": args_obj}, sort_keys=True))
        return pd.DataFrame({"id": df_gold["id"], "tools": tools_series, "gold_call": gold_calls})

    # 4) 兜底
    return pd.DataFrame({"id": df_gold["id"], "tools": tools_series, "gold_call": [None]*len(df_gold)})

all_parts = []

for split_name, pred_file, gold_file, eval_file in SPLITS:
    pred_path = os.path.join(PRED_DIR, pred_file)
    gold_path = os.path.join(SPLIT_DIR, gold_file)
    eval_path = os.path.join(REPORT_DIR, eval_file)

    if not (os.path.exists(pred_path) and os.path.exists(gold_path) and os.path.exists(eval_path)):
        continue

    # 读取 per-tool 统计，筛出 “joint < 1.0” 的工具（可疑工具）
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    per_tool = eval_data.get("per_tool", {})  # 你的 JSON 用的是 per_tool
    rows = []
    for tool_name, stats in per_tool.items():
        rows.append({
            "pred_tool": tool_name,
            "tool_n": stats.get("n"),
            "tool_tool_acc": stats.get("tool_acc"),
            "tool_arg_em": stats.get("arg_em"),
            "tool_exec_success": stats.get("exec_success"),
            "tool_joint": stats.get("joint"),
        })
    df_tool_stats = pd.DataFrame(rows)
    # 只保留“这个 split 内出现过错误”的工具
    df_bad_tools = df_tool_stats[df_tool_stats["tool_joint"] < 1.0].copy()
    if df_bad_tools.empty:
        continue

    # 读取预测（pred_*）
    df_pred = pd.read_csv(pred_path)
    for col in ["id","query","pred_tool","pred_args","raw_output"]:
        if col not in df_pred.columns:
            df_pred[col] = None

    # 读取 gold（splits/*），取出 tools & gold_call
    df_gold_raw = pd.read_csv(gold_path)
    df_gold = normalize_gold(df_gold_raw)

    # 先保留“可疑工具”的所有预测行
    df_pred_bad = df_pred.merge(df_bad_tools[["pred_tool"]], on="pred_tool", how="inner")

    # 并上 per-tool 统计（聚合指标）
    df_pred_bad = df_pred_bad.merge(df_bad_tools, on="pred_tool", how="left")

    # 并上 gold 的 tools / gold_call
    df_pred_bad = df_pred_bad.merge(df_gold, on="id", how="left")

    # 标注 split
    df_pred_bad["split_name"] = split_name

    # 输出列顺序
    final_cols = [
        "split_name", "id", "query",
        "pred_tool", "pred_args", "raw_output",
        "tool_n", "tool_tool_acc", "tool_arg_em", "tool_exec_success", "tool_joint",
        "tools", "gold_call"
    ]
    for c in final_cols:
        if c not in df_pred_bad.columns:
            df_pred_bad[c] = None

    all_parts.append(df_pred_bad[final_cols])

# 合并并导出
if all_parts:
    df_out = pd.concat(all_parts, ignore_index=True)
else:
    df_out = pd.DataFrame(columns=[
        "split_name","id","query","pred_tool","pred_args","raw_output",
        "tool_n","tool_tool_acc","tool_arg_em","tool_exec_success","tool_joint",
        "tools","gold_call"
    ])

os.makedirs(REPORT_DIR, exist_ok=True)
df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"wrote {len(df_out)} rows to {OUT_CSV}")
