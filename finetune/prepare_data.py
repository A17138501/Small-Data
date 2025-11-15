# -*- coding: utf-8 -*-
"""
鲁棒版 SFT 数据准备（可直接运行；支持单文件与批处理）：
- 兼容 gold_call 的多形态（JSON/双重JSON/单引号Python字面量、dict/list变体）
- 兼容字段别名：tool|tool_name|name|function.name；arguments|args|params|parameters
- 兼容 arguments 为字符串再解一次
- 自动抽查 20 条，确保 assistant JSON 可被 json.loads()
- 输出 bad rows CSV 便于排查
- 批处理会自动查找 train_{5,50,100,200}.csv → data/sft/train_*.jsonl，并生成 data/sft/stats.md

用法：
- 批处理（推荐）：python -m fineture.prepare_data
- 单文件：python -m fineture.prepare_data --split splits/train_50.csv --out data/sft/train_50.jsonl
"""
from __future__ import annotations
import os, sys, json, glob, argparse, random, ast
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
from collections import Counter

# ---- 项目根 & 默认路径 ----
_CUR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_CUR, ".."))

DEFAULT_SPLIT_DIRS = [
    os.path.join(_REPO, "data", "splits"),
    os.path.join(_REPO, "splits"),
    "/Users/yioha_/Desktop/Small-Data/splits",
    "/Users/rundongguo/Desktop/cogs 108/Untitled/Small-Data/splits",
    "/Users/iriswu/Desktop/3001 Small Data/Small-Data/splits",
    "/Users/yinghanding/Desktop/Small-Data/data/splits",
]
DEFAULT_SFT_DIR = os.path.join(_REPO, "data", "sft")
SIZES = [5, 50, 100, 200]

SYSTEM_PROMPT = "You are a function-calling planner. Carefully choose the correct tool and return a strict JSON object with fields: tool, arguments."
USER_TEMPLATE = (
    "Query: {query}\n"
    "Tools(JSON): {tools_json}\n"
    "Output a JSON with:\n"
    "{{\"tool\":\"<name>\", \"arguments\":{{...}}}}"
)

def ensure_dir(path: str):
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _autofind_train_csv(size: int) -> Optional[str]:
    default = os.path.join(_REPO, "data", "splits", f"train_{size}.csv")
    if os.path.exists(default):
        return default
    for d in DEFAULT_SPLIT_DIRS:
        cand = os.path.join(d, f"train_{size}.csv")
        if os.path.exists(cand):
            return cand
    cands = glob.glob(os.path.join(_REPO, "**", f"train_{size}.csv"), recursive=True)
    if cands:
        cands.sort(key=len)
        return cands[0]
    return None

def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _maybe_json_loads(s: Any) -> Any:
    """尝试把字符串逐步解成对象：JSON -> 双重JSON -> Python字面量 -> 原样"""
    if not isinstance(s, str):
        return s
    txt = s.strip()
    # 1) 直接 JSON
    try:
        return json.loads(txt)
    except Exception:
        pass
    # 2) 去掉包裹引号再解（双重 JSON）
    if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
        inner = txt[1:-1]
        try:
            return json.loads(inner)
        except Exception:
            pass
    # 3) Python 字面量（单引号 dict/list）
    try:
        return ast.literal_eval(txt)
    except Exception:
        pass
    # 都不行就原样返回
    return s

def _extract_tool_args(obj: Any) -> Tuple[str, Dict[str, Any], str]:
    """
    从各种形态中提取 (tool_name, arguments_dict, shape_tag)
    支持：
      - {"tool":"...","arguments":{...}}
      - {"name":"...","arguments"/"args"/"params"/"parameters":...}
      - {"function":{"name":"...","arguments":...}}
      - [{"tool":...,"arguments":...}, ...] 取首个
    arguments 若为字符串会再尝试解析一次
    """
    if isinstance(obj, str):
        obj = _maybe_json_loads(obj)

    def _normalize_args(a: Any) -> Dict[str, Any]:
        a = _maybe_json_loads(a)
        return a if isinstance(a, dict) else {}

    if isinstance(obj, dict):
        tool = obj.get("tool") or obj.get("tool_name") or obj.get("name")
        args = obj.get("arguments") or obj.get("args") or obj.get("params") or obj.get("parameters")
        if tool:
            return str(tool), _normalize_args(args), "dict:direct"

        func = obj.get("function")
        if isinstance(func, dict):
            tool = func.get("name") or func.get("tool") or func.get("tool_name")
            args = func.get("arguments") or func.get("args") or func.get("params") or func.get("parameters")
            if tool:
                return str(tool), _normalize_args(args), "dict:function"

    if isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, (dict, str)):
            t, a, tag = _extract_tool_args(first)
            if t:
                return t, a, f"list[0]->{tag}"

    return "", {}, "unrecognized"

def _parse_gold_call(gold_call_raw: Any) -> Tuple[str, Dict[str, Any], str]:
    obj = _maybe_json_loads(gold_call_raw)
    tool, args, tag = _extract_tool_args(obj)
    return tool, args, tag

def _read_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"query", "tools", "gold_call"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{os.path.basename(path)} 缺少必要列：{miss}")
    if "id" not in df.columns:
        df.insert(0, "id", range(len(df)))
    return df

def _build_one_example(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    q = str(row.get("query", ""))
    tools_raw = row.get("tools", "")
    tool_name, arguments, shape_tag = _parse_gold_call(row.get("gold_call", ""))

    if not tool_name:
        return None, shape_tag

    if isinstance(tools_raw, str):
        tools_json_str = tools_raw
    else:
        try:
            tools_json_str = _safe_json_dumps(tools_raw)
        except Exception:
            tools_json_str = str(tools_raw)

    user_content = USER_TEMPLATE.format(query=q, tools_json=tools_json_str)
    assistant_json_str = _safe_json_dumps({"tool": tool_name, "arguments": arguments})

    sample = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_json_str},
        ],
        "metadata": {"id": str(row.get("id", "")), "tool": tool_name}
    }
    return sample, shape_tag

def convert_split_to_jsonl(split_path: str, out_jsonl: str, validate_n: int = 20) -> Dict[str, Any]:
    print(f"📥 读取：{split_path}")
    df = _read_split(split_path)

    ok_examples: List[Dict[str, Any]] = []
    bad_rows: List[Dict[str, Any]] = []
    shape_counter = Counter()

    for _, r in df.iterrows():
        sample, tag = _build_one_example(r.to_dict())
        shape_counter[tag] += 1
        if sample is None:
            bad_rows.append(r.to_dict())
        else:
            ok_examples.append(sample)

    ensure_dir(out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ex in ok_examples:
            f.write(_safe_json_dumps(ex) + "\n")

    bad_csv = os.path.join(os.path.dirname(out_jsonl), f"_bad_rows_{os.path.basename(split_path).replace('.csv','')}.csv")
    if bad_rows:
        ensure_dir(bad_csv)
        pd.DataFrame(bad_rows).to_csv(bad_csv, index=False, encoding="utf-8")

    print(f"✅ 写出：{out_jsonl}（{len(ok_examples)} 条，丢弃 {len(bad_rows)} 条）")
    print("🔎 gold_call 形态分布：", dict(shape_counter))

    if ok_examples:
        random.seed(2025)
        sample = ok_examples if len(ok_examples) <= validate_n else random.sample(ok_examples, validate_n)
        bad = 0
        for ex in sample:
            try:
                json.loads(ex["messages"][-1]["content"])
            except Exception:
                bad += 1
        if bad:
            print(f"⚠️ 抽查解析失败 {bad}/{len(sample)} 条")
        else:
            print(f"🧪 抽查通过：{len(sample)}/{len(sample)}")

    user_lens = [len(ex["messages"][1]["content"]) for ex in ok_examples]
    asst_lens = [len(ex["messages"][2]["content"]) for ex in ok_examples]
    tools_cnt = Counter([ex["metadata"]["tool"] for ex in ok_examples])
    stats = {
        "split": os.path.basename(split_path),
        "jsonl": out_jsonl,
        "n": len(ok_examples),
        "dropped": len(bad_rows),
        "shape_counter": dict(shape_counter),
        "avg_user_len": (sum(user_lens)/len(user_lens)) if user_lens else 0.0,
        "avg_asst_len": (sum(asst_lens)/len(asst_lens)) if asst_lens else 0.0,
        "top_tools": tools_cnt.most_common(10),
        "bad_csv": bad_csv if bad_rows else None,
    }
    return stats

def write_stats_md(stats_list: List[Dict[str, Any]], out_md: str):
    ensure_dir(out_md)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# SFT Data Stats\n\n")
        for st in stats_list:
            f.write(f"## {st['split']}\n\n")
            f.write(f"- Output: `{st['jsonl']}`\n")
            f.write(f"- Samples: **{st['n']}** (Dropped: {st['dropped']})\n")
            f.write(f"- Avg user len (chars): **{st['avg_user_len']:.1f}**\n")
            f.write(f"- Avg asst len (chars): **{st['avg_asst_len']:.1f}**\n")
            f.write(f"- Bad rows CSV: `{st['bad_csv']}`\n\n" if st["bad_csv"] else "- Bad rows CSV: _none_\n\n")
            f.write("### gold_call shapes\n\n")
            for k, v in (st["shape_counter"] or {}).items():
                f.write(f"- {k}: {v}\n")
            f.write("\n### Top-10 Tools\n\n")
            if st["top_tools"]:
                f.write("| Tool | Count |\n|---|---:|\n")
                for name, cnt in st["top_tools"]:
                    f.write(f"| {name} | {cnt} |\n")
                f.write("\n")
            else:
                f.write("_No tool stats (empty set)_\n\n")
    print(f"📊 统计已写出：{out_md}")

def run_batch():
    out_dir = DEFAULT_SFT_DIR
    ensure_dir(out_dir)
    stats_all: List[Dict[str, Any]] = []
    for sz in SIZES:
        split_path = _autofind_train_csv(sz)
        if not split_path:
            print(f"⚠️ 未找到 train_{sz}.csv，跳过。")
            continue
        out_jsonl = os.path.join(out_dir, f"train_{sz}.jsonl")
        st = convert_split_to_jsonl(split_path, out_jsonl, validate_n=20)
        stats_all.append(st)
    if stats_all:
        write_stats_md(stats_all, os.path.join(out_dir, "stats.md"))
    else:
        print("⚠️ 未生成任何 JSONL。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--out",   type=str, default=None)
    args = parser.parse_args()

    if args.split and args.out:
        ensure_dir(args.out)
        st = convert_split_to_jsonl(args.split, args.out, validate_n=20)
        print(json.dumps(st, ensure_ascii=False, indent=2))
    else:
        run_batch()

if __name__ == "__main__":
    main()
