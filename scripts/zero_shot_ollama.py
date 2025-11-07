"""
Zero-shot baseline via Ollama -> unified CSV
Usage:
  python scripts/zero_shot_ollama.py \
    --model llama3.2:1b-instruct \
    --split splits/train_50.csv \
    --out experiments/base/llama3.2-1b/pred_train_50.csv
"""

import csv, json, argparse, time, sys
import requests
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# TEMPLATE = """You are a function-calling model. You will be given a user query and a list of available tools.
# Your job: choose exactly ONE tool and return ONLY a strict JSON object with keys "tool" and "args".
# - tools: a list of tool names (strings). Choose one of them.
# - args: a JSON object. If no arguments are needed, return {}.
# - No explanation, no code block, no extra text.
# Example output:
# {"tool":"search","args":{"q":"NYU"}} 

# TOOLS:
# {tools}

# QUERY:
# {query}
# """

TEMPLATE = """You are a function-calling model. You will be given a user query and a list of available tools.
Your job: choose exactly ONE tool and return ONLY a strict JSON object with keys "tool" and "args".
- tools: a list of tool names (strings). Choose one of them.
- args: a JSON object. If no arguments are needed, return {{}}.
- No explanation, no code block, no extra text.
Example output:
{{"tool":"search","args":{{"q":"NYU"}}}}

TOOLS:
{tools}

QUERY:
{query}
"""



def call_ollama(model: str, prompt: str, max_retries=3, timeout=60):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
        except Exception as e:
            if attempt == max_retries:
                return f"[ERROR] {e}"
            time.sleep(1.5 * attempt)
    return ""

def parse_json_obj(text: str):
    """
    从模型原文里尽量抠出一个最外层 JSON 对象 {"tool":..., "args":...}
    失败则返回 (None, None)
    """
    # 尝试直接解析
    try:
        obj = json.loads(text)
        tool = obj.get("tool")
        args = obj.get("args", {})
        if isinstance(tool, str) and isinstance(args, dict):
            return tool, json.dumps(args, ensure_ascii=False)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start:end+1])
            tool = obj.get("tool")
            args = obj.get("args", {})
            if isinstance(tool, str) and isinstance(args, dict):
                return tool, json.dumps(args, ensure_ascii=False)
        except Exception:
            return None, None
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:1b-instruct")
    ap.add_argument("--split", required=True, help="path to split csv (e.g., splits/train_50.csv)")
    ap.add_argument("--out", required=True, help="path to output csv")
    args = ap.parse_args()

    in_path = Path(args.split)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 读入 split；尽量兼容列名
    with in_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_in = [c.strip() for c in reader.fieldnames or []]

        # 猜测列名
        # 必选：id、query（可能叫 question / prompt）
        id_col = "id" if "id" in fieldnames_in else fieldnames_in[0]
        query_col = "query" if "query" in fieldnames_in else ("question" if "question" in fieldnames_in else ("prompt" if "prompt" in fieldnames_in else fieldnames_in[1]))
        tools_col = "tools" if "tools" in fieldnames_in else None  # 没有也行

        rows = list(reader)

    # 写统一输出
    out_cols = ["id", "query", "tools", "pred_tool", "pred_args", "raw_output"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()

        for i, r in enumerate(rows, 1):
            ex_id = r.get(id_col, str(i))
            query = r.get(query_col, "").strip()
            tools_list = r.get(tools_col, "").strip() if tools_col else ""

            prompt = TEMPLATE.format(
                tools=tools_list if tools_list else "[]",
                query=query
            )
            resp = call_ollama(args.model, prompt)
            tool, pargs = parse_json_obj(resp)

            writer.writerow({
                "id": ex_id,
                "query": query,
                "tools": tools_list,
                "pred_tool": tool or "",
                "pred_args": pargs or "",
                "raw_output": resp,
            })

            if i % 10 == 0:
                print(f"[{i}/{len(rows)}] done", file=sys.stderr)

    print(f"✅ wrote {out_path}  | rows={len(rows)}")

if __name__ == "__main__":
    main()
