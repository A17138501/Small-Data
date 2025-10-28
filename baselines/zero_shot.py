# baselines/zero_shot.py
import argparse, json, os, pathlib, time
import pandas as pd
from models.adapter import generate

def load_prompt_tpl(path: str) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8")

def _extract_json(text: str):
    # 从模型输出中找出第一个 {...} 的 JSON
    import re
    m = re.search(r"\{.*\}", text.strip(), re.S)
    if not m: return None
    s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        # 容错：去除尾随逗号等
        s = re.sub(r",\s*([}\]])", r"\1", s)
        try: return json.loads(s)
        except: return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="path to data/splits/test.csv")
    ap.add_argument("--prompt", default="prompts/zero_shot.txt")
    ap.add_argument("--model", required=True, help="e.g., openai:gpt-4o-mini")
    ap.add_argument("--out_dir", default="experiments/zero_shot")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_csv(args.split)
    # 约定输入列：query, tools, gold_call（三列，来自前面的标准化流程）
    # 评测器需要的预测列：pred_tool, pred_args
    out_dir = pathlib.Path(args.out_dir) / args.model.replace(":","_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "pred_test.csv"
    log_path = out_dir / "run.log"

    tpl = load_prompt_tpl(args.prompt)
    fails, rows = 0, []
    start = time.time()

    for i, r in df.iterrows():
        query = r["query"]
        tools_json = r["tools"]
        system = "You are a function-calling planner."

        prompt = tpl.replace("{{query}}", str(query)).replace("{{tools_json}}", str(tools_json))

        try:
            raw = generate(
                prompt,
                model_name=args.model,
                system=system,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            parsed = _extract_json(raw)
            if not parsed or "tool" not in parsed or "arguments" not in parsed:
                fails += 1
                pred_tool, pred_args = "", "{}"
            else:
                pred_tool = str(parsed["tool"])
                # 以字符串保存 JSON
                pred_args = json.dumps(parsed["arguments"], ensure_ascii=False)
        except Exception as e:
            fails += 1
            pred_tool, pred_args = "", "{}"
            raw = f"ERROR: {e}"

        rows.append({
            "id": i,
            "query": query,
            "tools": tools_json,           # 直接存标准化后的 JSON 字符串
            "raw_output": raw,             # 模型原文（便于排错）
            "pred_tool": pred_tool,
            "pred_args": pred_args,        # 仍保存为 JSON 字符串
        })
        cols = ["id","query","tools","raw_output","pred_tool","pred_args"]
        pd.DataFrame(rows)[cols].to_csv(out_csv, index=False, encoding="utf-8")


        if (i+1) % 10 == 0:
            print(f"[zero-shot] {i+1}/{len(df)}")

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

    dur = time.time() - start
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"model={args.model}\n")
        f.write(f"n={len(df)}  fails={fails}  fail_rate={fails/len(df):.3f}\n")
        f.write(f"seconds={dur:.1f}\n")
        f.write(f"split={args.split}\n")
        f.write(f"out={out_csv}\n")
    print(f"✅ wrote {out_csv}  | fail_rate={fails/len(df):.3f}  secs={dur:.1f}")

if __name__ == "__main__":
    main()
