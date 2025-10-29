# baselines/zero_shot.py
import argparse, json, pathlib, time
import pandas as pd
from models.adapter import generate

def load_prompt_tpl(path: str) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8")

def _extract_json(text: str):
    """从模型输出中提取第一个 JSON 对象，容错处理 ```json 块 和 尾逗号"""
    import re
    if text is None:
        return None
    s = str(text).strip()

    # 优先抓取 ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S | re.I)
    if m:
        cand = m.group(1)
    else:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return None
        cand = m.group(0)

    try:
        return json.loads(cand)
    except Exception:
        # 去尾逗号等小错误
        cand2 = re.sub(r",\s*([}\]])", r"\1", cand)
        try:
            return json.loads(cand2)
        except Exception:
            return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="CSV with columns: id,query,tools[,gold_call]")
    ap.add_argument("--prompt", default="prompts/zero_shot.txt")
    ap.add_argument("--model", required=True, help="e.g., openai:gpt-4o-mini")
    ap.add_argument("--out_dir", default="experiments/zero_shot")
    ap.add_argument("--out_csv", default=None, help="override output CSV path")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--strict_error", action="store_true", default=True,
                    help="Fail fast if LLM call or parsing fails (recommended).")
    ap.add_argument("--no-strict_error", dest="strict_error", action="store_false")
    args = ap.parse_args()

    # 读取并校验
    df = pd.read_csv(args.split, dtype=str)
    for col in ("id", "query", "tools"):
        if col not in df.columns:
            raise ValueError(f"split 缺少必要列: {col}")
    df["id"] = df["id"].astype(str).str.strip()

    out_dir = pathlib.Path(args.out_dir) / args.model.replace(":", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    default_out = out_dir / "pred_test.csv"
    out_path = pathlib.Path(args.out_csv) if args.out_csv else default_out
    log_path = out_dir / "run.log"

    tpl = load_prompt_tpl(args.prompt)
    fails, rows = 0, []
    start = time.time()

    for i, r in df.iterrows():
        id_val = r["id"]
        query = r["query"]
        tools_json = r["tools"]
        system = "You are a function-calling planner."

        # 渲染 prompt
        prompt = tpl.replace("{{query}}", str(query)).replace("{{tools_json}}", str(tools_json))

        try:
            raw = generate(
                prompt,
                model_name=args.model,
                system=system,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            fails += 1
            if args.strict_error:
                raise RuntimeError(f"LLM call failed on id={id_val}: {e}") from e
            raw = f"ERROR: {e}"
            parsed = None
        else:
            parsed = _extract_json(raw)

        if not parsed or "tool" not in parsed or "arguments" not in parsed:
            fails += 1
            if args.strict_error:
                short = (str(raw)[:200] + "...") if isinstance(raw, str) else str(raw)
                raise RuntimeError(f"Parse failed for id={id_val}. raw_output[:200]= {short}")
            pred_tool, pred_args = "", "{}"
        else:
            pred_tool = str(parsed["tool"]).strip()
            try:
                pred_args = json.dumps(parsed["arguments"], ensure_ascii=False)
            except Exception:
                pred_args = json.dumps({"_raw": str(parsed["arguments"])}, ensure_ascii=False)

        rows.append({
            "id": id_val,
            "query": query,
            "tools": tools_json,   # 标准化工具 JSON 字符串
            "raw_output": raw,     # 保留原文便于排错；若不需要可后续 drop
            "pred_tool": pred_tool,
            "pred_args": pred_args,
        })

        if (i + 1) % 10 == 0:
            print(f"[zero-shot] {i+1}/{len(df)}")

    # 一次性写盘
    cols = ["id", "query", "tools", "raw_output", "pred_tool", "pred_args"]
    pd.DataFrame(rows)[cols].to_csv(out_path, index=False, encoding="utf-8")

    dur = time.time() - start
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"model={args.model}\n")
        f.write(f"n={len(df)}  fails={fails}  fail_rate={fails/len(df):.3f}\n")
        f.write(f"seconds={dur:.1f}\n")
        f.write(f"split={args.split}\n")
        f.write(f"out={out_path}\n")
    print(f"✅ wrote {out_path}  | fail_rate={fails/len(df):.3f}  secs={dur:.1f}")

if __name__ == "__main__":
    main()



# # baselines/zero_shot.py — minimal: prompt -> pred file (id,pred_tool,pred_args)
# import argparse, json, pathlib, re
# import pandas as pd
# from models.adapter import generate

# def load_prompt_tpl(path: str) -> str:
#     return pathlib.Path(path).read_text(encoding="utf-8")

# def _extract_json(text: str):
#     """抓取首个 JSON 对象；容错 ```json ...``` 与尾逗号"""
#     if text is None:
#         return None
#     s = str(text).strip()
#     m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S | re.I)
#     cand = m.group(1) if m else (re.search(r"\{.*\}", s, flags=re.S).group(0) if re.search(r"\{.*\}", s, flags=re.S) else None)
#     if not cand:
#         return None
#     try:
#         return json.loads(cand)
#     except Exception:
#         cand2 = re.sub(r",\s*([}\]])", r"\1", cand)
#         try:
#             return json.loads(cand2)
#         except Exception:
#             return None

# def main():
#     ap = argparse.ArgumentParser(description="Zero-shot tool-calling -> pred CSV (id,pred_tool,pred_args)")
#     ap.add_argument("--split", required=True, help="CSV with columns: id,query,tools")
#     ap.add_argument("--prompt", required=True, help="prompt template path (uses {{query}} and {{tools_json}})")
#     ap.add_argument("--model", required=True, help="e.g., openai:gpt-4o-mini")
#     ap.add_argument("--out_csv", required=True, help="output CSV path for predictions")
#     ap.add_argument("--max_new_tokens", type=int, default=256)
#     ap.add_argument("--temperature", type=float, default=0.0)
#     args = ap.parse_args()

#     # 读入并校验
#     df = pd.read_csv(args.split, dtype=str)
#     for col in ("id", "query", "tools"):
#         if col not in df.columns:
#             raise ValueError(f"split 缺少必要列: {col}")
#     df["id"] = df["id"].astype(str).str.strip()

#     tpl = load_prompt_tpl(args.prompt)
#     out_rows = []

#     for _, r in df.iterrows():
#         id_val = r["id"]
#         query = r["query"]
#         tools_json = r["tools"]

#         prompt = tpl.replace("{{query}}", str(query)).replace("{{tools_json}}", str(tools_json))

#         # LLM 调用
#         raw = generate(
#             prompt,
#             model_name=args.model,
#             system="You are a function-calling planner.",
#             temperature=args.temperature,
#             max_new_tokens=args.max_new_tokens,
#         )

#         # 解析为 {"tool": "...", "arguments": {...}}
#         parsed = _extract_json(raw)
#         if parsed and isinstance(parsed, dict) and "tool" in parsed and "arguments" in parsed:
#             pred_tool = str(parsed["tool"]).strip()
#             try:
#                 pred_args = json.dumps(parsed["arguments"], ensure_ascii=False)
#             except Exception:
#                 pred_args = json.dumps({"_raw": str(parsed["arguments"])}, ensure_ascii=False)
#         else:
#             # 解析失败时写空（不中断）
#             pred_tool, pred_args = "", "{}"

#         out_rows.append({"id": id_val, "pred_tool": pred_tool, "pred_args": pred_args})

#     out_p = pathlib.Path(args.out_csv)
#     out_p.parent.mkdir(parents=True, exist_ok=True)
#     pd.DataFrame(out_rows, columns=["id","pred_tool","pred_args"]).to_csv(out_p, index=False, encoding="utf-8")
#     print(f"✅ wrote {out_p}")

# if __name__ == "__main__":
#     main()
