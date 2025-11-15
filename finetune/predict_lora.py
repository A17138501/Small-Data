#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, os, re, sys
from typing import Dict, Any, Optional, Tuple

# ---------- 可选依赖 ----------
try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
except Exception:
    mlx_load = None
    mlx_generate = None

try:
    import requests
except Exception:
    requests = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None


# ---------- Prompt ----------
SYSTEM_PROMPT = (
    "You are a helpful tool-calling assistant.\n"
    "Rules:\n"
    "1) Reply with JSON ONLY, no extra text.\n"
    '2) Format: {"tool": "<name>", "arguments": { ... }}\n'
    "3) Arguments must be valid JSON.\n"
    "4) If output is not valid JSON, it will be graded as failure."
)

USER_TEMPLATE = (
    "User Query:\n{query}\n\n"
    "Available Tools (JSON schema or names):\n{tools}\n\n"
    "Return ONLY the JSON tool call."
)

def build_prompt(query: str, tools: str) -> str:
    return (
        f"<SYSTEM>\n{SYSTEM_PROMPT}\n</SYSTEM>\n"
        f"<USER>\n{USER_TEMPLATE.format(query=query, tools=tools)}\n</USER>\n"
        f"<ASSISTANT>\n"
    )


# ---------- JSON 提取 ----------
def _strip_fences(text: str) -> str:
    """去掉 ```json 包裹."""
    if not text:
        return text
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
        t = re.sub(r"^\w+\n", "", t, count=1)
    return t

def _find_first_json_object(text: str) -> Optional[str]:
    """从文本中找到第一个完整 JSON { ... }."""
    if not text:
        return None
    t = _strip_fences(text)
    start = t.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return t[start:i+1]
    return None

def extract_json_tool(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """抽取 {"tool":..., "arguments":...}，失败返回 (None, None)."""
    if not raw:
        return None, None

    candidates = [raw, _find_first_json_object(raw)]
    for c in candidates:
        if not c:
            continue
        try:
            obj = json.loads(c)
        except Exception:
            continue

        tool = obj.get("tool", None)
        if tool is None:
            continue
        args = obj.get("arguments", None)
        args_str = json.dumps(args, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return str(tool), args_str

    return None, None


# ---------- 后端 1：Ollama ----------
def call_ollama(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    if requests is None:
        raise RuntimeError("requests not installed")

    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    url = "http://127.0.0.1:11434/api/generate"
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


# ---------- 后端 2：MLX + LoRA ----------
def call_mlx(model: str, adapter: Optional[str], prompt: str, temperature: float, max_tokens: int, tokenizer) -> str:
    # if mlx_load is None:
        # raise RuntimeError("mlx_lm not installed")
    # print("loading")
    # print("loaded!")
    # print("===============")
    # print(prompt)
    out_tokens = []
    # print("generating...")
    for chunk in mlx_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        # temperature=temperature,
        max_tokens=max_tokens,
        # stream=True,
    ):
        out_tokens.append(chunk)
    # print("done")
    return "".join(out_tokens)


# ---------- 后端 3：HuggingFace ----------
def call_hf(model_name: str, prompt: str, temperature: float, max_tokens: int, adapter: Optional[str]):
    if AutoTokenizer is None:
        raise RuntimeError("transformers not installed")

    dtype = None
    if torch and torch.cuda.is_available():
        dtype = torch.float16
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")

    inputs = tok(prompt, return_tensors="pt")
    if torch and torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    gen_ids = mdl.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        pad_token_id=tok.eos_token_id,
    )
    out = tok.decode(gen_ids[0], skip_special_tokens=True)
    if out.startswith(prompt):
        out = out[len(prompt):]
    return out.strip()


# ---------- 主流程 ----------
def main():
    from tqdm import tqdm
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="CSV with columns: id,query,tools")
    ap.add_argument("--out_csv", required=True)

    # 三模式
    ap.add_argument("--use_ollama", default=None)
    ap.add_argument("--mlx_model", default=None)
    ap.add_argument("--mlx_adapter", default=None)
    ap.add_argument("--hf_model", default=None)
    ap.add_argument("--hf_adapter", default=None)

    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=128)
    args = ap.parse_args()

    # 检查后端三选一
    backends = int(bool(args.use_ollama)) + int(bool(args.mlx_model)) + int(bool(args.hf_model))
    if backends != 1:
        print("❌ Error: choose exactly ONE backend (--use_ollama / --mlx_model / --hf_model)")
        sys.exit(1)

    # 加载 split
    rows = []
    with open(args.split, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({"id": r["id"], "query": r["query"], "tools": r["tools"]})
            if len(rows) == 100:
                break

    # 输出目录
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(
            wf,
            fieldnames=["id", "query", "tools", "pred_tool", "pred_args", "raw_output"]
        )
        writer.writeheader()

        model_obj, tokenizer = mlx_load(args.mlx_model, adapter_path=args.mlx_adapter)
        # model_obj, tokenizer = mlx_load(args.mlx_model, adapter_path=None)



        for r in tqdm(rows, desc="generating"):
            prompt = build_prompt(r["query"], r["tools"])
            try:
                if args.use_ollama:
                    raw = call_ollama(args.use_ollama, prompt, args.temperature, args.max_tokens)
                elif args.mlx_model:
                    # print("call mlx")
                    # raw = call_mlx(model_obj, args.mlx_adapter, prompt, args.temperature, args.max_tokens, tokenizer)
                    raw = call_mlx(model_obj, None, prompt, args.temperature, args.max_tokens, tokenizer)

                    print(raw)
                else:
                    raw = call_hf(args.hf_model, prompt, args.temperature, args.max_tokens, args.hf_adapter)
            except Exception as e:
                raise e
                raw = f"__ERROR__: {type(e).__name__}: {e}"

            tool, args_json = extract_json_tool(raw)
            writer.writerow({
                "id": r["id"],
                "query": r["query"],
                "tools": r["tools"],
                "pred_tool": tool or "",
                "pred_args": args_json or "",
                "raw_output": raw,
            })


if __name__ == "__main__":
    main()