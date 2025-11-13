# project/rules/tool_selector.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math, re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

def _tok(s: str) -> List[str]:
    if not s:
        return []
    return [w.lower() for w in _WORD_RE.findall(s)]

def _idf(docfreq: Dict[str, int], N: int) -> Dict[str, float]:
    return {t: math.log((N + 1.0) / (df + 1.0)) + 1.0 for t, df in docfreq.items()}

def _bm25_like(query_toks: List[str], docs: List[List[str]], k1: float = 1.2, b: float = 0.75) -> List[float]:
    if not docs:
        return []
    avgdl = sum(len(d) for d in docs) / max(1, len(docs))
    df = defaultdict(int)
    for d in docs:
        for t in set(d):
            df[t] += 1
    idf = _idf(df, len(docs))
    qtf = Counter(query_toks)
    scores = []
    for d in docs:
        tf = Counter(d)
        dl = len(d)
        sc = 0.0
        for t in qtf.keys():
            if t not in idf or tf[t] == 0:
                continue
            denom = tf[t] + k1 * (1 - b + b * (dl / (avgdl + 1e-9)))
            sc += idf[t] * ((tf[t] * (k1 + 1)) / (denom + 1e-9))
        scores.append(sc)
    return scores

def _flatten_tool_text(tool: Dict) -> Tuple[str, str, List[str]]:
    name = tool.get("name") or tool.get("tool_name") or ""
    desc = tool.get("description") or tool.get("desc") or ""
    arg_defs = tool.get("arguments") or tool.get("args") or []
    arg_names = []
    for a in arg_defs:
        nm = a.get("name") or a.get("arg") or ""
        if nm:
            arg_names.append(str(nm))
    return name, desc, arg_names

def select_tool(query: str, tools: List[Dict], lambda_arg: float = 0.2) -> Tuple[str, Dict, Dict]:
    """BM25(name+desc+arg_names) + λ*参数名与query重叠 进行工具选择"""
    q_tokens = _tok(query)
    if not tools:
        return "", {}, {"reason": "empty_tools"}

    docs, arg_name_sets, tool_names = [], [], []
    for t in tools:
        name, desc, arg_names = _flatten_tool_text(t)
        tool_names.append(name)
        docs.append(_tok(" ".join([name, desc] + arg_names)))
        arg_name_sets.append(set(s.lower() for s in arg_names))

    bm25_scores = _bm25_like(q_tokens, docs)
    qset = set(q_tokens)
    overlap_scores = [len(qset & s) for s in arg_name_sets]

    totals, details = [], []
    for i, nm in enumerate(tool_names):
        sc = bm25_scores[i] + lambda_arg * overlap_scores[i]
        totals.append(sc)
        details.append({
            "tool": nm, "bm25": bm25_scores[i],
            "arg_overlap": overlap_scores[i], "lambda": lambda_arg, "total": sc
        })

    best_idx = max(range(len(totals)), key=lambda i: totals[i]) if totals else 0
    return tool_names[best_idx], tools[best_idx], {"candidates": details, "chosen": details[best_idx]}
