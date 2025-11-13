# -*- coding: utf-8 -*-
from __future__ import annotations
import json, re
from typing import Dict, Any, List, Tuple

_NUM_RE      = re.compile(r"[-+]?\d+(?:\.\d+)?")
_INT_RE      = re.compile(r"[-+]?\d+")
_BOOL_TRUE   = {"true","yes","y","1","on"}
_BOOL_FALSE  = {"false","no","n","0","off"}
_DATE_RE     = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_WORD_RE     = re.compile(r"[A-Za-z0-9_]+")

def normalize_tools_field(tools_field: str | list | dict) -> List[Dict[str, Any]]:
    """接受 JSON 字符串、list、或 {'tools':[...]}，统一返回 list[dict]"""
    if isinstance(tools_field, str):
        tools = json.loads(tools_field)
    else:
        tools = tools_field
    if isinstance(tools, dict) and "tools" in tools:
        tools = tools["tools"]
    if not isinstance(tools, list):
        tools = [tools]
    return tools

def _tok(s: str) -> List[str]:
    if not s:
        return []
    return [w.lower() for w in _WORD_RE.findall(s)]

def _nearest_enum(query: str, enum_vals: List[str]) -> str | None:
    q = query.lower()
    enums = [str(e) for e in enum_vals]
    for e in enums:
        if e.lower() in q:
            return e
    qset = set(_tok(q))
    best, best_overlap = None, -1
    for e in enums:
        es = set(_tok(e))
        ov = len(qset & es)
        if ov > best_overlap:
            best, best_overlap = e, ov
    return best if best_overlap > 0 else None

def _coerce_type(val: Any, typ: str, enum_vals: List[str] | None = None) -> Tuple[Any, bool, str]:
    if val is None:
        return None, False, "none"
    t = (typ or "string").lower()
    try:
        if t in {"int","integer"}:
            if isinstance(val, int):
                return int(val), True, "ok"
            m = _INT_RE.search(str(val))
            return (int(m.group(0)), True, "regex-int") if m else (None, False, "no-int")
        if t in {"float","number"}:
            if isinstance(val, (int, float)):
                return float(val), True, "ok"
            m = _NUM_RE.search(str(val))
            return (float(m.group(0)), True, "regex-float") if m else (None, False, "no-float")
        if t in {"bool","boolean"}:
            s = str(val).lower().strip()
            if s in _BOOL_TRUE:  return True, True, "bool-true"
            if s in _BOOL_FALSE: return False, True, "bool-false"
            return None, False, "no-bool"
        if t in {"list","array"}:
            if isinstance(val, list):
                return val, True, "ok"
            s = str(val)
            if "," in s:
                return [x.strip() for x in s.split(",") if x.strip()], True, "csv"
            return [str(val)], True, "singleton"
        if t in {"dict","object"}:
            if isinstance(val, dict):
                return val, True, "ok"
            s = str(val)
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj, True, "json"
            except Exception:
                pass
            return {"value": str(val)}, True, "boxed"
        if t == "enum":
            if enum_vals:
                s = str(val)
                return s, (s in enum_vals), "enum"
            return str(val), True, "enum-noval"
        return str(val), True, "string"
    except Exception as e:
        return None, False, f"exception:{e}"

def fill_arguments(query: str, tool_schema: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    规则填参：
      - int/float: 正则
      - bool: 词表
      - date: yyyy-mm-dd 正则
      - enum: 精确/包含/Token 交集 最近邻
      - default: 用 schema 默认值；否则 None
    返回 (pred_args, debug_log)
    """
    args_def = tool_schema.get("arguments") or tool_schema.get("args") or []
    q = query
    tokens = _tok(query)

    pred, log = {}, {"per_arg": []}

    for a in args_def:
        name  = a.get("name") or a.get("arg") or ""
        typ   = (a.get("type") or a.get("dtype") or "string")
        req   = bool(a.get("required", False))
        enumv = a.get("enum") or a.get("choices") or None
        dflt  = a.get("default") if "default" in a else None

        raw, how = None, ""

        if "date" in name.lower() or typ.lower() in {"date","datetime"}:
            m = _DATE_RE.search(q)
            if m:
                raw, how = m.group(0), "regex-date"

        if raw is None and (typ.lower() == "enum" or enumv):
            cand = _nearest_enum(q, enumv or [])
            if cand is not None:
                raw, how = cand, "enum-nearest"

        if raw is None and typ.lower() in {"bool","boolean"}:
            for t in tokens:
                if t in _BOOL_TRUE:  raw, how = "true",  "bool-true";  break
                if t in _BOOL_FALSE: raw, how = "false", "bool-false"; break

        if raw is None and typ.lower() in {"int","integer"}:
            m = _INT_RE.search(q)
            if m: raw, how = m.group(0), "regex-int"

        if raw is None and typ.lower() in {"float","number"}:
            m = _NUM_RE.search(q)
            if m: raw, how = m.group(0), "regex-float"

        if raw is None:
            nm = name.lower()
            joined = " ".join(tokens)
            if nm and nm in joined:
                try:
                    idx = tokens.index(nm)
                    if idx + 1 < len(tokens):
                        raw, how = tokens[idx+1], "after-name"
                except Exception:
                    pass

        if raw is None and dflt is not None:
            raw, how = dflt, "default"

        if raw is None:
            raw, how = None, "missing"

        coerced, ok, note = _coerce_type(raw, typ, enumv if enumv else None)
        pred[name] = coerced
        log["per_arg"].append({
            "name": name, "type": typ, "required": req, "enum": enumv,
            "raw": raw, "how": how, "coerced": coerced, "ok": ok, "note": note
        })

    return pred, log
