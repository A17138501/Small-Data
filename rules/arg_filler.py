# -*- coding: utf-8 -*-
from __future__ import annotations
import json, re
from typing import Dict, Any, List, Tuple

# ===== Regex =====
_NUM_RE      = re.compile(r"[-+]?\d+(?:\.\d+)?")
_INT_RE      = re.compile(r"[-+]?\d+")
_BOOL_TRUE   = {"true","yes","y","1","on"}
_BOOL_FALSE  = {"false","no","n","0","off"}
_DATE_RE     = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_WORD_RE     = re.compile(r"[A-Za-z0-9_]+")


# ===== 工具字段规范化 =====
def normalize_tools_field(tools_field: str | list | dict) -> List[Dict[str, Any]]:
    """接受 JSON 字符串、list、或 {'tools':[...]}，统一返回 list[dict]"""
    if isinstance(tools_field, str):
        tools = json.loads(tools_field)
    else:
        tools = tools_field

    if isinstance(tools, dict) and "tools" in tools:
        tools = tools["tools"]

    if isinstance(tools, dict):
        tools = [tools]

    return tools


def _tok(s: str) -> List[str]:
    if not s:
        return []
    return [w.lower() for w in _WORD_RE.findall(s)]


def _nearest_enum(query: str, enum_vals: List[str]) -> str | None:
    """取枚举最接近 query 的值"""
    q = query.lower()
    enums = [str(e) for e in enum_vals]

    # 直接包含
    for e in enums:
        if e.lower() in q:
            return e

    # token 交集最大
    qset = set(_tok(q))
    best, best_overlap = None, -1
    for e in enums:
        es = set(_tok(e))
        ov = len(qset & es)
        if ov > best_overlap:
            best, best_overlap = e, ov

    return best if best_overlap > 0 else None


# ===== 类型强制 =====
def _coerce_type(val: Any, typ: str, enum_vals: List[str] | None = None) -> Tuple[Any, bool, str]:
    if val is None:
        return None, False, "none"

    t = (typ or "string").lower()

    try:
        # ------ int ------
        if t in {"int","integer"}:
            if isinstance(val, int):
                return int(val), True, "ok"
            m = _INT_RE.search(str(val))
            if m:
                return int(m.group(0)), True, "regex-int"
            return None, False, "no-int"

        # ------ float / number ------
        if t in {"float","number"}:
            if isinstance(val, (int,float)):
                return float(val), True, "ok"
            m = _NUM_RE.search(str(val))
            if m:
                return float(m.group(0)), True, "regex-float"
            return None, False, "no-float"

        # ------ bool ------
        if t in {"bool","boolean"}:
            s = str(val).lower().strip()
            if s in _BOOL_TRUE:  return True, True, "bool-true"
            if s in _BOOL_FALSE: return False, True, "bool-false"
            return None, False, "no-bool"

        # ------ list ------
        if t in {"list","array"}:
            if isinstance(val, list):
                return val, True, "ok"
            s = str(val)
            if "," in s:
                return [x.strip() for x in s.split(",") if x.strip()], True, "csv"
            return [str(val)], True, "singleton"

        # ------ dict ------
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

        # ------ enum ------
        if t == "enum":
            if enum_vals:
                s = str(val)
                return s, (s in enum_vals), "enum"
            return str(val), True, "enum-nochoices"

        # ------ default: string ------
        return str(val), True, "string"

    except Exception as e:
        return None, False, f"exception:{e}"


# ===== 核心：参数填充 =====
def fill_arguments(query: str, tool_schema: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    规则填参：
      - int/float: 正则
      - bool: yes/no...
      - enum: 最近邻
      - date: yyyy-mm-dd
      - name-based heuristics
      - default fallback
    返回 (pred_args, debug_log)
    """

    # ---- 支持多种参数字段格式 ----
    args_def = (
        tool_schema.get("arguments")
        or tool_schema.get("args")
        or tool_schema.get("parameters")        # 修复你遇到的问题
        or []
    )

    # ---- 如果是 dict，则转换成 list[dict] ----
    if isinstance(args_def, dict):
        fixed = []
        for arg_name, arg_info in args_def.items():
            if isinstance(arg_info, dict):
                item = {"name": arg_name}
                item.update(arg_info)
                fixed.append(item)
            else:
                fixed.append({"name": arg_name, "type": "string", "default": None})
        args_def = fixed

    # ---- 如果参数不是 list，兜底 ----
    if not isinstance(args_def, list):
        args_def = []

    q = query
    tokens = _tok(query)

    pred = {}
    log = {"per_arg": []}

    # ==========================================================
    #   逐个参数根据类型/名称进行抽取
    # ==========================================================
    for a in args_def:
        name  = a.get("name") or a.get("arg") or ""
        typ   = (a.get("type") or a.get("dtype") or "string")
        req   = bool(a.get("required", False))
        enumv = a.get("enum") or a.get("choices") or None
        dflt  = a.get("default") if "default" in a else None

        raw, how = None, ""

        # ---- 1) 日期 ----
        if "date" in name.lower() or typ.lower() in {"date","datetime"}:
            m = _DATE_RE.search(q)
            if m:
                raw, how = m.group(0), "regex-date"

        # ---- 2) 枚举 ----
        if raw is None and (typ.lower() == "enum" or enumv):
            cand = _nearest_enum(q, enumv or [])
            if cand is not None:
                raw, how = cand, "enum-nearest"

        # ---- 3) 布尔 ----
        if raw is None and typ.lower() in {"bool","boolean"}:
            for tok in tokens:
                if tok in _BOOL_TRUE:
                    raw, how = "true", "bool-true"
                    break
                if tok in _BOOL_FALSE:
                    raw, how = "false", "bool-false"
                    break

        # ---- 4) int ----
        if raw is None and typ.lower() in {"int","integer"}:
            m = _INT_RE.search(q)
            if m:
                raw, how = m.group(0), "regex-int"

        # ---- 5) float ----
        if raw is None and typ.lower() in {"float","number"}:
            m = _NUM_RE.search(q)
            if m:
                raw, how = m.group(0), "regex-float"

        # ---- 6) name-based heuristic ----
        if raw is None:
            nm = name.lower()
            joined = " ".join(tokens)
            if nm in joined:
                try:
                    idx = tokens.index(nm)
                    if idx + 1 < len(tokens):
                        raw, how = tokens[idx+1], "after-name"
                except Exception:
                    pass

        # ---- 7) Default fallback ----
        if raw is None and dflt is not None:
            raw, how = dflt, "default"

        # ---- 8) Missing ----
        if raw is None:
            raw, how = None, "missing"

        # ---- 强制类型转换 ----
        coerced, ok, note = _coerce_type(raw, typ, enumv)
        pred[name] = coerced

        # ---- Debug ----
        log["per_arg"].append({
            "name": name,
            "type": typ,
            "required": req,
            "enum": enumv,
            "raw": raw,
            "how": how,
            "coerced": coerced,
            "ok": ok,
            "note": note
        })

    return pred, log


