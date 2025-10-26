
# evaluation/validator.py
# Simple schema-based argument validator for tool calls.
# Supports: required, type (int/float/string/bool/list/dict), enum, ranges (min/max),
# nested object (dict with properties), list with items_type.

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

PRIMITIVES = {"int", "float", "string", "bool"}

def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)

def _is_float(x):
    return isinstance(x, float) or _is_int(x)

def _type_ok(val, t: str) -> bool:
    if t == "int":
        return _is_int(val)
    if t == "float":
        return _is_float(val)
    if t == "string":
        return isinstance(val, str)
    if t == "bool":
        return isinstance(val, bool)
    if t == "list":
        return isinstance(val, list)
    if t == "dict":
        return isinstance(val, dict)
    # unknown → accept
    return True

def _range_ok(val, schema) -> bool:
    # apply min/max if provided and val is numeric
    if not _is_float(val):
        return True
    if "min" in schema and val < schema["min"]:
        return False
    if "max" in schema and val > schema["max"]:
        return False
    return True

def _validate_list(name: str, val, schema, errors: List[str]) -> None:
    if not isinstance(val, list):
        errors.append(f"{name}: expected list, got {type(val).__name__}")
        return
    item_t = schema.get("items_type")
    if item_t:
        for i, it in enumerate(val):
            if not _type_ok(it, item_t):
                errors.append(f"{name}[{i}]: expected {item_t}, got {type(it).__name__}")

def _validate_dict(name: str, val, schema, errors: List[str]) -> None:
    if not isinstance(val, dict):
        errors.append(f"{name}: expected dict, got {type(val).__name__}")
        return
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    # required keys
    for rk in required:
        if rk not in val:
            errors.append(f"{name}: missing required key '{rk}'")
    # type checks
    for k, sub in props.items():
        if k not in val:
            continue
        v = val[k]
        t = sub.get("type")
        if t and not _type_ok(v, t):
            errors.append(f"{name}.{k}: expected {t}, got {type(v).__name__}")
            continue
        if t in ("int","float") and not _range_ok(v, sub):
            errors.append(f"{name}.{k}: out of range")
        if t == "list":
            _validate_list(f"{name}.{k}", v, sub, errors)
        if t == "dict":
            _validate_dict(f"{name}.{k}", v, sub, errors)

def build_schema_map(tools: list) -> Dict[str, dict]:
    """
    Build a {tool_name: schema} map from tools (JSON list).
    We accept a flexible shape; we look for parameters/args spec.
    Expected fields (best-effort): 
      - name: str
      - parameters: JSON schema-ish dict OR a simplified {type, properties, required}
      - Or, args_schema / schema / tool_schema as fallbacks.
    """
    out = {}
    if not isinstance(tools, list):
        return out
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name") or t.get("tool_name")
        if not name:
            continue
        # Try to find a schema-like object
        schema = t.get("parameters") or t.get("args_schema") or t.get("schema") or t.get("tool_schema")
        # If schema is None, build a permissive default
        if not isinstance(schema, dict):
            schema = {"type": "dict", "properties": {}, "required": []}
        # Normalize top-level to {type:"dict",properties,required}
        if "type" not in schema:
            schema["type"] = "dict"
        if schema["type"] == "object":
            schema["type"] = "dict"
        # JSON Schema compatibility: properties/required
        if "properties" not in schema:
            # allow "args" or "fields" as aliases
            schema["properties"] = schema.get("args") or schema.get("fields") or {}
        if "required" not in schema:
            schema["required"] = schema.get("required_fields") or []
        out[name] = schema
    return out

def validate_call(tool_name: str, arguments: dict, tools_schema: Dict[str, dict]) -> Tuple[bool, List[str]]:
    """
    Returns (ok, errors). If tool unknown, we only check that args is a dict.
    """
    errors: List[str] = []
    if not isinstance(tool_name, str) or not tool_name:
        return False, ["missing tool_name"]
    if not isinstance(arguments, dict):
        return False, [f"arguments must be dict, got {type(arguments).__name__}"]
    schema = tools_schema.get(tool_name)
    if not schema:
        # unknown tool → lenient: consider this pass if args is dict
        return True, []
    t = schema.get("type", "dict")
    if t == "dict":
        _validate_dict("arguments", arguments, schema, errors)
    elif t == "list":
        _validate_list("arguments", arguments, schema, errors)
    else:
        # primitive root schema (rare)
        if not _type_ok(arguments, t):
            errors.append(f"arguments: expected {t}")
    # enum at top-level (rare; usually for primitives)
    if "enum" in schema and arguments not in schema["enum"]:
        errors.append("arguments: value not in enum")
    return (len(errors) == 0), errors
