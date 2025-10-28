# models/adapter.py
import os
from typing import Optional

# --- 可选：你也可以让 baselines/zero_shot.py 传入 system，没传就用默认 ---
_DEFAULT_SYSTEM = (
    "You are a function-calling planner. "
    "Pick exactly ONE tool and return ONLY JSON with keys: name, arguments. "
    "Arguments must satisfy the tool's JSON schema."
)

_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        _openai_client = OpenAI(api_key=key)
    return _openai_client

def generate(prompt: str,
             model_name: str,
             *,
             temperature: float = 0.0,
             max_new_tokens: int = 256,
             system: Optional[str] = None) -> str:
    """
    统一接口：根据 model_name 前缀分发到不同后端。
    目前实现：openai:xxx （例如 openai:gpt-4o-mini）
    """
    if model_name.startswith("openai:"):
        model = model_name.split(":", 1)[1]
        client = _get_openai_client()
        sys_msg = system or _DEFAULT_SYSTEM
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_new_tokens,
            # 让模型直接输出 JSON（新 SDK 支持）
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    # 其他后端占位（mistral:, qwen:, local: 等）
    raise ValueError(f"Unsupported model backend: {model_name}")
