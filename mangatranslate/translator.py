# -*- coding: utf-8 -*-
"""
Translation module using OpenAI-compatible Gemini 3.8 Flash API with local caching and retries.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import hashlib
import json
import os
import re
import time
import requests


SYSTEM_PROMPT = """你是一位资深的日漫汉化组翻译专家。请将给出的日文漫画对话、旁白或拟声词翻译为地道、自然、符合中文阅读习惯的简体中文。

规则要求：
1. 必须返回纯 JSON 格式：{"translations": ["译文1", "译文2", ...]}，译文数量必须与输入的 lines 数量严格一致且顺序一一对应。
2. 准确传达角色的语气（傲娇、惊慌、娇羞、命令等）、叹号、问号及拟声词（如“あっあっ” -> “啊、啊”、“わっ” -> “哇”）。
3. 人名及称谓遵循中文习惯（如“河嶋”保留或译为“河岛”，“桃ちゃん”译为“小桃”）。
4. 绝对不要输出任何翻译说明、注释、寒暄或额外的 Markdown 格式（如解释代码块）。直接返回 JSON。"""


class GeminiTranslator:
    """Translator communicating with OpenAI-compatible API endpoint with caching and retries."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8045/v1",
        api_key: Optional[str] = None,
        model: str = "gemini-3.8-flash-tiered",
        temperature: float = 0.2,
        timeout: int = 60,
        max_retries: int = 3,
        cache_path: Optional[Union[str, Path]] = None,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("MANGATRANSLATE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        # Local cache setup
        self.cache_path = Path(cache_path) if cache_path else Path(".cache/translation_cache.json")
        self.cache: Dict[str, List[str]] = self._load_cache()

    def _load_cache(self) -> Dict[str, List[str]]:
        if self.cache_path.is_file():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _get_cache_key(self, lines: List[str]) -> str:
        content = f"{self.model}:{chr(10).join(lines)}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def translate_lines(
        self, 
        lines: List[str], 
        context_hint: str = ""
    ) -> List[str]:
        """
        Translate a list of Japanese text lines to Simplified Chinese.
        
        Args:
            lines: List of Japanese text lines for a page.
            context_hint: Optional page context or scene hint.
            
        Returns:
            List of translated Chinese lines matching the input length.
        """
        if not lines:
            return []

        # Check local cache first
        cache_key = self._get_cache_key(lines)
        if cache_key in self.cache:
            cached_res = self.cache[cache_key]
            if len(cached_res) == len(lines):
                return cached_res

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        user_content = {"lines": lines}
        if context_hint:
            user_content["context"] = context_hint

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    parsed = self._extract_json(content)
                    
                    if parsed and "translations" in parsed:
                        translations = parsed["translations"]
                        if len(translations) == len(lines):
                            # Save to cache
                            self.cache[cache_key] = translations
                            self._save_cache()
                            return translations
                        else:
                            last_error = f"Count mismatch: expected {len(lines)}, got {len(translations)}"
                    else:
                        last_error = f"Failed to extract JSON from response: {content[:100]}"
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as e:
                last_error = f"Request exception: {type(e).__name__} {e}"

            time.sleep(2 * attempt)

        # Fallback if batch translation failed: return original lines or error note
        print(f"[Warning] Translation failed after {self.max_retries} attempts ({last_error}). Falling back to line-by-line or raw text.")
        return self._fallback_translate(lines, headers)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from model response, handling Markdown code blocks."""
        text = text.strip()
        # Direct match
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except Exception:
                pass
        
        # Match ```json ... ```
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        
        # Regex search for any balanced { ... }
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return None

    def _fallback_translate(self, lines: List[str], headers: Dict[str, str]) -> List[str]:
        """Translate line-by-line as a resilient fallback."""
        results = []
        for line in lines:
            if not line.strip():
                results.append("")
                continue
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "把这句日漫台词翻译为简练自然的简体中文，直接输出译文："},
                            {"role": "user", "content": line},
                        ],
                        "temperature": 0.1,
                    },
                    timeout=20,
                )
                if r.ok:
                    results.append(r.json()["choices"][0]["message"]["content"].strip())
                else:
                    results.append(line)
            except Exception:
                results.append(line)
        return results
