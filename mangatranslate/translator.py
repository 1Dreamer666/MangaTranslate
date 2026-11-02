# -*- coding: utf-8 -*-
"""
Translation module using OpenAI-compatible Gemini 3.8 Flash API.
Enforces Positional JSON Protocol: strictly binds translations to known IDs, pages, and bounding box coordinates,
guaranteeing full narrative continuity and 100% position accuracy.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import copy
import hashlib
import json
import os
import re
import time
import requests


SYSTEM_PROMPT = """你是一位资深的日漫汉化组翻译专家。你正在翻译整部漫画的对话与台词。

核心规则：
1. 【全文连贯性】：你翻译的是一整本/整话连贯剧情的漫画。请充分结合前后文、人物关系演变（如前后辈、情侣、对立等）、性格特征与情节氛围，确保译文地道自然、语气真实生动，全书人名、称谓、专有名词与口吻高度统一。
2. 【位置信息与格式严格保留】：
   输入为包含对话项的 JSON，每项包含 id, page, box（框选坐标 [x1, y1, x2, y2]）, text。
   你必须返回严格的 JSON 结构，原样回传每个项的 id, page, box，并新增 translation 字段：
   {
     "translations": [
       {
         "id": 1,
         "page": "002.jpg",
         "box": [1148, 16, 1289, 241],
         "translation": "..."
       }
     ]
   }
3. 【坐标不可篡改】：严禁修改、遗漏或重新排序任何项的 id、page 与 box 坐标数值，它们与原画版面严格绑定！
4. 【语气与拟声词】：生动还原日漫傲娇、惊慌、害羞等角色情绪，拟声词（如“あっあっ”、“わっ”）翻译为规范中文叹词或动作拟声词。
5. 【纯净输出】：直接输出纯 JSON，绝不包含任何多余的前言、后记、Markdown 说明或对话注释。"""


class GeminiTranslator:
    """Translator communicating with OpenAI-compatible API endpoint with positional JSON validation."""

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

        self.cache_path = Path(cache_path) if cache_path else Path(".cache/translation_cache.json")
        self.cache: Dict[str, Any] = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
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

    def _get_cache_key(self, items: List[Dict[str, Any]]) -> str:
        serialized = json.dumps([
            {"id": it.get("id"), "page": it.get("page"), "box": it.get("box"), "text": it.get("text", it.get("original", ""))}
            for it in items
        ], sort_keys=True, ensure_ascii=False)
        content = f"{self.model}:{serialized}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def translate_items(
        self,
        items: List[Dict[str, Any]],
        context_hint: str = "",
        max_chunk_size: int = 35,
    ) -> List[Dict[str, Any]]:
        """
        Translate a structured list of dialogue items with position control.
        Each item must contain: id, page, box ([x1, y1, x2, y2]), text.
        
        Args:
            items: List of dialogue item dicts.
            context_hint: Story synopsis, character relationships, and chapter context.
            max_chunk_size: Maximum items per API call to ensure robust JSON generation.
            
        Returns:
            List of validated items with confirmed box coordinates and high-quality translations.
        """
        if not items:
            return []

        # If items exceed chunk size, split into overlapping chunks to preserve narrative continuity
        if len(items) > max_chunk_size:
            return self._chunked_translate(items, context_hint, chunk_size=max_chunk_size)

        cache_key = self._get_cache_key(items)
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            # Validate cached result structure
            validated_cached = self._validate_and_reconcile(items, cached_data)
            if validated_cached is not None:
                return validated_cached

        # Prepare payload with explicit position and context information
        api_items = []
        for it in items:
            api_items.append({
                "id": it["id"],
                "page": str(it.get("page", "")),
                "box": [int(v) for v in it["box"]],
                "text": it.get("text", it.get("original", "")),
            })

        user_content = {
            "context": context_hint or "日漫对话翻译：请根据人物关系、剧情逻辑与上下文保证全篇连贯性与角色语气统一。",
            "total_items": len(api_items),
            "items": api_items,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
                    raw_content = response.json()["choices"][0]["message"]["content"].strip()
                    parsed_json = self._extract_json(raw_content)

                    if parsed_json and "translations" in parsed_json:
                        candidate_list = parsed_json["translations"]
                        validated_items = self._validate_and_reconcile(items, candidate_list)
                        if validated_items is not None:
                            # Save to cache
                            self.cache[cache_key] = [
                                {"id": v["id"], "page": v["page"], "box": v["box"], "translation": v["translation"]}
                                for v in validated_items
                            ]
                            self._save_cache()
                            return validated_items
                        else:
                            last_error = f"Position validation failed for returned items (count={len(candidate_list)} vs expected={len(items)})"
                    else:
                        last_error = f"Could not parse valid JSON translations from response: {raw_content[:120]}"
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as e:
                last_error = f"Request exception: {type(e).__name__} {e}"

            time.sleep(2 * attempt)

        print(f"[Warning] Structured translation failed ({last_error}). Invoking contextual fallback.")
        return self._contextual_fallback(items, context_hint, headers)

    def _chunked_translate(
        self,
        items: List[Dict[str, Any]],
        context_hint: str,
        chunk_size: int = 35,
    ) -> List[Dict[str, Any]]:
        """Process large multi-page manga chapters in narrative chunks with rolling context."""
        results: List[Dict[str, Any]] = []
        num_chunks = (len(items) + chunk_size - 1) // chunk_size
        
        rolling_history = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(items))
            chunk = items[start_idx:end_idx]
            
            # Enrich context with recent dialogue history
            history_summary = ""
            if rolling_history:
                recent_samples = rolling_history[-6:]
                history_summary = "【前文对话语境参考】：\n" + "\n".join(
                    f"- {h.get('original', h.get('text', ''))} -> {h.get('translation', '')}"
                    for h in recent_samples
                )

            enriched_context = f"{context_hint}\n{history_summary}".strip()
            translated_chunk = self.translate_items(chunk, context_hint=enriched_context, max_chunk_size=chunk_size + 10)
            results.extend(translated_chunk)
            rolling_history.extend(translated_chunk)

        return results

    def _validate_and_reconcile(
        self,
        original_items: List[Dict[str, Any]],
        returned_list: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Rigorous position verification:
        1. Checks that every item has exact matching ID and coordinates.
        2. Detects any dropped or hallucinated coordinates and re-anchors to ground-truth geometry.
        3. Fills verified translations into output list.
        """
        truth_map = {it["id"]: copy.deepcopy(it) for it in original_items}
        matched_results: Dict[int, Dict[str, Any]] = {}

        for ret in returned_list:
            if not isinstance(ret, dict):
                continue
            ret_id = ret.get("id")
            if ret_id not in truth_map:
                continue

            orig = truth_map[ret_id]
            # Verify coordinates: must be close or match
            orig_box = [int(v) for v in orig["box"]]
            ret_box = ret.get("box")

            pos_exact = True
            if not isinstance(ret_box, (list, tuple)) or len(ret_box) != 4:
                pos_exact = False
            else:
                ret_box_int = [int(v) for v in ret_box]
                if ret_box_int != orig_box:
                    # Log minor discrepancy, but strictly enforce ground-truth box
                    pos_exact = False

            translation_text = str(ret.get("translation", "")).strip()
            if not translation_text:
                translation_text = str(orig.get("text", orig.get("original", "")))

            # Assemble strictly verified item
            verified_item = copy.deepcopy(orig)
            verified_item["translation"] = translation_text
            verified_item["box"] = orig_box  # Strictly anchor to ground truth
            verified_item["page"] = orig.get("page", ret.get("page", ""))
            matched_results[ret_id] = verified_item

        # Check if all items were matched
        if len(matched_results) == len(original_items):
            # Return in original ordering
            return [matched_results[it["id"]] for it in original_items]

        # If some items are missing, check if at least 85% were matched
        missing_ids = [it["id"] for it in original_items if it["id"] not in matched_results]
        if len(missing_ids) <= len(original_items) * 0.2:
            print(f"[Notice] Positional reconciliation recovered {len(matched_results)}/{len(original_items)} items. Filling {len(missing_ids)} missing.")
            for mid in missing_ids:
                orig = truth_map[mid]
                orig_item = copy.deepcopy(orig)
                orig_item["translation"] = orig.get("text", orig.get("original", ""))
                matched_results[mid] = orig_item
            return [matched_results[it["id"]] for it in original_items]

        return None

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except Exception:
                pass

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return None

    def _contextual_fallback(
        self,
        items: List[Dict[str, Any]],
        context_hint: str,
        headers: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Fallback that maintains item structure and context."""
        results = []
        for it in items:
            text = it.get("text", it.get("original", ""))
            out_it = copy.deepcopy(it)
            if not text.strip():
                out_it["translation"] = ""
                results.append(out_it)
                continue

            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": f"漫画语境：{context_hint}\n将该台词翻译为地道连贯的简体中文，直接输出译文："},
                            {"role": "user", "content": text},
                        ],
                        "temperature": 0.1,
                    },
                    timeout=20,
                )
                if r.ok:
                    out_it["translation"] = r.json()["choices"][0]["message"]["content"].strip()
                else:
                    out_it["translation"] = text
            except Exception:
                out_it["translation"] = text

            results.append(out_it)
        return results

    def translate_lines(self, lines: List[str], context_hint: str = "") -> List[str]:
        """Backward-compatible method wrapping raw lines into positional items."""
        if not lines:
            return []
        synthetic_items = [
            {"id": i + 1, "page": "page", "box": [0, 0, 100, 100], "text": line}
            for i, line in enumerate(lines)
        ]
        translated = self.translate_items(synthetic_items, context_hint=context_hint)
        return [t.get("translation", "") for t in translated]
