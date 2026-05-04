"""Unified LLM client with multi-provider support.

Supports Anthropic, OpenAI, and DeepSeek APIs with prompt caching
and configurable concurrency.  Supports batch APIs (50% cost) for
OpenAI and Anthropic when use_batch_api=True.

Providers:
  - anthropic: Claude models with explicit cache_control on system prompt
  - openai: GPT models via OpenAI API
  - deepseek: DeepSeek-V3 via OpenAI-compatible API (automatic prefix caching)
  - deepseek-reasoner: DeepSeek-R1 (no JSON mode, no temperature — uses regex parsing)
  - gemini: Gemini models via OpenAI-compatible API

Usage:
    from llm_client import llm_batch

    results = llm_batch(
        prompts=["prompt1", "prompt2"],
        system_prompt="You are a helpful assistant.",
        provider="deepseek",
    )

    # 50% cheaper via batch API (async, may take minutes-hours):
    results = llm_batch(
        prompts=["prompt1", "prompt2"],
        system_prompt="You are a helpful assistant.",
        provider="openai",
        use_batch_api=True,
    )
"""

import io
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from config import S2_API_KEY  # just to trigger dotenv loading via config
from cost_tracker import tracker

# Models that don't support temperature parameter (auto-detected at runtime)
_NO_TEMP_MODELS: set[str] = set()

# ── Defaults ─────────────────────────────────────────────────────────────────

PROVIDER_DEFAULTS = {
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "max_workers": 4,
        "max_tokens": 4096,
    },
    "openai": {
        "model": "gpt-4.1-mini",
        "max_workers": 4,
        "max_tokens": 4096,
    },
    "deepseek": {
        "model": "deepseek-chat",
        "max_workers": 16,  # rate-limited at ~100 concurrent
        "max_tokens": 4000,
    },
    "deepseek-reasoner": {
        "model": "deepseek-reasoner",
        "max_workers": 16,  # rate-limited at ~100 concurrent
        "max_tokens": 16000,  # includes reasoning tokens
    },
    "sglang": {
        "model": "default",  # sglang serves whatever model is loaded
        "max_workers": 32,   # local GPU — high concurrency OK
        "max_tokens": 500,
        "base_url": "http://localhost:40000/v1",
    },
    "gemini": {
        "model": "gemini-2.5-pro",
        "max_workers": 8,
        "max_tokens": 4000,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}


# ── JSON parsing ─────────────────────────────────────────────────────────────

def parse_llm_json(content: str) -> dict:
    """Extract JSON from LLM response text."""
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {"score": 0, "reasoning": f"parse_error: {content[:200]}"}


def error_result(error):
    """Consistent error fallback for any LLM call."""
    return {"score": 0, "curiosity_gap": f"error: {error}",
            "candidate_fact": None, "reasoning": f"error: {error}"}


# ── Provider implementations ─────────────────────────────────────────────────

def _batch_anthropic(prompts, system_prompt, model, max_tokens, max_workers,
                     temperature=0.0):
    from anthropic import Anthropic
    client = Anthropic()

    # Cache the system prompt for reuse across calls
    system_msg = [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }]

    results = [None] * len(prompts)

    def _call(idx, user_prompt):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_msg,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
            )
            tracker.record("anthropic", model,
                           resp.usage.input_tokens, resp.usage.output_tokens)
            return idx, parse_llm_json(resp.content[0].text)
        except Exception as e:
            return idx, error_result(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_call, i, p) for i, p in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"anthropic/{model}", leave=False):
            idx, result = future.result()
            results[idx] = result

    return results


def _batch_openai_compat(prompts, system_prompt, model, max_tokens, max_workers,
                         base_url=None, api_key=None, json_mode=True,
                         temperature=0.0, extra_body=None, _provider="openai"):
    """OpenAI-compatible batch caller. Works for OpenAI, DeepSeek, and others."""
    from openai import OpenAI

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    client = OpenAI(**kwargs)
    results = [None] * len(prompts)
    _errors_shown = [0]

    # Check module-level cache for models that don't support temperature
    _skip_temp = [model in _NO_TEMP_MODELS]

    def _call(idx, user_prompt):
        try:
            call_kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_tokens,
            )
            if not _skip_temp[0]:
                call_kwargs["temperature"] = temperature
            if json_mode:
                call_kwargs["response_format"] = {"type": "json_object"}
            if extra_body:
                call_kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**call_kwargs)
            if resp.usage:
                tracker.record(_provider, model,
                               resp.usage.prompt_tokens,
                               resp.usage.completion_tokens)
            choice = resp.choices[0]
            content = choice.message.content or ""
            # Reasoning models (Qwen3, DeepSeek-R1 via sglang) put output
            # in reasoning_content; fall back to it when content is empty
            if not content.strip():
                content = getattr(choice.message, "reasoning_content", None) or ""
            try:
                return idx, json.loads(content)
            except json.JSONDecodeError:
                parsed = parse_llm_json(content)
                # If regex parsing also failed, preserve raw text
                if "parse_error" in parsed.get("reasoning", ""):
                    parsed["text"] = content.strip()
                return idx, parsed
        except Exception as e:
            # Auto-detect models that don't support temperature — retry once
            if "temperature" in str(e) and "not support" in str(e):
                if not _skip_temp[0]:
                    _skip_temp[0] = True
                    _NO_TEMP_MODELS.add(model)
                    print(f"\n  [llm_client] {model} doesn't support temperature, retrying without it")
                return _call(idx, user_prompt)
            if _errors_shown[0] < 3:
                print(f"\n  [llm_client] Error #{_errors_shown[0]+1}: {e}")
                _errors_shown[0] += 1
            return idx, error_result(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_call, i, p) for i, p in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"{model}", leave=False):
            idx, result = future.result()
            results[idx] = result

    return results


def _batch_deepseek_reasoner(prompts, system_prompt, model, max_tokens, max_workers):
    """DeepSeek-R1 batch caller. JSON mode supported, temperature ignored."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    results = [None] * len(prompts)

    def _call(idx, user_prompt):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            if resp.usage:
                tracker.record("deepseek-reasoner", model,
                               resp.usage.prompt_tokens,
                               resp.usage.completion_tokens)
            choice = resp.choices[0]
            # R1 returns reasoning_content + content
            reasoning = getattr(choice.message, "reasoning_content", None) or ""
            content = choice.message.content or ""
            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                parsed = parse_llm_json(content)
            # Attach reasoning trace for inspection
            if reasoning:
                parsed["_reasoning"] = reasoning
            return idx, parsed
        except Exception as e:
            return idx, error_result(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_call, i, p) for i, p in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"reasoner/{model}", leave=False):
            idx, result = future.result()
            results[idx] = result

    return results


# ── Batch API implementations (50% cost reduction) ──────────────────────────

BATCH_POLL_INTERVAL = 30  # seconds between status checks


def _batch_api_openai(prompts, system_prompt, model, max_tokens,
                      temperature=0.0):
    """OpenAI Batch API — 50% cheaper, async processing."""
    from openai import OpenAI
    client = OpenAI()

    # Build JSONL request body
    lines = []
    for i, prompt in enumerate(prompts):
        request = {
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
        }
        lines.append(json.dumps(request))

    jsonl_content = "\n".join(lines)

    # Upload input file
    print(f"  [batch-openai] Uploading {len(prompts)} requests...")
    input_file = client.files.create(
        file=io.BytesIO(jsonl_content.encode()),
        purpose="batch",
    )

    # Create batch
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  [batch-openai] Batch {batch.id} created, waiting...")

    # Poll for completion
    while True:
        batch = client.batches.retrieve(batch.id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else len(prompts)

        if status == "completed":
            print(f"  [batch-openai] Completed: {completed}/{total}")
            break
        elif status in ("failed", "expired", "cancelled"):
            print(f"  [batch-openai] Batch {status}!")
            errors = []
            if batch.errors and batch.errors.data:
                for err in batch.errors.data[:5]:
                    errors.append(f"    {err.code}: {err.message}")
                print("\n".join(errors))
            return [error_result(f"Batch {status}")] * len(prompts)
        else:
            print(f"  [batch-openai] Status: {status} ({completed}/{total})",
                  end="\r")
            time.sleep(BATCH_POLL_INTERVAL)

    # Download results
    output_content = client.files.content(batch.output_file_id)
    result_lines = output_content.text.strip().split("\n")

    results = [None] * len(prompts)
    for line in result_lines:
        entry = json.loads(line)
        idx = int(entry["custom_id"].split("-")[1])
        resp = entry.get("response", {})
        if resp.get("status_code") == 200:
            body = resp.get("body", {})
            usage = body.get("usage", {})
            if usage:
                tracker.record("openai-batch", model,
                               usage.get("prompt_tokens", 0),
                               usage.get("completion_tokens", 0))
            content = body.get("choices", [{}])[0].get(
                "message", {}).get("content", "")
            try:
                results[idx] = json.loads(content)
            except json.JSONDecodeError:
                results[idx] = parse_llm_json(content)
        else:
            err = resp.get("error", {}).get("message", "unknown error")
            results[idx] = error_result(err)

    # Fill any missing results
    for i in range(len(results)):
        if results[i] is None:
            results[i] = error_result("missing from batch output")

    # Cleanup uploaded file
    try:
        client.files.delete(input_file.id)
    except Exception:
        pass

    return results


def _batch_api_anthropic(prompts, system_prompt, model, max_tokens,
                         temperature=0.0):
    """Anthropic Message Batches API — 50% cheaper, async processing."""
    from anthropic import Anthropic
    client = Anthropic()

    # Build batch requests
    requests = []
    for i, prompt in enumerate(prompts):
        requests.append({
            "custom_id": f"req-{i}",
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                "system": system_prompt,
            },
        })

    # Create batch
    print(f"  [batch-anthropic] Submitting {len(prompts)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  [batch-anthropic] Batch {batch.id} created, waiting...")

    # Poll for completion
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        status = batch.processing_status

        counts = batch.request_counts
        completed = (counts.succeeded + counts.errored
                     + counts.expired + counts.canceled)

        if status == "ended":
            print(f"  [batch-anthropic] Ended: {counts.succeeded} succeeded, "
                  f"{counts.errored} errored")
            break
        else:
            print(f"  [batch-anthropic] Status: {status} "
                  f"({completed}/{counts.processing + completed})", end="\r")
            time.sleep(BATCH_POLL_INTERVAL)

    # Retrieve results
    results = [None] * len(prompts)
    for entry in client.messages.batches.results(batch.id):
        idx = int(entry.custom_id.split("-")[1])
        if entry.result.type == "succeeded":
            usage = entry.result.message.usage
            if usage:
                tracker.record("anthropic-batch", model,
                               usage.input_tokens, usage.output_tokens)
            if entry.result.message.content:
                content = entry.result.message.content[0].text
                try:
                    results[idx] = json.loads(content)
                except json.JSONDecodeError:
                    results[idx] = parse_llm_json(content)
            else:
                results[idx] = error_result("empty response content")
        else:
            results[idx] = error_result(
                f"batch entry {entry.result.type}")

    for i in range(len(results)):
        if results[i] is None:
            results[i] = error_result("missing from batch output")

    return results


# ── Public API ───────────────────────────────────────────────────────────────

def llm_batch(
    prompts: list[str],
    system_prompt: str,
    provider: str = "deepseek",
    model: str = None,
    max_tokens: int = None,
    max_workers: int = None,
    temperature: float = 0.0,
    use_batch_api: bool = False,
) -> list[dict]:
    """Send a batch of prompts to an LLM provider with concurrent execution.

    Args:
        prompts: List of user prompts.
        system_prompt: System prompt (cached/reused across calls).
        provider: "anthropic", "openai", "deepseek", or "deepseek-reasoner".
        model: Model override (uses provider default if None).
        max_tokens: Max output tokens per response.
        max_workers: Concurrent workers (higher for DeepSeek since no rate limits).
        temperature: Sampling temperature (default: 0.0). Ignored by deepseek-reasoner.
        use_batch_api: Use async batch API for 50% cost reduction (OpenAI/Anthropic).

    Returns:
        List of parsed JSON dicts, one per prompt, in the same order.
    """
    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["deepseek"])
    model = model or defaults["model"]
    max_tokens = max_tokens or defaults["max_tokens"]
    max_workers = max_workers or defaults["max_workers"]

    if not prompts:
        return []

    # Route to batch API if requested (50% cheaper for OpenAI/Anthropic)
    if use_batch_api:
        if provider == "openai":
            return _batch_api_openai(
                prompts, system_prompt, model, max_tokens,
                temperature=temperature)
        elif provider == "anthropic":
            return _batch_api_anthropic(
                prompts, system_prompt, model, max_tokens,
                temperature=temperature)
        else:
            print(f"  [batch] Batch API not available for {provider}, "
                  f"falling back to real-time")

    if provider == "anthropic":
        return _batch_anthropic(prompts, system_prompt, model, max_tokens, max_workers,
                                temperature=temperature)

    elif provider == "openai":
        return _batch_openai_compat(prompts, system_prompt, model, max_tokens, max_workers,
                                    temperature=temperature)

    elif provider == "deepseek":
        return _batch_openai_compat(
            prompts, system_prompt, model, max_tokens, max_workers,
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=temperature,
            _provider="deepseek",
        )

    elif provider == "deepseek-reasoner":
        # R1's max_tokens covers reasoning + content combined.
        # Default 16K is generous — reasoning typically uses 2-8K,
        # leaving plenty for the short JSON answer.
        r1_max_tokens = max(max_tokens + 8000, 16384)
        return _batch_deepseek_reasoner(
            prompts, system_prompt, model, r1_max_tokens, max_workers,
        )

    elif provider == "sglang":
        base_url = defaults.get("base_url", "http://localhost:40000/v1")
        # If no model specified, query sglang for the served model name
        if model == "default":
            try:
                from openai import OpenAI
                _client = OpenAI(base_url=base_url, api_key="EMPTY")
                models = _client.models.list()
                model = models.data[0].id if models.data else "default"
            except Exception:
                pass
        return _batch_openai_compat(
            prompts, system_prompt, model, max_tokens, max_workers,
            base_url=base_url,
            api_key="EMPTY",
            json_mode=False,  # local models may not support JSON mode
            temperature=temperature,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
            _provider="sglang",
        )

    elif provider == "gemini":
        return _batch_openai_compat(
            prompts, system_prompt, model, max_tokens, max_workers,
            base_url=defaults.get("base_url"),
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
            _provider="gemini",
        )

    else:
        print(f"Unknown provider: {provider}")
        return [error_result(f"Unknown provider: {provider}")] * len(prompts)


def llm_single(
    prompt: str,
    system_prompt: str,
    provider: str = "deepseek",
    model: str = None,
    max_tokens: int = None,
) -> dict:
    """Send a single prompt. Convenience wrapper around llm_batch."""
    results = llm_batch(
        [prompt], system_prompt,
        provider=provider, model=model, max_tokens=max_tokens, max_workers=1,
    )
    return results[0]
