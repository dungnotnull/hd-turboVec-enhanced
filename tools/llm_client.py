"""
LLM Client — Unified Claude / OpenAI / Ollama client for TurboVec Enhanced.

Provider priority: Claude → OpenAI → Ollama (automatic fallback on error)
Streaming: supported for all providers
Cost tracking: logs USD cost per call via MemoryManager
Retry: exponential backoff (1s, 2s, 4s) on RateLimitError / NetworkError
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

PROVIDER_ORDER = ["claude", "openai", "ollama"]

COST_PER_1K = {
    "claude-opus-4-8":     {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-6":   {"input": 0.003, "output": 0.015},
    "gpt-4o":              {"input": 0.005, "output": 0.015},
    "gpt-4o-mini":         {"input": 0.00015, "output": 0.0006},
    "ollama":              {"input": 0.0, "output": 0.0},
}


class LLMClient:
    """
    Unified LLM client supporting Claude, OpenAI, and Ollama.

    Environment variables:
      ANTHROPIC_API_KEY  — Claude API key
      OPENAI_API_KEY     — OpenAI API key
      OLLAMA_BASE_URL    — Ollama base URL (default: http://localhost:11434)
      LLM_PROVIDER       — Force a specific provider (claude|openai|ollama)
      LLM_MODEL          — Force a specific model
    """

    def __init__(self, memory=None):
        self.memory = memory
        self._forced_provider = os.getenv("LLM_PROVIDER", "").lower() or None
        self._forced_model = os.getenv("LLM_MODEL", "") or None

        self._claude_client = None
        self._openai_client = None
        self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ──────────────────────── Public API ────────────────────────

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        use_case: str = "",
    ) -> Dict[str, Any]:
        """
        Generate a completion with automatic provider fallback.

        Args:
            prompt: User prompt string
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)
            use_case: Label for cost tracking (e.g. 'rag_synthesis')

        Returns:
            dict with keys: text, provider, model, prompt_tokens, completion_tokens, cost_usd
        """
        providers = [self._forced_provider] if self._forced_provider else PROVIDER_ORDER
        last_error = None

        for provider in providers:
            try:
                result = self._call_with_retry(
                    provider=provider,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self._log_cost(provider=provider, model=result["model"],
                               prompt_tokens=result["prompt_tokens"],
                               completion_tokens=result["completion_tokens"],
                               cost_usd=result["cost_usd"],
                               use_case=use_case)
                return result
            except Exception as exc:
                logger.warning("Provider %s failed: %s — trying next", provider, exc)
                last_error = exc

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        """
        Stream completion tokens from the first available provider.

        Yields: individual text chunks as they arrive
        """
        providers = [self._forced_provider] if self._forced_provider else PROVIDER_ORDER
        for provider in providers:
            try:
                yield from self._stream_provider(provider, prompt, system, max_tokens)
                return
            except Exception as exc:
                logger.warning("Streaming from %s failed: %s — trying next", provider, exc)
        raise RuntimeError("All LLM providers failed during streaming")

    # ──────────────────────── Retry Logic ────────────────────────

    def _call_with_retry(self, provider: str, **kwargs) -> Dict[str, Any]:
        delays = [1, 2, 4]
        for attempt, delay in enumerate(delays + [None]):
            try:
                return self._call_provider(provider=provider, **kwargs)
            except Exception as exc:
                err_str = str(exc).lower()
                is_retryable = any(kw in err_str for kw in
                                   ["rate limit", "overloaded", "timeout", "503", "529", "connection"])
                if delay is not None and is_retryable:
                    logger.info("Retrying %s in %ds (attempt %d)", provider, delay, attempt + 1)
                    time.sleep(delay)
                else:
                    raise

    def _call_provider(self, provider: str, prompt: str, system: Optional[str],
                       max_tokens: int, temperature: float) -> Dict[str, Any]:
        if provider == "claude":
            return self._call_claude(prompt, system, max_tokens, temperature)
        elif provider == "openai":
            return self._call_openai(prompt, system, max_tokens, temperature)
        elif provider == "ollama":
            return self._call_ollama(prompt, system, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ──────────────────────── Claude ────────────────────────

    def _get_claude(self):
        if self._claude_client is None:
            import anthropic
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._claude_client = anthropic.Anthropic(api_key=key)
        return self._claude_client

    def _call_claude(self, prompt: str, system: Optional[str], max_tokens: int,
                     temperature: float) -> Dict[str, Any]:
        client = self._get_claude()
        model = self._forced_model or "claude-opus-4-8"
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = client.messages.create(**kwargs)
        text = response.content[0].text
        pt = response.usage.input_tokens
        ct = response.usage.output_tokens
        cost = (pt / 1000) * COST_PER_1K.get(model, {}).get("input", 0.015) + \
               (ct / 1000) * COST_PER_1K.get(model, {}).get("output", 0.075)
        return {"text": text, "provider": "claude", "model": model,
                "prompt_tokens": pt, "completion_tokens": ct, "cost_usd": round(cost, 6)}

    def _stream_provider(self, provider: str, prompt: str, system: Optional[str],
                         max_tokens: int) -> Generator[str, None, None]:
        if provider == "claude":
            client = self._get_claude()
            model = self._forced_model or "claude-opus-4-8"
            messages = [{"role": "user", "content": prompt}]
            kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
            if system:
                kwargs["system"] = system
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
        elif provider == "openai":
            oai = self._get_openai()
            model = self._forced_model or "gpt-4o"
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            stream = oai.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        elif provider == "ollama":
            import urllib.request
            model = self._forced_model or "llama3"
            payload = json.dumps({
                "model": model, "prompt": prompt, "stream": True,
                "options": {"num_predict": max_tokens}
            }).encode()
            req = urllib.request.Request(
                f"{self._ollama_base_url}/api/generate",
                data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                for line in resp:
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        if data.get("response"):
                            yield data["response"]

    # ──────────────────────── OpenAI ────────────────────────

    def _get_openai(self):
        if self._openai_client is None:
            import openai
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._openai_client = openai.OpenAI(api_key=key)
        return self._openai_client

    def _call_openai(self, prompt: str, system: Optional[str], max_tokens: int,
                     temperature: float) -> Dict[str, Any]:
        oai = self._get_openai()
        model = self._forced_model or "gpt-4o"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = oai.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        text = response.choices[0].message.content
        pt = response.usage.prompt_tokens
        ct = response.usage.completion_tokens
        cost = (pt / 1000) * COST_PER_1K.get(model, {}).get("input", 0.005) + \
               (ct / 1000) * COST_PER_1K.get(model, {}).get("output", 0.015)
        return {"text": text, "provider": "openai", "model": model,
                "prompt_tokens": pt, "completion_tokens": ct, "cost_usd": round(cost, 6)}

    # ──────────────────────── Ollama ────────────────────────

    def _call_ollama(self, prompt: str, system: Optional[str], max_tokens: int,
                     temperature: float) -> Dict[str, Any]:
        import urllib.request
        model = self._forced_model or "llama3"
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        payload = json.dumps({
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }).encode()
        req = urllib.request.Request(
            f"{self._ollama_base_url}/api/generate",
            data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("response", "")
        return {"text": text, "provider": "ollama", "model": model,
                "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    # ──────────────────────── Cost Logging ────────────────────────

    def _log_cost(self, provider: str, model: str, prompt_tokens: int,
                  completion_tokens: int, cost_usd: float, use_case: str):
        if self.memory is not None:
            try:
                self.memory.log_llm_cost(
                    provider=provider, model=model,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    cost_usd=cost_usd, use_case=use_case,
                )
            except Exception as exc:
                logger.debug("Cost logging failed: %s", exc)
