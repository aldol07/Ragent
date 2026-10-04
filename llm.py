"""Thin wrapper around OpenRouter's OpenAI-compatible chat API."""
import json
import re

from openai import OpenAI

from config import LLM_FALLBACK_MODELS, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE, OPENROUTER_API_KEY, OPENROUTER_BASE_URL


class LLMNotConfigured(RuntimeError):
    pass


class LLM:
    def __init__(self, api_key=OPENROUTER_API_KEY, model=LLM_MODEL, fallbacks=LLM_FALLBACK_MODELS):
        if not api_key:
            raise LLMNotConfigured("OPENROUTER_API_KEY is not set. Add it to your .env file.")
        self.model = model
        self.fallbacks = [m for m in fallbacks if m != model]
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            default_headers={"X-Title": "Ragent"},
            timeout=60,
            max_retries=2,
        )
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.last_model = model

    def _track(self, usage):
        if usage:
            self.usage["prompt_tokens"] += usage.prompt_tokens or 0
            self.usage["completion_tokens"] += usage.completion_tokens or 0

    def _routing(self, require_tools=False):
        """OpenRouter extras: model fallback list and, for tool calls, providers that support tools."""
        extra = {}
        if self.fallbacks:
            extra["models"] = [self.model, *self.fallbacks]
        if require_tools:
            extra["provider"] = {"require_parameters": True}
        return extra

    def reset_usage(self):
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def complete(self, messages, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS):
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            extra_body=self._routing(),
        )
        self._track(resp.usage)
        return (resp.choices[0].message.content or "").strip()

    def complete_json(self, messages, **kwargs):
        """Ask for JSON and parse it leniently (models sometimes wrap it in ```json)."""
        return parse_json(self.complete(messages, **kwargs))

    def stream(self, messages, tools=None, tool_choice=None):
        """Stream one assistant turn.

        Yields ("text", str) for content deltas and finally ("done", message_dict),
        where message_dict is an assistant message (possibly with tool_calls).
        """
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=self._routing(require_tools=bool(tools)),
        )
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        content, calls = [], {}
        for event in self.client.chat.completions.create(**kwargs):
            if getattr(event, "model", None):
                self.last_model = event.model
            if getattr(event, "usage", None):
                self._track(event.usage)
            if not event.choices:
                continue
            delta = event.choices[0].delta
            if delta.content:
                content.append(delta.content)
                yield "text", delta.content
            for tc in delta.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments

        message = {"role": "assistant", "content": "".join(content)}
        if calls:
            message["tool_calls"] = [
                {
                    "id": c["id"] or f"call_{i}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                }
                for i, c in sorted(calls.items())
            ]
        yield "done", message


def parse_json(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
