"""Extract per-subagent teacher conversations from a DMBP run trace.

The run trace (``agent_context/logs/trace.jsonl``) holds one OpenTelemetry
GenAI span per line. Each extraction subagent launched through the
orchestrator's ``Agent`` tool has an ``invoke_agent`` span whose descendants
are its ``chat`` (model call) and ``execute_tool`` spans. Chat spans log input
messages as deltas (``gen_ai.input.messages.offset``) that chain into the
model-visible conversation.

Message payloads are read from span events, which hold the full JSON; span
attributes are truncated by the tracer at 100,000 characters and are used only
when no event exists.

Instrumentation quirks that are tolerated:

- every model call also emits a placeholder chat span with an empty output; its
  input delta is still valid and is sometimes the only copy;
- real calls are usually emitted twice with the same response ID;
- ``cache_control`` markers move between messages from turn to turn and are
  dropped, as are ``provider_specific_fields``;
- the SDK fills default tool arguments (for example ``replace_all``) into the
  assistant message it echoes back, so the next prompt is not byte-identical to
  the raw model output. Both are kept: ``messages`` is what the model saw and
  ``outputs`` is what each call returned;
- ``execute_tool`` span results occasionally differ in whitespace from the tool
  result the model saw. The conversation is authoritative; mismatches are
  counted in ``stats``.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
AGENT_CONTEXT_PLACEHOLDER = "{AGENT_CONTEXT}"
_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"
_EMPTY_OUTPUT = [{"type": "text", "text": ""}]
# Transport hints that are not model-visible; the SDK moves cache_control between turns.
_TRANSPORT_KEYS = ("provider_specific_fields", "cache_control")
_TRUNCATION_MARKER = "... [truncated "
_PAYLOAD_KEYS = (
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.definitions",
)


class TraceError(ValueError):
    """Raised when a trace cannot be reconstructed deterministically."""


def _json_attribute(attributes: dict[str, Any], key: str, default: Any) -> Any:
    value = attributes.get(key)
    if value is None:
        return default
    if isinstance(value, str) and value.endswith(" chars]") and _TRUNCATION_MARKER in value:
        raise TraceError(f"Attribute {key} was truncated by the tracer and has no event payload")
    return json.loads(value) if isinstance(value, str) else value


def _with_event_payloads(span: dict[str, Any]) -> dict[str, Any]:
    """Replace message attributes with the untruncated copies from span events."""
    attributes = dict(span["attributes"])
    for event in span.get("events", []):
        payload = event.get("attributes", {}).get("payload")
        if event.get("name") in _PAYLOAD_KEYS and payload is not None:
            attributes[event["name"]] = json.dumps(json.loads(payload)[event["name"]])
    return {**span, "attributes": attributes}


def _seconds(start: str, end: str) -> float:
    def parse(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    return (parse(end) - parse(start)).total_seconds()


def _strip_provider_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_provider_fields(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_provider_fields(item) for key, item in value.items() if key not in _TRANSPORT_KEYS}
    return value


def _text(content: Any) -> str:
    if isinstance(content, str):
        try:
            content = json.loads(content) if content.startswith("[") else content
        except json.JSONDecodeError:
            return content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def _parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    return [part for part in content if isinstance(part, dict)] if isinstance(content, list) else []


def _signature(content: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Identify an assistant turn by its block types, tool-call IDs, and text."""
    return [(part.get("type", ""), part.get("id") or part.get("text", "")) for part in content]


def _descendants(span_id: str, children: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    found, stack = [], [span_id]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            if child["attributes"].get("gen_ai.operation.name") != "invoke_agent":
                stack.append(child["context"]["span_id"])
    return found


def _is_placeholder(span: dict[str, Any]) -> bool:
    attributes = span["attributes"]
    return (
        _json_attribute(attributes, "gen_ai.output.messages", []) == _EMPTY_OUTPUT
        and not attributes.get("gen_ai.usage.output_tokens")
        and not attributes.get("gen_ai.response.finish_reasons")
    )


def _model_calls(chat_spans: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for span in chat_spans:
        if _is_placeholder(span):
            continue
        response_id = span["attributes"].get("gen_ai.response.id")
        if not response_id:
            raise TraceError(f"{label}: chat span {span['context']['span_id']} has no response id")
        groups.setdefault(response_id, []).append(span)

    calls = []
    for response_id, spans in groups.items():
        outputs = {
            json.dumps(_json_attribute(s["attributes"], "gen_ai.output.messages", []), sort_keys=True) for s in spans
        }
        if len(outputs) != 1:
            raise TraceError(f"{label}: duplicate spans for {response_id} disagree on output")
        calls.append(min(spans, key=lambda s: s["attributes"].get("gen_ai.input.messages.offset", 0)))
    return sorted(calls, key=lambda s: s["start_time"])


def _history(chat_spans: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    expected_count = 0
    ordered = sorted(
        chat_spans, key=lambda s: (s["start_time"], s["attributes"].get("gen_ai.input.messages.offset", 0))
    )
    for span in ordered:
        attributes = span["attributes"]
        offset = attributes.get("gen_ai.input.messages.offset", 0)
        delta = _strip_provider_fields(_json_attribute(attributes, "gen_ai.input.messages", []))
        expected_count = max(expected_count, attributes.get("gen_ai.input.messages.total_count", 0))
        if offset > len(history):
            raise TraceError(f"{label}: delta offset {offset} skips history of {len(history)} messages")
        if history[offset : offset + len(delta)] != delta[: len(history) - offset]:
            raise TraceError(f"{label}: delta at offset {offset} contradicts earlier messages")
        if offset + len(delta) > len(history):
            history = history[:offset] + delta
    if len(history) != expected_count:
        raise TraceError(f"{label}: reconstructed {len(history)} messages, expected {expected_count}")
    return history


def extract_subagents(trace_path: Path, run_id: str) -> dict[str, dict[str, Any]]:
    """Return one teacher record per extraction subagent, keyed by subagent name.

    A subagent the orchestrator launched more than once has several entries in
    ``invocations``, in launch order. Absolute run paths are replaced by
    ``{AGENT_CONTEXT}``.
    """
    spans = [
        _with_event_payloads(json.loads(line))
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {span["context"]["span_id"]: span for span in spans}
    children: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        children.setdefault(span["parent_id"], []).append(span)

    invocations: dict[str, list[dict[str, Any]]] = {}
    for invoke in sorted(spans, key=lambda span: span["start_time"]):
        parent = by_id.get(invoke["parent_id"], {})
        launched_by_agent_tool = parent.get("attributes", {}).get("gen_ai.tool.name") == "Agent"
        if invoke["attributes"].get("gen_ai.operation.name") != "invoke_agent" or not launched_by_agent_tool:
            continue
        subagent = invoke["attributes"].get("gen_ai.agent.name")
        if not subagent:
            raise TraceError(f"invoke_agent span {invoke['context']['span_id']} has no agent name")
        label = f"{subagent}[{len(invocations.get(subagent, []))}]"
        descendants = _descendants(invoke["context"]["span_id"], children)
        invocations.setdefault(subagent, []).append(_extract_invocation(label, invoke, parent, descendants))

    records = {}
    for subagent, items in invocations.items():
        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "subagent_name": subagent,
            "invocations": items,
            "stats": _aggregate([item["stats"] for item in items]),
        }
        serialized = json.dumps(record).replace(f"/data/runs/{run_id}/agent_context", AGENT_CONTEXT_PLACEHOLDER)
        records[subagent] = json.loads(serialized)
    return records


def _extract_invocation(
    label: str, invoke: dict[str, Any], agent_tool: dict[str, Any], spans: list[dict[str, Any]]
) -> dict[str, Any]:
    chat_spans = [s for s in spans if s["attributes"].get("gen_ai.operation.name") == "chat"]
    calls = _model_calls(chat_spans, label)
    if not calls:
        raise TraceError(f"{label}: no model calls")
    tool_spans = {
        s["attributes"]["gen_ai.tool.call.id"]: s
        for s in spans
        if s["attributes"].get("gen_ai.operation.name") == "execute_tool"
    }

    first = calls[0]["attributes"]
    system_blocks = _json_attribute(first, "gen_ai.system_instructions", None)
    tools = _json_attribute(first, "gen_ai.tool.definitions", None)
    if system_blocks is None or tools is None:
        raise TraceError(f"{label}: first model call has no system instructions or tool definitions")
    system = _strip_provider_fields(
        [block for block in system_blocks if not block.get("text", "").startswith(_BILLING_HEADER_PREFIX)]
    )
    tools = _strip_provider_fields(tools)

    outputs = [_strip_provider_fields(_json_attribute(c["attributes"], "gen_ai.output.messages", [])) for c in calls]
    history = _history(chat_spans, label)
    echoed = [_signature(_parts(m)) for m in history if m.get("role") == "assistant"]
    if echoed != [_signature(output) for output in outputs[:-1]]:
        raise TraceError(f"{label}: assistant turns in the conversation do not match the model outputs")
    messages = history + [{"role": "assistant", "content": outputs[-1]}]

    tool_results = {
        part["tool_use_id"]: part
        for message in messages
        for part in _parts(message)
        if part.get("type") == "tool_result"
    }
    for output in outputs:
        for part in output:
            if part.get("type") == "tool_use" and part.get("id") not in tool_spans:
                raise TraceError(f"{label}: tool call {part.get('id')} has no execute_tool span")
    span_mismatches = 0
    for call_id, result in tool_results.items():
        span = tool_spans.get(call_id)
        if span is None:
            raise TraceError(f"{label}: tool result {call_id} has no execute_tool span")
        if _text(result.get("content")) != _text(span["attributes"].get("gen_ai.tool.call.result", "")):
            span_mismatches += 1

    tool_errors = []
    for call_id, span in sorted(tool_spans.items(), key=lambda item: item[1]["start_time"]):
        attributes = span["attributes"]
        if span["status"].get("status_code") == "ERROR" or tool_results.get(call_id, {}).get("is_error"):
            tool_errors.append(
                {
                    "tool": attributes.get("gen_ai.tool.name"),
                    "call_id": call_id,
                    "error": _text(attributes.get("gen_ai.tool.call.result", ""))[:300],
                }
            )

    last = calls[-1]["attributes"]
    stats = {
        "models": sorted({c["attributes"].get("gen_ai.request.model") for c in calls}),
        "model_calls": len(calls),
        "tool_calls": len(tool_spans),
        "tool_calls_by_name": dict(
            sorted(Counter(s["attributes"].get("gen_ai.tool.name") for s in tool_spans.values()).items())
        ),
        "tool_errors": len(tool_errors),
        "tool_result_span_mismatches": span_mismatches,
        "input_tokens": sum(c["attributes"].get("gen_ai.usage.input_tokens", 0) for c in calls),
        "cache_read_input_tokens": sum(c["attributes"].get("gen_ai.usage.cache_read_input_tokens", 0) for c in calls),
        "output_tokens": sum(c["attributes"].get("gen_ai.usage.output_tokens", 0) for c in calls),
        "final_prompt_tokens": last.get("gen_ai.usage.input_tokens", 0)
        + last.get("gen_ai.usage.cache_read_input_tokens", 0),
        "finish_reason": (last.get("gen_ai.response.finish_reasons") or [None])[-1],
        "wall_seconds": round(_seconds(invoke["start_time"], invoke["end_time"]), 3),
    }
    return {
        "agent_tool_arguments": _json_attribute(agent_tool["attributes"], "gen_ai.tool.call.arguments", {}),
        "system": system,
        "tools": tools,
        "messages": messages,
        "outputs": outputs,
        "tool_errors": tool_errors,
        "stats": stats,
    }


def _aggregate(stats: list[dict[str, Any]]) -> dict[str, Any]:
    tool_calls_by_name: Counter[str] = Counter()
    for item in stats:
        tool_calls_by_name.update(item["tool_calls_by_name"])
    summed = (
        "model_calls",
        "tool_calls",
        "tool_errors",
        "tool_result_span_mismatches",
        "input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
    )
    return {
        "invocations": len(stats),
        "models": sorted({model for item in stats for model in item["models"]}),
        **{key: sum(item[key] for item in stats) for key in summed},
        "tool_calls_by_name": dict(sorted(tool_calls_by_name.items())),
        "max_final_prompt_tokens": max(item["final_prompt_tokens"] for item in stats),
        "finish_reason": stats[-1]["finish_reason"],
        "wall_seconds": round(sum(item["wall_seconds"] for item in stats), 3),
    }
