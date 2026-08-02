"""Compile natural-language user turns into cache-safe execution contracts.

The task compiler is intentionally deterministic and lightweight. It does not
rewrite the stored transcript or the cached system prompt; callers append the
compiled contract to the current API request's user message only. That makes it
safe for long-lived conversations and prompt caching while still giving literal
small models a concrete execution contract for the current turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


_EXECUTION_TERMS = {
    "check",
    "fix",
    "debug",
    "test",
    "verify",
    "audit",
    "build",
    "implement",
    "configure",
    "route",
    "routing",
    "deploy",
    "run",
    "investigate",
    "triage",
    "install",
    "setup",
    "ship",
    "validate",
}

_DESTRUCTIVE_TERMS = {
    "delete",
    "destroy",
    "wipe",
    "reset",
    "remove credential",
    "drop table",
    "purge",
    "revoke",
    "publish",
    "post",
    "purchase",
    "buy",
    "charge",
    "merge",
    "deploy prod",
    "production deploy",
}

_PRIVACY_TERMS = {
    "secret",
    "token",
    "api key",
    "credential",
    "auth",
    "private",
    "personal",
    "customer",
    "financial",
    "strategy",
    "proprietary",
    "repo",
    "codebase",
}

_RESEARCH_TERMS = {
    "research",
    "summarize",
    "compare",
    "analyze",
    "find",
    "source",
    "evidence",
    "what do we know",
}

_MINI_MODEL_RE = re.compile(r"(?:^|[-_/])(?:gpt[-_])?5\.4[-_]?mini(?:$|[-_/])|\bmini\b", re.I)


@dataclass(frozen=True)
class CompiledTask:
    """A compiled current-turn contract."""

    enabled: bool
    level: int
    text: str
    intent: str
    sensitivity: str
    requires_approval: bool


def _contains_any(text: str, terms: set[str]) -> bool:
    """Return True when text contains any term.

    Single-word terms match on token boundaries so verbs such as "implement"
    do not fire on nouns like "implementations". Multi-word phrases still use
    substring matching because they are natural-language fragments rather than
    standalone tokens.
    """
    lowered = text.lower()
    for term in terms:
        normalized = term.lower().strip()
        if not normalized:
            continue
        if re.search(r"\s", normalized):
            if normalized in lowered:
                return True
            continue
        if re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", lowered):
            return True
    return False


def _is_mini_model(model: str) -> bool:
    return bool(_MINI_MODEL_RE.search(model or ""))


def _enabled_from_config(config: Mapping[str, Any] | None) -> bool | str:
    if not isinstance(config, Mapping):
        return False
    raw_agent_cfg = config.get("agent")
    agent_cfg = raw_agent_cfg if isinstance(raw_agent_cfg, Mapping) else {}
    value = agent_cfg.get("task_compiler", False)
    return value


def task_compiler_enabled(config: Mapping[str, Any] | None, *, model: str = "") -> bool:
    """Return whether the task compiler should run for this model/config.

    Config values:
    - False / "off" / "disabled": disabled
    - True / "on": enabled
    - "auto": enabled for GPT-5.4-mini-like models only
    """
    value = _enabled_from_config(config)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "off", "false", "disabled", "none"}:
            return False
        if normalized == "auto":
            return _is_mini_model(model)
        return normalized in {"on", "true", "enabled"}
    return bool(value)


def classify_task(user_message: str, *, model: str = "") -> tuple[int, str, str, bool]:
    """Classify the prompt into scaffold level, intent, sensitivity, approval.

    Levels are deliberately coarse:
    0: no contract
    1: light answer/research structure
    2: execution contract
    3: strict autonomous contract for literal/small models
    4: approval-gated contract
    """
    text = user_message or ""
    lower = text.lower()
    requires_approval = _contains_any(lower, _DESTRUCTIVE_TERMS)
    sensitive = _contains_any(lower, _PRIVACY_TERMS)
    execution = _contains_any(lower, _EXECUTION_TERMS)
    research = _contains_any(lower, _RESEARCH_TERMS)

    if execution:
        intent = "execution"
        level = 2
    elif research:
        intent = "research"
        level = 1
    else:
        intent = "answer"
        level = 0

    if level >= 2 and _is_mini_model(model):
        level = 3
    if requires_approval:
        level = 4

    sensitivity = "private-or-sensitive" if sensitive else "low"
    return level, intent, sensitivity, requires_approval


def compile_task_contract(
    user_message: str,
    *,
    model: str = "",
    provider: str = "",
    platform: str = "",
    config: Mapping[str, Any] | None = None,
) -> CompiledTask:
    """Compile a current-turn execution contract for ``user_message``.

    Returns ``enabled=False`` when config/model heuristics say not to inject or
    when the prompt is too simple to need scaffolding.
    """
    if not task_compiler_enabled(config, model=model):
        return CompiledTask(False, 0, "", "disabled", "low", False)

    level, intent, sensitivity, requires_approval = classify_task(user_message, model=model)
    if level <= 0:
        return CompiledTask(False, level, "", intent, sensitivity, requires_approval)

    strict = level >= 3
    lines: list[str] = [
        "[Hermes execution contract for this turn]",
        "Purpose: convert the user's natural-language request into an executable task without changing their intent.",
        f"Intent: {intent}",
        f"Sensitivity: {sensitivity}",
        "",
    ]

    if level == 1:
        lines.extend([
            "Goal:",
            "- Answer the user's core question with a clear conclusion and evidence-backed reasoning.",
            "Output:",
            "- Start with the conclusion, then the key evidence/tradeoffs, then the recommended next step if any.",
            "Stop rules:",
            "- Do not ask a follow-up unless a missing fact materially changes the answer.",
        ])
    else:
        lines.extend([
            "Goal:",
            "- Complete the user's requested outcome, not merely describe a plan.",
            "Success criteria:",
            "- The needed discovery/checks/actions are performed before the final response when safe and available.",
            "- The final response distinguishes verified results from blockers or assumptions.",
            "Critical rules:",
            "1. Use available tools for live state, files, commands, web/current facts, and verification; do not rely on memory for those.",
            "2. Do the action before reporting it. Do not end with a promise to act later if tools can make progress now.",
            "3. If you modify or build something, run the most relevant validation available before claiming success.",
            "4. Preserve the user's original intent; this contract is scaffolding, not a replacement request.",
            "Execution order:",
            "1. Identify the source of truth and inspect it.",
            "2. Take the smallest safe action that advances the requested outcome.",
            "3. Verify with real tool output or a concrete artifact.",
            "4. Report outcome first, then concise evidence and any remaining blocker.",
            "Decision rules:",
            "- Ask for clarification only when ambiguity changes which system/account/file to touch or creates meaningful risk.",
            "- If a tool or path fails, try one reasonable alternate route before stopping.",
            "- If evidence is missing after reasonable attempts, say exactly what is missing and why.",
        ])
        if sensitivity == "private-or-sensitive":
            lines.extend([
                "Privacy/routing rule:",
                "- Do not send private, credential, proprietary, customer, financial, or strategy content to low-trust/non-approved external lanes unless explicitly approved.",
            ])
        if strict:
            lines.extend([
                "Small/literal-model strictness:",
                "- Follow the execution order literally.",
                "- Prefer concrete tool calls and explicit verification over inference.",
                "- Do not keep the conversation going with an unnecessary follow-up question.",
            ])
        if requires_approval:
            lines.extend([
                "Approval gate:",
                "- Do not execute irreversible/destructive/external side effects such as publishing, purchasing, credential deletion, production deploys, destructive resets, or merge-to-main without explicit user approval.",
                "- Prepare the exact proposed action/payload and ask for approval instead.",
            ])
        lines.extend([
            "Output:",
            "- Be concise. Include completed actions, verification evidence, and blockers/next action if any.",
        ])

    lines.append("[/Hermes execution contract]")
    return CompiledTask(True, level, "\n".join(lines), intent, sensitivity, requires_approval)


def append_text_to_user_message(content: Any, text: str) -> Any:
    """Return user content with API-only text appended.

    Handles the normal string-content path and multimodal list content. The
    caller should use this only on copied API messages, never on persisted
    transcript messages.
    """
    if not text:
        return content
    if isinstance(content, str):
        return content + "\n\n" + text
    if isinstance(content, list):
        blocks = list(content)
        blocks.append({"type": "text", "text": text})
        return blocks
    return content


def append_contract_to_user_message(content: Any, contract: CompiledTask) -> Any:
    """Return API-only user content with the compiled contract appended."""
    if not contract.enabled or not contract.text:
        return content
    return append_text_to_user_message(content, contract.text)
