"""Tests for cache-safe natural-language task compilation."""

from __future__ import annotations

from agent.task_compiler import (
    append_contract_to_user_message,
    classify_task,
    compile_task_contract,
    task_compiler_enabled,
)


def test_task_compiler_auto_enables_for_mini_only():
    cfg = {"agent": {"task_compiler": "auto"}}
    assert task_compiler_enabled(cfg, model="gpt-5.4-mini") is True
    assert task_compiler_enabled(cfg, model="gpt-5.5") is False


def test_neuralwatt_prompt_gets_strict_execution_contract_for_mini():
    cfg = {"agent": {"task_compiler": "auto"}}

    compiled = compile_task_contract(
        "Check whether NeuralWatt GLM-5.2 is configured and working on this machine. Keep it brief.",
        model="gpt-5.4-mini",
        provider="openai-codex",
        config=cfg,
    )

    assert compiled.enabled is True
    assert compiled.level == 3
    assert compiled.intent == "execution"
    assert "Do the action before reporting it" in compiled.text
    assert "Execution order:" in compiled.text
    assert "Verify with real tool output" in compiled.text
    assert "Small/literal-model strictness" in compiled.text


def test_research_prompt_gets_light_contract_when_forced_on():
    compiled = compile_task_contract(
        "Analyze this megathread for implementations we should mine.",
        model="gpt-5.5",
        config={"agent": {"task_compiler": True}},
    )

    assert compiled.enabled is True
    assert compiled.level == 1
    assert compiled.intent == "research"
    assert "Start with the conclusion" in compiled.text


def test_single_word_execution_terms_match_words_not_substrings():
    implementation_reference = compile_task_contract(
        "Analyze this megathread for implementations we should mine.",
        model="gpt-5.5",
        config={"agent": {"task_compiler": True}},
    )
    implementation_request = compile_task_contract(
        "Implement the fix and verify it.",
        model="gpt-5.5",
        config={"agent": {"task_compiler": True}},
    )

    assert implementation_reference.level == 1
    assert implementation_reference.intent == "research"
    assert implementation_request.level == 2
    assert implementation_request.intent == "execution"


def test_destructive_prompt_adds_approval_gate():
    level, intent, sensitivity, approval = classify_task(
        "Delete the stale production credential and deploy prod",
        model="gpt-5.4-mini",
    )

    assert level == 4
    assert intent == "execution"
    assert sensitivity == "private-or-sensitive"
    assert approval is True

    compiled = compile_task_contract(
        "Delete the stale production credential and deploy prod",
        model="gpt-5.4-mini",
        config={"agent": {"task_compiler": True}},
    )
    assert "Approval gate:" in compiled.text
    assert "without explicit user approval" in compiled.text


def test_append_contract_does_not_mutate_original_string():
    compiled = compile_task_contract(
        "Fix the failing test and verify it.",
        model="gpt-5.4-mini",
        config={"agent": {"task_compiler": True}},
    )
    original = "Fix the failing test and verify it."

    result = append_contract_to_user_message(original, compiled)

    assert original == "Fix the failing test and verify it."
    assert result.startswith(original)
    assert "[Hermes execution contract for this turn]" in result


def test_append_contract_to_multimodal_blocks_appends_text_block():
    compiled = compile_task_contract(
        "Verify this screenshot.",
        model="gpt-5.4-mini",
        config={"agent": {"task_compiler": True}},
    )
    original = [{"type": "text", "text": "Verify this screenshot."}]

    result = append_contract_to_user_message(original, compiled)

    assert original == [{"type": "text", "text": "Verify this screenshot."}]
    assert result is not original
    assert result[-1]["type"] == "text"
    assert "execution contract" in result[-1]["text"]
