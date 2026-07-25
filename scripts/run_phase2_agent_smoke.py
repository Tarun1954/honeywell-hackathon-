"""Run three deterministic Phase 2 agent cycles over real MCP stdio.

Usage:
    python -m scripts.run_phase2_agent_smoke
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import anyio

from src.phase2_agent import AgentCycleResult, Phase2AgentOrchestrator
from src.phase2_mock_services import PHASE1_ZONE_IDS
from src.scripted_provider import ScriptedProvider, ScriptedToolCall


def _reasoning_arguments(summary: str) -> dict[str, Any]:
    return {
        "decision_summary": summary,
        "objective_tags": [
            "thermal_comfort",
            "energy_reduction",
            "safety",
        ],
        "tradeoff_summary": (
            "Preserve configured comfort while avoiding unnecessary demand."
        ),
        "confidence": 0.95,
    }


def _set_commands(
    heating_c: float,
    cooling_c: float,
) -> list[dict[str, Any]]:
    return [
        {
            "mode": "set",
            "zone_id": zone_id,
            "heating_c": heating_c,
            "cooling_c": cooling_c,
        }
        for zone_id in PHASE1_ZONE_IDS
    ]


def _base_calls(prefix: str) -> list[ScriptedToolCall]:
    return [
        ScriptedToolCall(
            call_id=f"{prefix}-read",
            tool_name="read_sensor_data",
            arguments={"history_steps": 2},
        ),
        ScriptedToolCall(
            call_id=f"{prefix}-carbon",
            tool_name="get_grid_carbon_intensity",
            arguments={"forecast_steps": 4},
        ),
        ScriptedToolCall(
            call_id=f"{prefix}-reasoning",
            tool_name="log_reasoning",
            arguments=_reasoning_arguments(
                "Use a deterministic five-zone thermostat action."
            ),
        ),
    ]


def build_success_provider() -> ScriptedProvider:
    calls = _base_calls("success")
    calls.append(
        ScriptedToolCall(
            call_id="success-action",
            tool_name="set_control_action",
            arguments={
                "commands": _set_commands(20.0, 26.0),
                "hold_steps": 1,
            },
        )
    )
    return ScriptedProvider(calls, name="successful-script")


def build_correction_provider() -> ScriptedProvider:
    calls = _base_calls("correction")
    calls.extend(
        (
            ScriptedToolCall(
                call_id="correction-unsafe",
                tool_name="set_control_action",
                arguments={
                    "commands": _set_commands(15.0, 31.0),
                    "hold_steps": 1,
                },
            ),
            ScriptedToolCall(
                call_id="correction-safe",
                tool_name="set_control_action",
                arguments={
                    "commands": _set_commands(20.0, 26.0),
                    "hold_steps": 1,
                },
            ),
        )
    )
    return ScriptedProvider(calls, name="correction-script")


def build_fallback_provider() -> ScriptedProvider:
    calls = _base_calls("fallback")
    for index, (heating_c, cooling_c) in enumerate(
        ((15.0, 31.0), (15.5, 30.5), (18.0, 18.5)),
        start=1,
    ):
        calls.append(
            ScriptedToolCall(
                call_id=f"fallback-unsafe-{index}",
                tool_name="set_control_action",
                arguments={
                    "commands": _set_commands(
                        heating_c,
                        cooling_c,
                    ),
                    "hold_steps": 1,
                },
            )
        )
    return ScriptedProvider(calls, name="fallback-script")


def _print_trace(label: str, result: AgentCycleResult) -> None:
    action_outcomes = _action_outcomes(result)
    payload = {
        "tools": list(result.record.tool_sequence),
        "action_outcomes": action_outcomes,
        "terminal_status": result.record.terminal_status.value,
        "action_status": (
            result.record.action_status.value
            if result.record.action_status is not None
            else None
        ),
        "action_id": result.record.action_id,
        "corrected_action_proposals": (
            result.record.corrected_action_proposals
        ),
        "fallback_used": result.record.fallback_used,
        "rounds": result.record.rounds_used,
        "tool_calls": result.record.tool_calls_used,
    }
    print(
        f"{label} trace: "
        + json.dumps(payload, separators=(",", ":"), sort_keys=True),
        flush=True,
    )


def _action_outcomes(result: AgentCycleResult) -> list[str]:
    return [
        event.status
        for event in result.trace
        if event.tool_name == "set_control_action"
        and event.status in {"accepted", "rejected", "duplicate"}
    ]


def _validate_smoke_result(
    label: str,
    result: AgentCycleResult,
) -> None:
    expected = {
        "successful-cycle": ("accepted", ["accepted"], False),
        "correction": (
            "accepted",
            ["rejected", "accepted"],
            False,
        ),
        "fallback": (
            "fallback_accepted",
            ["rejected", "rejected", "rejected", "accepted"],
            True,
        ),
    }
    expected_terminal, expected_outcomes, expected_fallback = expected[label]
    if (
        result.record.terminal_status.value != expected_terminal
        or _action_outcomes(result) != expected_outcomes
        or result.record.fallback_used is not expected_fallback
        or result.record.action_id is None
    ):
        raise RuntimeError(
            f"{label} did not meet its deterministic smoke expectations"
        )


async def run_smoke_scenarios() -> tuple[AgentCycleResult, ...]:
    """Run success, correction, and fallback in isolated stdio cycles."""

    orchestrator = Phase2AgentOrchestrator()
    scenarios = (
        ("successful-cycle", build_success_provider()),
        ("correction", build_correction_provider()),
        ("fallback", build_fallback_provider()),
    )
    results: list[AgentCycleResult] = []
    for label, provider in scenarios:
        result = await orchestrator.run_cycle(
            provider,
            run_id=label,
        )
        _print_trace(label, result)
        _validate_smoke_result(label, result)
        results.append(result)
    return tuple(results)


def main() -> None:
    """Configure diagnostics on stderr and execute all three scenarios."""

    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s %(message)s",
    )
    anyio.run(run_smoke_scenarios)


if __name__ == "__main__":
    main()


__all__ = [
    "build_correction_provider",
    "build_fallback_provider",
    "build_success_provider",
    "run_smoke_scenarios",
]
