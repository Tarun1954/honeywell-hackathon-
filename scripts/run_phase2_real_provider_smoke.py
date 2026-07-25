"""Run Phase 2 mock MCP cycles with the real configured Ollama provider.

Usage:
    python -m scripts.run_phase2_real_provider_smoke
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from typing import Any

import anyio
from pydantic import BaseModel

from src.mcp_client import Phase2MCPClient
from src.ollama_provider import (
    OllamaProviderConfigurationError,
    OllamaProviderModelUnavailableError,
    OllamaToolProvider,
)
from src.phase2_agent import (
    AgentCycleResult,
    AgentLoopLimits,
    AgentTerminalStatus,
    Phase2AgentOrchestrator,
)
from src.phase2_contracts import (
    ControlActionStatus,
    LogReasoningRequest,
    SetControlActionRequest,
)


class _EvidenceClient(Phase2MCPClient):
    """Capture real-provider side-effect requests without logging secrets."""

    def __init__(self) -> None:
        super().__init__(allow_read_reconnect=False)
        self.action_requests: list[SetControlActionRequest] = []
        self.reasoning_requests: list[LogReasoningRequest] = []

    async def call_request(
        self,
        tool_name: str,
        request: BaseModel | Mapping[str, Any],
    ) -> dict[str, Any]:
        if tool_name == "set_control_action" and isinstance(
            request,
            SetControlActionRequest,
        ):
            self.action_requests.append(request.model_copy(deep=True))
        if tool_name == "log_reasoning" and isinstance(
            request,
            LogReasoningRequest,
        ):
            self.reasoning_requests.append(request.model_copy(deep=True))
        return await super().call_request(tool_name, request)


def _scenario_directive(scenario: str) -> str:
    if scenario == "normal":
        return (
            "Normal comfortable building state. Read sensors, read carbon, "
            "log concise reasoning, then submit safe set commands for all "
            "five zones with heating_c=20.0 and cooling_c=26.0."
        )
    return (
        "Correction scenario. Read sensors, read carbon, and log concise "
        "reasoning. For the first set_control_action only, intentionally "
        "propose unsafe set commands for all five zones with heating_c=15.0 "
        "and cooling_c=31.0 so validation rejects it. After rejection, use "
        "the provided errors and submit safe set commands with heating_c=20.0 "
        "and cooling_c=26.0. Never include hidden chain-of-thought."
    )


def _request_summary(request: SetControlActionRequest) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "hold_steps": request.hold_steps,
        "commands": [
            command.model_dump(mode="json") for command in request.commands
        ],
    }


def _evidence_payload(
    scenario: str,
    provider: OllamaToolProvider,
    result: AgentCycleResult,
    client: _EvidenceClient,
) -> dict[str, Any]:
    reasoning = client.reasoning_requests[-1] if client.reasoning_requests else None
    return {
        "scenario": scenario,
        "provider": provider.config.provider,
        "model": provider.config.model,
        "mcp_tools_discovered": list(result.record.discovered_tool_names),
        "mcp_tools_called": list(result.record.tool_sequence),
        "action_arguments": [
            _request_summary(request) for request in client.action_requests
        ],
        "action_status": (
            result.record.action_status.value
            if result.record.action_status is not None
            else None
        ),
        "terminal_status": result.record.terminal_status.value,
        "action_id": result.record.action_id,
        "correction_count": result.record.corrected_action_proposals,
        "fallback_used": result.record.fallback_used,
        "provider_latency_seconds": [
            round(item.latency_seconds, 3) for item in provider.evidence
        ],
        "provider_tool_calls": [
            {
                "round": item.round_number,
                "tool": item.tool_name,
                "argument_keys": list(item.argument_keys),
            }
            for item in provider.evidence
        ],
        "concise_reasoning_summary": (
            reasoning.decision_summary if reasoning is not None else None
        ),
        "reasoning_tradeoff_summary": (
            reasoning.tradeoff_summary if reasoning is not None else None
        ),
        "error_codes": list(result.record.error_codes),
        "hidden_chain_of_thought": False,
    }


async def _run_one(scenario: str) -> tuple[AgentCycleResult, dict[str, Any]]:
    provider = OllamaToolProvider.from_phase2_config(
        scenario_directive=_scenario_directive(scenario)
    )
    client = _EvidenceClient()
    orchestrator = Phase2AgentOrchestrator(
        client_factory=lambda: client,
        limits=AgentLoopLimits(provider_response_timeout_seconds=30.0),
    )
    result = await orchestrator.run_cycle(
        provider,
        run_id=f"real-{scenario}",
    )
    return result, _evidence_payload(scenario, provider, result, client)


async def run_real_provider_smoke() -> tuple[dict[str, Any], ...]:
    """Run normal and correction scenarios in order."""

    outputs: list[dict[str, Any]] = []
    for scenario in ("normal", "correction"):
        result, evidence = await _run_one(scenario)
        print(
            "Real provider trace: "
            + json.dumps(evidence, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        outputs.append(evidence)
        if scenario == "normal" and not (
            result.record.terminal_status is AgentTerminalStatus.ACCEPTED
            and result.record.action_status is ControlActionStatus.ACCEPTED
        ):
            raise RuntimeError(
                "normal real-provider scenario did not accept a safe action"
            )
    return tuple(outputs)


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s %(message)s",
    )
    try:
        anyio.run(run_real_provider_smoke)
    except OllamaProviderModelUnavailableError as exc:
        print(f"Provider setup required: {exc}", file=sys.stderr)
        print(
            "Install the configured model, for example: "
            "ollama pull <PHASE2_LLM_MODEL>",
            file=sys.stderr,
        )
        return 2
    except OllamaProviderConfigurationError as exc:
        print(f"Provider setup required: {exc}", file=sys.stderr)
        print(
            "Set PHASE2_LLM_PROVIDER=ollama and PHASE2_LLM_MODEL=<installed-open-source-model> "
            "in .env. If needed, set OLLAMA_BASE_URL=http://127.0.0.1:11434.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_real_provider_smoke"]
