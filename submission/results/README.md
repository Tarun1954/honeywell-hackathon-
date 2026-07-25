# Submitted result artifacts

The result files are copied from the tracked `runs/final/` evidence directory.

## Real Ollama comparison

- `live_ollama_smoke_report.json` (verified four-hour live smoke evidence used
  by the explicitly labeled demo replay mode)
- `ollama_24h_comparison.json`
- `ollama_24h_comparison.csv`
- `ollama_24h_energy_peak.png`
- `ollama_24h_comfort.png`
- `ollama_24h_actions_latency.png`

These files describe the matched **24-hour real Ollama-supervised hybrid
comparison**.

## Deterministic and ScriptedProvider evidence

- `results_summary.json`
- `results_summary.csv`
- `energy_comparison.png`
- `zone_temperature_pmv.png`
- `action_setpoint_evidence.png`

The aggregate summary was generated before the separate 24-hour Ollama
comparison and therefore retains a `real_ollama: pending` field. That field
applies only to this earlier aggregation snapshot. The current real-provider
result is the separately labeled `ollama_24h_comparison.*` pair above. The
seven-day 6.747% result remains deterministic Phase 1 evidence and is not an
Ollama savings claim.
