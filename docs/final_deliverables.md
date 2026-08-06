# Final deliverables checklist

Status reflects repository contents at this documentation checkpoint.

- [x] Source code: EnergyPlus wrapper, live adapter, MCP, agent, providers,
  validation, runners, aggregation, and tests
- [x] Baseline and API-ready EnergyPlus files:
  [`models/baseline.idf`](../models/baseline.idf) and
  [`models/optimized_runtime.idf`](../models/optimized_runtime.idf). The
  controlled run uses live API actuator overrides; no permanently modified
  controlled IDF exists. See
  [`submission/models/MODEL_NOTES.md`](../submission/models/MODEL_NOTES.md).
- [x] MCP implementation:
  [`src/mcp_server.py`](../src/mcp_server.py) and
  [`src/mcp_client.py`](../src/mcp_client.py)
- [x] Real Ollama evidence:
  [`runs/final/ollama_24h_comparison.json`](../runs/final/ollama_24h_comparison.json)
  and the [24-hour report](ollama_24h_comparison.md)
- [x] JSON/CSV results:
  [`runs/final/ollama_24h_comparison.json`](../runs/final/ollama_24h_comparison.json),
  [`runs/final/ollama_24h_comparison.csv`](../runs/final/ollama_24h_comparison.csv),
  [`runs/final/results_summary.json`](../runs/final/results_summary.json), and
  [`runs/final/results_summary.csv`](../runs/final/results_summary.csv)
- [x] Charts and static dashboard: tracked PNGs under `runs/final/` and
  [`docs/dashboard/index.html`](dashboard/index.html)
- [x] Project README: [`README.md`](../README.md)
- [x] Architecture report:
  [`docs/system_architecture.md`](system_architecture.md)
- [x] Video: submitted separately through the assessment portal; intentionally
  not stored in this repository
- [x] Presentation: submitted separately through the assessment portal;
  intentionally not stored in this repository
- [x] GitHub URL:
  [github.com/Tarun1954/honeywell-hackathon-](https://github.com/Tarun1954/honeywell-hackathon-)
