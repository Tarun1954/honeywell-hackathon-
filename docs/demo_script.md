# Eco-Loop three-minute demo script

Target duration: **2 minutes 55 seconds**. The spoken narration below is
approximately 330 words. At a measured 125-135 words per minute, it remains
within the three-minute limit. Rehearse once with a timer and do not pause to
debug during recording.

Recommended command:

```powershell
python -m scripts.run_final_demo --mode auto
```

For the most reliable recording when local services are uncertain:

```powershell
python -m scripts.run_final_demo --mode replay
```

Replay mode is explicitly labeled as previously captured verified evidence and
must never be described as live.

## 0:00-0:15 — Problem and result

On screen: README title and the dashboard headline.

> Eco-Loop connects EnergyPlus to a local Ollama model through MCP, with a
> hard safety boundary. In a matched 24-hour run it reduced electricity by
> 6.565 percent and peak demand from 17.258 to 15.566 kilowatts.

## 0:15-0:35 — Architecture

On screen: `docs/system_architecture.md` at the Mermaid diagram.

> Every 15 simulated minutes, EnergyPlus supplies five-zone temperatures, PMV,
> occupancy, setpoints, demand, and energy. An adapter builds a typed snapshot.
> MCP exposes five tools. Ollama proposes an hourly action; validation decides
> whether it is safe; only accepted commands reach the existing actuators.

## 0:35-0:50 — Start the workflow

On screen: terminal. Run the launcher.

> I am starting the demo launcher. Auto mode uses the proven live four-hour
> smoke. If Ollama, the model, or EnergyPlus is unavailable, it clearly
> switches to saved verified evidence instead of pretending the replay is live.

## 0:50-1:18 — Live sensors

On screen: launcher sections 1 and 2.

> EnergyPlus has started. Here are the simulated timestamp and snapshot ID.
> Each zone shows temperature, Fanger PMV, and occupancy, followed by facility
> demand and energy. These are live API values entering the agent, or, when
> labeled replay, the exact verified captured values.

## 1:18-1:43 — MCP and Ollama

On screen: launcher sections 3 and 4.

> MCP discovers exactly five tools: sensor reading, grid carbon, concise
> reasoning logging, control submission, and runtime-error parsing. The trace
> shows every tool called in order. Ollama selects a complete five-zone action.
> The display summarizes the action without exposing hidden chain-of-thought.

## 1:43-2:10 — Safety and writeback

On screen: launcher sections 5 through 7.

> Before writeback, safety checks zone coverage, bounds, occupied limits,
> deadband, snapshot freshness, hold duration, and runtime errors. Unsafe
> values are rejected, never clamped. Here the action is accepted. We see each
> setpoint change, five successful actuator writes, and zero severe or fatal
> errors. On failure, deterministic fallback restores the original schedules.

## 2:10-2:50 — Quantitative results and trade-off

On screen: launcher section 8, then the three dashboard charts.

> The matched result uses the same model, weather, timestep, and initial state.
> Baseline electricity was 180.556 kilowatt-hours; the Ollama hybrid used
> 168.703, saving 11.853. All 24 hourly actions were accepted without fallback.
> The trade-off is visible: PMV compliance decreased from 96.82 to 95.45
> percent, and comfort violations rose from seven to ten. The separate
> seven-day 6.747 percent result is deterministic Phase 1, not Ollama.

## 2:50-2:55 — Close

On screen: dashboard title.

> Eco-Loop demonstrates local AI supervision with measurable results,
> auditable tools, validated actions, and a deterministic safe fallback.
