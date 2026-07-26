# Simulated 24-hour electricity cost estimate

This is **offline post-processing of the existing verified comparison**. It did
not rerun or modify EnergyPlus, Ollama, MCP, the agent loop, safety validation,
or the existing comparison artifacts.

## Assumption

- Geography/sector: **Illinois commercial**
- Energy-price proxy: **15.36 cents/kWh**
- Observation period: **May 2026**
- Source publication date: **2026-07-23**
- Source: [U.S. Energy Information Administration - Electric Power Monthly Table 5.6.A, Average Price of Electricity to Ultimate Customers by End-Use Sector, by State](https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a)

The EIA value is a statewide monthly average retail price, not a utility rate
schedule for this simulated building.

## Energy-only estimate

| Metric | Baseline | Ollama hybrid |
| --- | ---: | ---: |
| Verified electricity | 180.556 kWh | 168.703 kWh |
| Simulated energy cost | $27.733 | $25.913 |

- Electricity reduction: **11.853 kWh**
- Simulated energy-cost savings: **$1.821 (6.565%)**

## Demand charge

Demand-cost status: **not_estimated**.

A 24-hour simulated peak is not a utility billing demand and no specific monthly tariff was selected.

The verified baseline/controlled peaks remain
**17.258/15.566 kW**, but
no currency value is assigned to them.

## Required disclaimer

**This is a simulated cost estimate based on a statewide average commercial retail-price proxy and verified 24-hour EnergyPlus results. It is not an actual tariff calculation or utility bill.**
