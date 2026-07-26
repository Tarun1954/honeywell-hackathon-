"use strict";

const state = {
  comparison: null,
  cost: null,
  summary: null,
  events: [],
  eventIndex: 0,
  replayTimer: null,
  liveTimer: null,
};

const byId = (id) => document.getElementById(id);
const number = (value, digits = 3) =>
  Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "--";
const money = (value) =>
  Number.isFinite(Number(value)) ? `$${Number(value).toFixed(3)}` : "--";
const titleCase = (value) =>
  String(value || "--")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

async function getJson(path) {
  const response = await fetch(path, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json();
}

function activatePage(pageName) {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.page === pageName);
  });
  document.querySelectorAll(".page").forEach((page) => {
    page.classList.toggle("active", page.id === `page-${pageName}`);
  });
  window.history.replaceState(null, "", `#${pageName}`);
}

function populateOverview() {
  const comparison = state.comparison;
  if (!comparison) return;
  const electricity = comparison.electricity;
  const peak = comparison.peak_demand;
  const actions = comparison.control_actions;
  const health = comparison.energyplus.controlled;

  byId("baselineEnergy").textContent = number(electricity.baseline_total_kwh);
  byId("controlledEnergy").textContent = number(
    electricity.controlled_total_kwh,
  );
  byId("energyReduction").textContent = number(
    electricity.absolute_difference_kwh,
  );
  byId("energyReductionPercent").textContent =
    `${number(Math.abs(electricity.percentage_difference_from_baseline))}% reduction`;
  byId("peakReduction").textContent = number(
    (peak.baseline_w - peak.controlled_w) / 1000,
  );
  byId("exitStatus").textContent = String(health.exit_status);
  byId("severeErrors").textContent = String(health.severe_count);
  byId("fatalErrors").textContent = String(health.fatal_count);
  byId("actuatorWrites").textContent = String(
    actions.successful_actuator_zone_writes,
  );
}

function populateCost() {
  const cost = state.cost;
  if (!cost) return;
  const estimate = cost.energy_cost_estimate;
  const assumptions = cost.assumptions;
  byId("baselineCost").textContent = money(estimate.baseline_usd);
  byId("controlledCost").textContent = money(estimate.controlled_usd);
  byId("costSavings").textContent = money(estimate.savings_usd);
  byId("costSavingsPercent").textContent =
    `${number(estimate.savings_percent)}% energy-only savings`;
  byId("tariffRate").textContent =
    `${number(assumptions.energy_rate_cents_per_kwh, 2)} cents/kWh`;
  byId("tariffGeography").textContent = assumptions.geography;
  byId("tariffSector").textContent = titleCase(assumptions.customer_sector);
  byId("tariffPeriod").textContent = assumptions.observation_period;
  byId("costDisclaimer").textContent = cost.disclaimer;
  byId("tariffSource").href = assumptions.source_url;
}

function renderEvidence(evidence) {
  const list = byId("evidenceList");
  list.replaceChildren();
  evidence.entries.forEach((entry) => {
    const item = document.createElement("article");
    item.className = `evidence-item${entry.available ? "" : " unavailable"}`;
    const copy = document.createElement("div");
    const label = document.createElement("strong");
    label.textContent = entry.label;
    const status = document.createElement("small");
    status.textContent = entry.available
      ? "Verified artifact available"
      : "Artifact is not currently available";
    copy.append(label, status);
    item.append(copy);
    if (entry.available) {
      const link = document.createElement("a");
      link.href = entry.download_url;
      link.textContent = "Download";
      item.append(link);
    }
    list.append(item);
  });
}

function setLiveEvents(payload) {
  state.events = payload.events || [];
  state.eventIndex = Math.max(0, state.events.length - 1);
  byId("liveEvidenceBadge").textContent = payload.source_label;
  renderCurrentEvent();
}

async function loadDateOptions() {
  const mode = byId("viewMode").value;
  const runId = byId("runSelection").value;
  const dateSelect = byId("evidenceDate");
  const previousDate = dateSelect.value;
  const payload = await getJson(
    `/api/live/dates?run_id=${encodeURIComponent(runId)}&mode=${encodeURIComponent(mode)}`,
  );
  dateSelect.replaceChildren();
  const allDates = document.createElement("option");
  allDates.value = "";
  allDates.textContent = payload.available_dates.length
    ? "All available dates"
    : "No dated evidence";
  dateSelect.append(allDates);
  payload.available_dates.forEach((date) => {
    const option = document.createElement("option");
    option.value = date;
    option.textContent = new Date(`${date}T00:00:00Z`).toLocaleDateString([], {
      dateStyle: "long",
      timeZone: "UTC",
    });
    dateSelect.append(option);
  });
  dateSelect.disabled = payload.available_dates.length === 0;
  dateSelect.value = payload.available_dates.includes(previousDate)
    ? previousDate
    : "";
}

function renderCurrentEvent() {
  const total = state.events.length;
  const event = total ? state.events[state.eventIndex] : null;
  const count = total ? state.eventIndex + 1 : 0;
  byId("timelineCount").textContent = `${count} / ${total}`;
  byId("timelineProgress").style.width =
    total > 0 ? `${(count / total) * 100}%` : "0%";

  if (!event) {
    byId("snapshotTimestamp").textContent = "No live evidence for this selection";
    byId("snapshotId").textContent = "--";
    byId("zoneRows").replaceChildren();
    return;
  }

  const timestamp = new Date(event.simulated_timestamp);
  byId("snapshotTimestamp").textContent = Number.isNaN(timestamp.valueOf())
    ? event.simulated_timestamp
    : timestamp.toLocaleString([], {
        dateStyle: "medium",
        timeStyle: "short",
        timeZone: "UTC",
      });
  byId("snapshotId").textContent = event.snapshot_id;
  byId("facilityDemand").textContent =
    `${number(event.facility_electricity_demand_w / 1000)} kW`;
  byId("facilityEnergy").textContent =
    `${number(event.facility_electricity_kwh_since_start)} kWh`;

  const zoneNames = Object.keys(event.zone_temperatures_c || {});
  const occupiedCount = zoneNames.filter(
    (zone) => Number(event.occupancy?.[zone] || 0) > 0,
  ).length;
  byId("occupiedZones").textContent = `${occupiedCount} / ${zoneNames.length}`;

  const zoneRows = byId("zoneRows");
  zoneRows.replaceChildren();
  zoneNames.forEach((zone) => {
    const row = document.createElement("tr");
    const occupancy = Number(event.occupancy?.[zone] || 0);
    const chosen = event.chosen_setpoints?.[zone];
    const previous = event.previous_setpoints?.[zone];
    const setpoints = chosen || previous || {};
    const values = [
      zone,
      `${number(event.zone_temperatures_c[zone], 2)} °C`,
      number(event.pmv?.[zone], 2),
      occupancy > 0 ? "Occupied" : "Unoccupied",
      `${number(setpoints.heating_c, 1)} / ${number(setpoints.cooling_c, 1)} °C`,
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === 3) {
        cell.className = occupancy > 0 ? "occupied" : "unoccupied";
      }
      row.append(cell);
    });
    zoneRows.append(row);
  });

  const actionStatus = String(event.action_status || "no_action");
  byId("actionStatus").textContent = titleCase(actionStatus);
  byId("actionStatus").className = `action-status ${actionStatus}`;
  const selectedZones = Object.keys(event.chosen_setpoints || {}).length;
  byId("decisionSummary").textContent =
    selectedZones > 0
      ? `Validated setpoint action for ${selectedZones} zones`
      : "No hourly control action at this timestep";
  byId("providerName").textContent = event.provider_name || "No provider call";
  byId("fallbackUsed").textContent = event.fallback_used ? "Used" : "No";
  byId("correctionCount").textContent = String(
    event.corrected_action_count ?? 0,
  );
  byId("writeResult").textContent = titleCase(
    event.actuator_write_result?.status || "not available",
  );

  const toolList = byId("toolList");
  toolList.replaceChildren();
  const tools = event.mcp_tools_called || [];
  if (tools.length === 0) {
    const empty = document.createElement("span");
    empty.className = "empty-state";
    empty.textContent = "No MCP call at this timestep";
    toolList.append(empty);
  } else {
    tools.forEach((tool) => {
      const token = document.createElement("span");
      token.className = "tool-token";
      token.textContent = tool;
      toolList.append(token);
    });
  }
}

function stopPlayback() {
  if (state.replayTimer) window.clearInterval(state.replayTimer);
  if (state.liveTimer) window.clearInterval(state.liveTimer);
  state.replayTimer = null;
  state.liveTimer = null;
}

async function loadLiveEvidence() {
  stopPlayback();
  const mode = byId("viewMode").value;
  const runId = byId("runSelection").value;
  const selectedDate = byId("evidenceDate").value;
  const dateQuery = selectedDate
    ? `&date=${encodeURIComponent(selectedDate)}`
    : "";
  const payload = await getJson(
    `/api/live/events?run_id=${encodeURIComponent(runId)}&mode=${encodeURIComponent(mode)}${dateQuery}`,
  );
  setLiveEvents(payload);

  byId("modeNote").textContent =
    mode === "live"
      ? "Read-only polling of the latest recorded run artifact. No simulation or model is started."
      : "Replays a verified artifact. It does not start EnergyPlus or Ollama.";

  if (mode === "live") {
    state.liveTimer = window.setInterval(async () => {
      try {
        const latest = await getJson(
          `/api/live/latest?run_id=${encodeURIComponent(runId)}&mode=live${dateQuery}`,
        );
        if (latest.events?.length) setLiveEvents(latest);
      } catch {
        byId("connectionStatus").textContent = "Evidence refresh unavailable";
      }
    }, 4000);
  }
}

function startReplay() {
  stopPlayback();
  if (state.events.length === 0) return;
  state.eventIndex = 0;
  renderCurrentEvent();
  state.replayTimer = window.setInterval(() => {
    if (state.eventIndex >= state.events.length - 1) {
      window.clearInterval(state.replayTimer);
      state.replayTimer = null;
      return;
    }
    state.eventIndex += 1;
    renderCurrentEvent();
  }, 700);
}

async function initialize() {
  try {
    const [comparison, cost, summary, evidence] = await Promise.all([
      getJson("/api/results/ollama-24h"),
      getJson("/api/results/cost"),
      getJson("/api/results/summary"),
      getJson("/api/evidence"),
    ]);
    state.comparison = comparison;
    state.cost = cost;
    state.summary = summary;
    populateOverview();
    populateCost();
    renderEvidence(evidence);
    await loadDateOptions();
    await loadLiveEvidence();
    byId("connectionStatus").classList.add("loaded");
    byId("connectionStatus").innerHTML = "<i></i> Evidence loaded";
  } catch (error) {
    byId("connectionStatus").textContent = "Evidence unavailable";
    byId("modeNote").textContent =
      "The dashboard server could not load one or more approved artifacts.";
  }

  const page = window.location.hash.slice(1);
  if (["overview", "live", "results", "cost", "evidence"].includes(page)) {
    activatePage(page);
  }
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => activatePage(button.dataset.page));
});

byId("viewMode").addEventListener("change", async () => {
  await loadDateOptions();
  await loadLiveEvidence();
});
byId("runSelection").addEventListener("change", async () => {
  const runId = byId("runSelection").value;
  if (runId === "cost_estimate") activatePage("cost");
  else if (runId === "phase1_deterministic") activatePage("results");
  else activatePage("live");
  await loadDateOptions();
  await loadLiveEvidence();
});
byId("evidenceDate").addEventListener("change", loadLiveEvidence);

byId("replayButton").addEventListener("click", async () => {
  byId("viewMode").value = "replay";
  activatePage("live");
  await loadLiveEvidence();
  startReplay();
});

byId("demoButton").addEventListener("click", async () => {
  byId("viewMode").value = "replay";
  byId("runSelection").value = "ollama_24h_comparison";
  byId("evidenceDate").value = "";
  activatePage("live");
  await loadDateOptions();
  await loadLiveEvidence();
  startReplay();
});

byId("previousEvent").addEventListener("click", () => {
  stopPlayback();
  state.eventIndex = Math.max(0, state.eventIndex - 1);
  renderCurrentEvent();
});

byId("nextEvent").addEventListener("click", () => {
  stopPlayback();
  state.eventIndex = Math.min(state.events.length - 1, state.eventIndex + 1);
  renderCurrentEvent();
});

initialize();
