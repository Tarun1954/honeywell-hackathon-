import comparison from "../runs/final/ollama_24h_comparison.json";

export const dynamic = "force-static";

const format = (value, digits = 3) =>
  Number(value).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });

const percentReduction = Math.abs(
  comparison.electricity.percentage_difference_from_baseline
);

const flow = [
  ["01", "EnergyPlus", "15-minute telemetry"],
  ["02", "MCP boundary", "Five typed tools"],
  ["03", "Ollama agent", "Hourly proposal"],
  ["04", "Safety gate", "Bounds + deadband"],
  ["05", "EMS actuators", "Four-step hold"]
];

export default function Home() {
  const baseline = comparison.baseline_comfort;
  const controlled = comparison.controlled_comfort;
  const actions = comparison.control_actions;

  return (
    <main>
      <nav className="nav shell" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="Eco-Loop home">
          <span className="brand-mark">EL</span>
          <span>Eco-Loop</span>
        </a>
        <div className="nav-links">
          <a href="#results">Results</a>
          <a href="#safety">Safety</a>
          <a href="#evidence">Evidence</a>
          <a
            className="repo-link"
            href="https://github.com/Tarun1954/honeywell-hackathon-"
          >
            GitHub ↗
          </a>
        </div>
      </nav>

      <header className="hero shell" id="top">
        <div className="hero-copy">
          <p className="eyebrow">
            <span className="live-dot" aria-hidden="true" />
            Verified EnergyPlus + MCP + Ollama evidence
          </p>
          <h1>
            Building control that is <em>measurable, bounded,</em> and
            recoverable.
          </h1>
          <p className="lede">
            A local language model supervises a five-zone EnergyPlus building
            once per simulated hour. Every thermostat command crosses a typed
            MCP boundary and an independent safety gate before writeback.
          </p>
          <div className="hero-actions">
            <a className="button primary" href="#results">
              Explore verified results
            </a>
            <a
              className="button secondary"
              href="/evidence/ollama_24h_comparison.json"
            >
              Inspect JSON evidence
            </a>
          </div>
        </div>
        <aside className="hero-card" aria-label="Headline result">
          <span className="card-label">24-hour matched comparison</span>
          <strong>{format(percentReduction)}%</strong>
          <span className="card-caption">less facility electricity</span>
          <div className="mini-comparison">
            <span>Baseline</span>
            <b>{format(comparison.electricity.baseline_total_kwh)} kWh</b>
            <span>Ollama hybrid</span>
            <b>{format(comparison.electricity.controlled_total_kwh)} kWh</b>
          </div>
          <p>EnergyPlus completed with zero severe and zero fatal errors.</p>
        </aside>
      </header>

      <section className="flow-section shell" aria-labelledby="architecture">
        <div className="section-intro">
          <p className="kicker">Closed-loop architecture</p>
          <h2 id="architecture">One observable path from sensor to actuator</h2>
        </div>
        <ol className="flow">
          {flow.map(([number, title, detail]) => (
            <li key={number}>
              <span>{number}</span>
              <strong>{title}</strong>
              <small>{detail}</small>
            </li>
          ))}
        </ol>
      </section>

      <section className="results shell" id="results" aria-labelledby="results-title">
        <div className="section-intro split">
          <div>
            <p className="kicker">Real-model comparison</p>
            <h2 id="results-title">Energy and peak demand fell</h2>
          </div>
          <p>
            Same model, weather, timestep, run-period start, and initial
            snapshot. The controlled case used <code>{comparison.model}</code>.
          </p>
        </div>

        <div className="metrics">
          <article>
            <span>Electricity reduction</span>
            <strong>{format(comparison.electricity.absolute_difference_kwh)}</strong>
            <small>kWh</small>
          </article>
          <article>
            <span>Baseline peak</span>
            <strong>{format(comparison.peak_demand.baseline_w / 1000)}</strong>
            <small>kW</small>
          </article>
          <article className="accent">
            <span>Controlled peak</span>
            <strong>{format(comparison.peak_demand.controlled_w / 1000)}</strong>
            <small>kW</small>
          </article>
          <article>
            <span>Accepted actions</span>
            <strong>{actions.real_actions_accepted}</strong>
            <small>of {actions.hourly_decision_count} hourly decisions</small>
          </article>
        </div>

        <div className="chart-grid">
          <figure className="chart-card wide">
            <div className="figure-copy">
              <p className="kicker">Electricity + demand</p>
              <h3>Matched 24-hour outcome</h3>
            </div>
            <img
              src="/evidence/ollama_24h_energy_peak.png"
              alt="Bar charts comparing baseline and Ollama-hybrid electricity and peak demand."
            />
          </figure>
          <figure className="chart-card">
            <div className="figure-copy">
              <p className="kicker">Comfort</p>
              <h3>The trade-off stays visible</h3>
            </div>
            <img
              src="/evidence/ollama_24h_comfort.png"
              alt="Range charts comparing baseline and Ollama-hybrid temperature and PMV."
            />
          </figure>
          <article className="tradeoff-card">
            <p className="kicker">Honest evaluation</p>
            <h3>Lower energy, slightly lower comfort compliance</h3>
            <dl>
              <div>
                <dt>Occupied PMV compliance</dt>
                <dd>
                  {format(baseline.occupied_pmv_compliance_percent, 2)}% →{" "}
                  {format(controlled.occupied_pmv_compliance_percent, 2)}%
                </dd>
              </div>
              <div>
                <dt>Comfort violations</dt>
                <dd>
                  {baseline.occupied_comfort_violation_count} →{" "}
                  {controlled.occupied_comfort_violation_count}
                </dd>
              </div>
            </dl>
            <p>
              The prototype demonstrates a measurable energy benefit, not a
              finished comfort policy. Future optimization should explicitly
              balance both objectives.
            </p>
          </article>
        </div>
      </section>

      <section className="safety" id="safety" aria-labelledby="safety-title">
        <div className="shell safety-grid">
          <div>
            <p className="kicker light">Independent safety layer</p>
            <h2 id="safety-title">
              The model proposes. Deterministic code decides.
            </h2>
            <p>
              Unknown zones, incomplete actions, unsafe occupied setpoints,
              stale snapshots, deadband violations, non-finite values, and
              severe runtime errors are rejected before actuator writeback.
            </p>
          </div>
          <div className="safety-stats">
            <article>
              <strong>16–24 °C</strong>
              <span>heating bounds</span>
            </article>
            <article>
              <strong>20–30 °C</strong>
              <span>cooling bounds</span>
            </article>
            <article>
              <strong>≥ 1 °C</strong>
              <span>minimum deadband</span>
            </article>
            <article>
              <strong>4 × 15 min</strong>
              <span>maximum action hold</span>
            </article>
          </div>
        </div>
      </section>

      <section className="evidence shell" id="evidence" aria-labelledby="evidence-title">
        <div className="section-intro split">
          <div>
            <p className="kicker">Auditable by design</p>
            <h2 id="evidence-title">Open the source evidence</h2>
          </div>
          <p>
            The dashboard presents checked-in artifacts. It does not control a
            live building or mutate simulation state.
          </p>
        </div>
        <div className="evidence-grid">
          <a href="/evidence/ollama_24h_comparison.json">
            <span>01</span>
            <strong>Comparison JSON</strong>
            <small>Machine-readable metrics ↗</small>
          </a>
          <a href="/evidence/ollama_24h_comparison.csv">
            <span>02</span>
            <strong>Comparison CSV</strong>
            <small>Tabular evidence ↗</small>
          </a>
          <a href="/evidence/results_summary.json">
            <span>03</span>
            <strong>Evidence summary</strong>
            <small>Separated evidence classes ↗</small>
          </a>
          <a href="https://github.com/Tarun1954/honeywell-hackathon-/blob/phase-2/docs/system_architecture.md">
            <span>04</span>
            <strong>Architecture report</strong>
            <small>Detailed failure paths ↗</small>
          </a>
        </div>
      </section>

      <footer>
        <div className="shell footer-inner">
          <div>
            <a className="brand" href="#top">
              <span className="brand-mark">EL</span>
              <span>Eco-Loop Building Agents</span>
            </a>
            <p>Honeywell hackathon proof of concept.</p>
          </div>
          <p>
            Evidence class: <strong>{comparison.evidence_class}</strong>
            <br />
            Status: <strong>{comparison.status}</strong>
          </p>
        </div>
      </footer>
    </main>
  );
}
