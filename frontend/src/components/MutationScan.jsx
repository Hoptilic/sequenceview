import { useMemo, useState } from "react";

import { requestJson } from "../lib/api.js";
import { formatNumber } from "../lib/format.js";

const CELL_W = 8;
const CELL_H = 11;
const LEFT = 16;
const BARS = 46;
const TOP = BARS + 8;

function signed(value, decimals) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(decimals)}`;
}

export default function MutationScan({ header, sequence, focus, onFocus, maxRegion = 250 }) {
  const length = sequence.length;
  const [start, setStart] = useState(1);
  const [end, setEnd] = useState(Math.min(length, maxRegion));
  const [metric, setMetric] = useState("log_odds");
  const [state, setState] = useState({ status: "idle", error: "", scan: null });

  async function run() {
    setState({ status: "loading", error: "", scan: null });
    try {
      const scan = await requestJson("/api/mutscan", {
        sequence: `>${header || "sequence"}\n${sequence}`,
        start: start - 1,
        end,
      });
      setState({ status: "ready", error: "", scan });
    } catch (error) {
      setState({ status: "error", error: error.message, scan: null });
    }
  }

  const { scan } = state;
  const key = metric === "log_odds" ? "delta_log_odds" : "delta_probability";

  const geometry = useMemo(() => {
    if (!scan) {
      return null;
    }
    const alphabet = scan.alphabet.split("");
    const sensitivities = scan.positions.map((position) => Math.max(...position[key].map(Math.abs)));
    const peak = Math.max(...scan.positions.flatMap((position) => position[key].map(Math.abs)), metric === "log_odds" ? 0.5 : 0.05);
    return { alphabet, sensitivities, peak, width: LEFT + scan.positions.length * CELL_W, height: TOP + alphabet.length * CELL_H + 4 };
  }, [scan, key, metric]);

  const detail = useMemo(() => {
    if (!scan || focus === null || focus === undefined) {
      return null;
    }
    const position = scan.positions.find((item) => item.index === focus);
    if (!position) {
      return null;
    }
    const ranked = scan.alphabet
      .split("")
      .map((acid, index) => ({ acid, delta: position[key][index] }))
      .filter((item) => item.acid !== position.residue);
    ranked.sort((left, right) => right.delta - left.delta);
    return { position, raising: ranked.slice(0, 3), lowering: ranked.slice(-3).reverse() };
  }, [scan, focus, key]);

  const topPositions = useMemo(() => {
    if (!scan || !geometry) {
      return [];
    }
    return scan.positions
      .map((position, index) => ({ position, value: geometry.sensitivities[index] }))
      .sort((left, right) => right.value - left.value)
      .slice(0, 6);
  }, [scan, geometry]);

  const decimals = metric === "log_odds" ? 2 : 3;
  const unit = metric === "log_odds" ? "log-odds" : "probability";

  return (
    <div>
      <div className="scan-controls">
        <label>
          From
          <input type="number" min="1" max={length} value={start} onChange={(event) => setStart(Math.max(1, Number(event.target.value) || 1))} />
        </label>
        <label>
          To
          <input type="number" min="1" max={length} value={end} onChange={(event) => setEnd(Math.min(length, Number(event.target.value) || length))} />
        </label>
        <button type="button" className="button" onClick={run} disabled={state.status === "loading" || end < start || end - start + 1 > maxRegion}>
          {state.status === "loading" ? "Scanning..." : "Run mutation scan"}
        </button>
        <span className="meta">{Math.max(0, end - start + 1) * 20} variants (limit {maxRegion} residues per scan)</span>
      </div>
      {end - start + 1 > maxRegion ? <p className="error">Choose a region of at most {maxRegion} residues.</p> : null}
      {state.error ? <p className="error">{state.error}</p> : null}

      {scan && geometry ? (
        <div>
          <div className="field-head">
            <p className="meta">
              Baseline P(oxidoreductase) {formatNumber(scan.baseline_oxidoreductase, 4)} | log-odds {formatNumber(scan.baseline_log_odds, 2)}
            </p>
            <span className="field-tools">
              <button type="button" className={`link${metric === "log_odds" ? " active" : ""}`} onClick={() => setMetric("log_odds")}>
                Log-odds
              </button>
              <button type="button" className={`link${metric === "probability" ? " active" : ""}`} onClick={() => setMetric("probability")}>
                Probability
              </button>
            </span>
          </div>

          <div className="table-scroll">
            <svg className="mutmap" viewBox={`0 0 ${geometry.width} ${geometry.height}`} style={{ minWidth: Math.min(geometry.width * 1.6, 1400) }} role="img" aria-label="Mutation scan heat map">
              {scan.positions.map((position, column) => {
                const x = LEFT + column * CELL_W;
                const barHeight = Math.max(1, (geometry.sensitivities[column] / geometry.peak) * BARS);
                const selected = position.index === focus;
                return (
                  <g key={position.index} onClick={() => onFocus(position.index)} className="mut-column">
                    <title>{`${position.residue}${position.index + 1}: largest change ${geometry.sensitivities[column].toFixed(decimals)} ${unit}`}</title>
                    <rect x={x} y="0" width={CELL_W} height={geometry.height} className="mut-hit" />
                    <rect x={x + 1} y={BARS - barHeight} width={CELL_W - 2} height={barHeight} className="mut-bar" />
                    {position[key].map((value, row) => {
                      const amount = Math.min(1, Math.abs(value) / geometry.peak);
                      return (
                        <rect
                          key={row}
                          x={x}
                          y={TOP + row * CELL_H}
                          width={CELL_W - 0.5}
                          height={CELL_H - 0.5}
                          className={value >= 0 ? "mut-up" : "mut-down"}
                          style={{ "--w": amount }}
                        />
                      );
                    })}
                    {scan.alphabet.indexOf(position.residue) >= 0 ? (
                      <circle cx={x + CELL_W / 2} cy={TOP + scan.alphabet.indexOf(position.residue) * CELL_H + CELL_H / 2} r="1.3" className="mut-wt" />
                    ) : null}
                    {selected ? <rect x={x} y="0" width={CELL_W} height={geometry.height} className="mut-selected" /> : null}
                  </g>
                );
              })}
              {geometry.alphabet.map((acid, row) => (
                <text key={acid} x={LEFT - 4} y={TOP + row * CELL_H + CELL_H - 2} textAnchor="end" className="mut-label">
                  {acid}
                </text>
              ))}
              {scan.positions.map((position, column) =>
                (position.index + 1) % 10 === 0 || column === 0 ? (
                  <text key={`t${position.index}`} x={LEFT + column * CELL_W + CELL_W / 2} y={geometry.height + 9} textAnchor="middle" className="mut-label">
                    {position.index + 1}
                  </text>
                ) : null,
              )}
            </svg>
          </div>

          <div className="legend">
            <span className="legend-swatch mut-up" style={{ "--w": 0.8 }}>raises oxidoreductase {unit}</span>
            <span className="legend-swatch mut-down" style={{ "--w": 0.8 }}>lowers it</span>
            <span>dot marks the wild-type residue; bars show the largest change at each position</span>
          </div>

          <p className="meta">
            Most sensitive positions:{" "}
            {topPositions.map(({ position, value }, index) => (
              <span key={position.index}>
                {index ? ", " : ""}
                <button type="button" className="link" onClick={() => onFocus(position.index)}>
                  {position.residue}{position.index + 1}
                </button>{" "}
                ({value.toFixed(decimals)})
              </span>
            ))}
          </p>

          {detail ? (
            <table className="data data-compact scan-detail">
              <thead>
                <tr>
                  <th>
                    {detail.position.residue}
                    {detail.position.index + 1}
                  </th>
                  <th className="num">Change ({unit})</th>
                </tr>
              </thead>
              <tbody>
                {detail.raising.map((item) => (
                  <tr key={`up${item.acid}`}>
                    <th>to {item.acid}, raises</th>
                    <td className="num">{signed(item.delta, decimals)}</td>
                  </tr>
                ))}
                {detail.lowering.map((item) => (
                  <tr key={`down${item.acid}`}>
                    <th>to {item.acid}, lowers</th>
                    <td className="num">{signed(item.delta, decimals)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="note">Click a column to see that position's largest effects and to mark it on the sequence and structure.</p>
          )}
          <p className="note">
            Each cell is the change in the classifier's output for a single substitution, computed in silico with this model.
            It shows what the network is sensitive to; it is not a measurement of enzyme activity or stability. Inputs are
            padded to the sequence length for speed, which changes baseline outputs only marginally.
          </p>
        </div>
      ) : null}
    </div>
  );
}
