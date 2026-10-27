import { useMemo, useState } from "react";

import { ChargePlot, HydropathyPlot } from "./Charts.jsx";
import SequenceViewer, { AttributionStrip } from "./SequenceViewer.jsx";
import { downloadFile, formatInteger, formatNumber, formatPercent } from "../lib/format.js";
import { findMotifs, motifMask } from "../lib/motif.js";
import { GROUP_LABELS, normalizeFrequencies, residueGroup } from "../lib/residues.js";
import { buildHtmlReport, buildReportCsv } from "../lib/report.js";

function fileStem(result) {
  const id = (result.input?.id || "report").replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 40);
  return `sequenceview-${id}`;
}

export default function Detail({ result, onShare, shareState }) {
  const [mode, setMode] = useState("attribution");
  const [motifText, setMotifText] = useState("");

  const { prediction, analysis, attribution, warnings = [], model } = result;
  const sequence = analysis.prediction_sequence;
  const scores = attribution?.scores;

  const motif = useMemo(() => findMotifs(sequence, motifText), [sequence, motifText]);
  const mask = useMemo(() => motifMask(sequence.length, motif.matches), [sequence, motif]);

  const aminoBars = useMemo(() => normalizeFrequencies(analysis.amino_acid_frequency), [analysis]);
  const groupTotals = useMemo(() => {
    const totals = { nonpolar: 0, polar: 0, positive: 0, negative: 0 };
    aminoBars.forEach((item) => {
      totals[residueGroup(item.acid)] += item.value;
    });
    return totals;
  }, [aminoBars]);

  const isUncertain = prediction.is_uncertain;
  const verdictTone = isUncertain ? "uncertain" : "called";
  const probabilities = [
    { label: "Hydrolase", ec: "EC 3.*", value: prediction.probabilities.hydrolase, tone: "a" },
    { label: "Oxidoreductase", ec: "EC 1.*", value: prediction.probabilities.oxidoreductase, tone: "b" },
  ];

  const residueCount = analysis.analysis_sequence.length;
  const structure = analysis.secondary_structure_fraction;
  const instability = analysis.instability_index;
  const properties = [
    ["Length", formatInteger(analysis.length), analysis.invalid_residues.length ? `${sequence.length} including non-standard` : ""],
    ["Molecular weight", `${formatNumber(analysis.molecular_weight, 2)} Da`, `${formatNumber(analysis.molecular_weight / residueCount, 2)} Da per residue`],
    ["Isoelectric point", formatNumber(analysis.isoelectric_point, 3), ""],
    ["Net charge at pH 7", formatNumber(analysis.charge_at_ph7, 2), ""],
    ["Instability index", formatNumber(instability, 2), instability < 40 ? "below 40: likely stable" : "40 or above: possibly unstable"],
    ["GRAVY", formatNumber(analysis.gravy, 3), analysis.gravy > 0 ? "overall hydrophobic" : "overall hydrophilic"],
    ["Aromaticity", formatNumber(analysis.aromaticity, 3), ""],
    [
      "Extinction coefficient at 280 nm",
      `${formatInteger(analysis.extinction_coefficient_reduced)} / ${formatInteger(analysis.extinction_coefficient_oxidized)}`,
      "M⁻¹ cm⁻¹, reduced / oxidized cysteines",
    ],
    [
      "Secondary structure (helix / turn / sheet)",
      `${formatNumber(structure.helix, 3)} / ${formatNumber(structure.turn, 3)} / ${formatNumber(structure.sheet, 3)}`,
      "fraction of residues in each propensity class, estimated from composition",
    ],
  ];

  return (
    <div>
      {warnings.length ? (
        <section className="section notice-block" aria-label="Warnings">
          <h2>Warnings</h2>
          <ul className="plain-list">
            {warnings.map((warning) => (
              <li key={warning.code}>{warning.message}</li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="section">
        <h2>Prediction</h2>
        <table className="data">
          <tbody>
            <tr>
              <th>Call</th>
              <td className={`call call-${verdictTone}`}>{prediction.class_name}</td>
              <td className="note">{isUncertain ? `below the ${formatPercent(prediction.confidence_threshold)} threshold` : ""}</td>
            </tr>
            <tr>
              <th>Top class</th>
              <td>{prediction.predicted_class}</td>
              <td />
            </tr>
            <tr>
              <th>Confidence</th>
              <td>{formatPercent(prediction.confidence)}</td>
              <td className="note">
                {prediction.calibrated
                  ? `temperature-scaled (T = ${formatNumber(prediction.temperature, 2)})`
                  : "uncalibrated softmax output; likely overconfident"}
              </td>
            </tr>
            <tr>
              <th>Margin</th>
              <td>{formatPercent(prediction.margin)}</td>
              <td />
            </tr>
          </tbody>
        </table>

        <div className="probs">
          {probabilities.map((item) => (
            <div className="prob-row" key={item.label}>
              <span>
                {item.label} <small>{item.ec}</small>
              </span>
              <div className="track">
                <div className={`fill fill-${item.tone}`} style={{ width: `${item.value * 100}%` }} />
              </div>
              <span className="num">{formatPercent(item.value)}</span>
            </div>
          ))}
        </div>
        <p className="note stamp">
          Model {model.version}. Sequence trimmed to the first {model.max_length} residues for prediction.
        </p>
      </section>

      <section className="section">
        <div className="field-head">
          <h2>Sequence</h2>
          <span className="field-tools">
            <button type="button" className={`link${mode === "attribution" ? " active" : ""}`} onClick={() => setMode("attribution")} disabled={!scores}>
              Attribution
            </button>
            <button type="button" className={`link${mode === "class" ? " active" : ""}`} onClick={() => setMode("class")}>
              Residue class
            </button>
          </span>
        </div>

        {mode === "attribution" && scores ? <AttributionStrip scores={scores} /> : null}

        <div className="motif-row">
          <label htmlFor="motifInput">Motif search</label>
          <input
            id="motifInput"
            type="text"
            value={motifText}
            onChange={(event) => setMotifText(event.target.value)}
            placeholder="PROSITE (C-x(2,4)-C) or regular expression"
            spellCheck={false}
          />
          <span className={motif.error ? "error" : "meta"}>
            {motif.error || (motifText.trim() ? `${motif.matches.length} match${motif.matches.length === 1 ? "" : "es"}` : "")}
          </span>
        </div>

        <SequenceViewer sequence={sequence} scores={scores} mode={mode} motifMask={mask} predictedClass={attribution?.target} />
        {mode === "attribution" && scores ? (
          <p className="note">
            Gradient times input on the logit difference for the predicted class, scaled to the largest magnitude. It shows
            which positions the output is most sensitive to, not which residues are catalytically relevant.
          </p>
        ) : null}
      </section>

      <section className="section">
        <h2>Sequence properties</h2>
        <table className="data">
          <tbody>
            {properties.map(([label, value, note]) => (
              <tr key={label}>
                <th>{label}</th>
                <td className="num">{value}</td>
                <td className="note">{note}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="plots">
          <figure>
            <figcaption>Hydropathy profile (window {analysis.hydropathy_profile.window})</figcaption>
            <HydropathyPlot profile={analysis.hydropathy_profile} />
          </figure>
          <figure>
            <figcaption>Net charge versus pH</figcaption>
            <ChargePlot curve={analysis.charge_curve} isoelectricPoint={analysis.isoelectric_point} />
          </figure>
        </div>
      </section>

      <section className="section">
        <h2>Composition</h2>
        <table className="data data-compact">
          <thead>
            <tr>
              <th>Class</th>
              <th className="num">Fraction</th>
            </tr>
          </thead>
          <tbody>
            {Object.keys(GROUP_LABELS).map((group) => (
              <tr key={group}>
                <th className={`res-${group}`}>{GROUP_LABELS[group]}</th>
                <td className="num">{formatPercent(groupTotals[group])}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="amino-grid">
          {aminoBars.map((item) => (
            <div key={item.acid} className="amino-row">
              <span className={`acid res-${residueGroup(item.acid)}`}>{item.acid}</span>
              <div className="track">
                <div className={`fill fill-${residueGroup(item.acid)}`} style={{ width: `${item.value * 100}%` }} />
              </div>
              <span className="num">{formatPercent(item.value)}</span>
            </div>
          ))}
        </div>
      </section>

      {analysis.invalid_residues.length ? (
        <section className="section notice">
          Non-standard residues ({analysis.invalid_residues.join(", ")}) were excluded from the property calculations and
          encoded as unknown for the classifier.
        </section>
      ) : null}

      <section className="section">
        <h2>Export and share</h2>
        <span className="actions">
          <button type="button" className="button" onClick={() => downloadFile(JSON.stringify(result, null, 2), `${fileStem(result)}.json`, "application/json;charset=utf-8")}>
            JSON
          </button>
          <button type="button" className="button" onClick={() => downloadFile(buildReportCsv(result), `${fileStem(result)}.csv`, "text/csv;charset=utf-8")}>
            CSV
          </button>
          <button type="button" className="button" onClick={() => downloadFile(buildHtmlReport(result), `${fileStem(result)}.html`, "text/html;charset=utf-8")}>
            HTML report
          </button>
          <button type="button" className="button" onClick={onShare}>
            {shareState === "copied" ? "Link copied" : "Copy link"}
          </button>
        </span>
        {shareState === "too-long" ? <p className="note">Input is too long to embed in a link. Use the HTML report instead.</p> : null}
        {shareState === "failed" ? <p className="note">Clipboard access was denied. The link is in the address bar.</p> : null}
      </section>
    </div>
  );
}
