import { AMINO_ACID_ORDER, chunkSequence } from "./residues.js";
import { escapeHtml, formatInteger, formatNumber, formatPercent, toCsv } from "./format.js";

export function buildReportCsv(result) {
  const prediction = result?.prediction || {};
  const analysis = result?.analysis || {};
  const probabilities = prediction.probabilities || {};
  const frequency = analysis.amino_acid_frequency || {};
  const structure = analysis.secondary_structure_fraction || {};

  return toCsv([
    ["field", "value"],
    ["record_id", result?.input?.id ?? ""],
    ["model_version", result?.model?.version ?? ""],
    ["predicted_function", prediction.class_name ?? ""],
    ["predicted_top_class", prediction.predicted_class ?? ""],
    ["confidence", prediction.confidence ?? ""],
    ["confidence_threshold", prediction.confidence_threshold ?? ""],
    ["temperature", prediction.temperature ?? ""],
    ["is_uncertain", prediction.is_uncertain ?? ""],
    ["hydrolase_probability", probabilities.hydrolase ?? ""],
    ["oxidoreductase_probability", probabilities.oxidoreductase ?? ""],
    ["sequence_length", analysis.length ?? ""],
    ["molecular_weight", analysis.molecular_weight ?? ""],
    ["isoelectric_point", analysis.isoelectric_point ?? ""],
    ["charge_at_ph7", analysis.charge_at_ph7 ?? ""],
    ["aromaticity", analysis.aromaticity ?? ""],
    ["instability_index", analysis.instability_index ?? ""],
    ["gravy", analysis.gravy ?? ""],
    ["extinction_coefficient_reduced", analysis.extinction_coefficient_reduced ?? ""],
    ["extinction_coefficient_oxidized", analysis.extinction_coefficient_oxidized ?? ""],
    ["helix_fraction", structure.helix ?? ""],
    ["turn_fraction", structure.turn ?? ""],
    ["sheet_fraction", structure.sheet ?? ""],
    ["warnings", (result?.warnings || []).map((warning) => warning.code).join(";")],
    ["invalid_residues", (analysis.invalid_residues || []).join(";")],
    [],
    ["amino_acid", "frequency_fraction"],
    ...AMINO_ACID_ORDER.map((acid) => [acid, frequency[acid] ?? 0]),
  ]);
}

export function buildBatchCsv(results) {
  return toCsv([
    [
      "index", "id", "length", "predicted_function", "predicted_top_class", "confidence", "margin",
      "hydrolase_probability", "oxidoreductase_probability", "molecular_weight", "isoelectric_point",
      "instability_index", "gravy", "warnings", "error",
    ],
    ...results.map((row) => [
      row.index,
      row.id,
      row.length ?? "",
      row.class_name ?? "",
      row.predicted_class ?? "",
      row.confidence ?? "",
      row.margin ?? "",
      row.probabilities?.hydrolase ?? "",
      row.probabilities?.oxidoreductase ?? "",
      row.molecular_weight ?? "",
      row.isoelectric_point ?? "",
      row.instability_index ?? "",
      row.gravy ?? "",
      (row.warnings || []).map((warning) => warning.code).join(";"),
      row.error ?? "",
    ]),
  ]);
}

function tableRows(rows) {
  return rows
    .map(([label, value]) => `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value)}</td></tr>`)
    .join("");
}

export function buildHtmlReport(result, generatedAt = new Date()) {
  const prediction = result.prediction;
  const analysis = result.analysis;
  const model = result.model || {};
  const structure = analysis.secondary_structure_fraction || {};
  const sequence = analysis.prediction_sequence || analysis.normalized_sequence || "";
  const sequenceText = chunkSequence(sequence)
    .map((line) => `${String(line.start + 1).padStart(5)}  ${line.blocks.map((block) => block.text).join(" ")}`)
    .join("\n");

  const predictionRows = [
    ["Call", prediction.class_name],
    ["Top class", prediction.predicted_class],
    ["Confidence", formatPercent(prediction.confidence)],
    ["Confidence threshold", formatPercent(prediction.confidence_threshold)],
    ["Hydrolase probability", formatPercent(prediction.probabilities?.hydrolase, 2)],
    ["Oxidoreductase probability", formatPercent(prediction.probabilities?.oxidoreductase, 2)],
    ["Temperature scaling", prediction.calibrated ? `T = ${formatNumber(prediction.temperature, 3)}` : "none"],
  ];
  const propertyRows = [
    ["Length", formatInteger(analysis.length)],
    ["Molecular weight", `${formatNumber(analysis.molecular_weight, 2)} Da`],
    ["Isoelectric point", formatNumber(analysis.isoelectric_point, 3)],
    ["Net charge at pH 7", formatNumber(analysis.charge_at_ph7, 2)],
    ["Instability index", formatNumber(analysis.instability_index, 2)],
    ["GRAVY", formatNumber(analysis.gravy, 3)],
    ["Aromaticity", formatNumber(analysis.aromaticity, 3)],
    ["Extinction coefficient (reduced / oxidized)", `${formatInteger(analysis.extinction_coefficient_reduced)} / ${formatInteger(analysis.extinction_coefficient_oxidized)} M-1 cm-1`],
    ["Helix / turn / sheet fraction", `${formatNumber(structure.helix, 3)} / ${formatNumber(structure.turn, 3)} / ${formatNumber(structure.sheet, 3)}`],
  ];
  const warnings = (result.warnings || []).map((warning) => `<li>${escapeHtml(warning.message)}</li>`).join("");

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SequenceView report: ${escapeHtml(result.input?.id)}</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.3rem;border-bottom:2px solid #1a1a1a;padding-bottom:.4rem}
h2{font-size:1rem;margin-top:1.6rem}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid #d0d0d0}
th{font-weight:500;color:#555;width:42%}
pre{background:#f4f4f4;padding:10px;overflow-x:auto;font:13px/1.6 ui-monospace,monospace}
.meta{color:#555;font-size:.85rem}
</style>
</head>
<body>
<h1>SequenceView report</h1>
<p class="meta">Record: ${escapeHtml(result.input?.id)}<br>
Generated: ${escapeHtml(generatedAt.toISOString())}<br>
Model version: ${escapeHtml(model.version || "unknown")} (SHA-256 ${escapeHtml(model.checkpoint_sha256 || "unknown")})</p>
${warnings ? `<h2>Warnings</h2><ul>${warnings}</ul>` : ""}
<h2>Prediction</h2>
<table>${tableRows(predictionRows)}</table>
<h2>Sequence properties</h2>
<table>${tableRows(propertyRows)}</table>
<h2>Sequence</h2>
<pre>${escapeHtml(sequenceText)}</pre>
<p class="meta">Predictions come from a baseline classifier trained on UniProt annotations and may be wrong.</p>
</body>
</html>
`;
}
