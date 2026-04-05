import { useMemo, useState } from "react";

const AMINO_ACID_ORDER = "ACDEFGHIKLMNPQRSTVWY".split("");

function normalizeFrequencies(frequencyMap) {
  const values = Object.values(frequencyMap || {});
  const maxValue = values.length ? Math.max(...values) : 0;
  const divideBy = maxValue > 1 ? 100 : 1;

  return AMINO_ACID_ORDER.map((acid) => ({
    acid,
    value: Number(frequencyMap?.[acid] || 0) / divideBy,
  }));
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function makeCsvValue(value) {
  const text = String(value ?? "");
  if (text.includes(",") || text.includes("\n") || text.includes('"')) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function buildReportCsv(result) {
  const prediction = result?.prediction || {};
  const analysis = result?.analysis || {};
  const probabilities = prediction.probabilities || {};
  const aaFrequency = analysis.amino_acid_frequency || {};

  const rows = [
    ["field", "value"],
    ["predicted_function", prediction.class_name || ""],
    ["predicted_top_class", prediction.predicted_class || ""],
    ["confidence", prediction.confidence ?? ""],
    ["confidence_threshold", prediction.confidence_threshold ?? ""],
    ["is_uncertain", prediction.is_uncertain ?? ""],
    ["sequence_length", analysis.length ?? ""],
    ["molecular_weight", analysis.molecular_weight ?? ""],
    ["isoelectric_point", analysis.isoelectric_point ?? ""],
    ["aromaticity", analysis.aromaticity ?? ""],
    ["instability_index", analysis.instability_index ?? ""],
    ["gravy", analysis.gravy ?? ""],
    ["hydrolase_probability", probabilities.hydrolase ?? ""],
    ["oxidoreductase_probability", probabilities.oxidoreductase ?? ""],
    ["invalid_residues", (analysis.invalid_residues || []).join(";")],
    [],
    ["amino_acid", "frequency_fraction"],
    ...AMINO_ACID_ORDER.map((acid) => [acid, aaFrequency[acid] ?? 0]),
  ];

  return rows
    .map((row) => row.map((value) => makeCsvValue(value)).join(","))
    .join("\n");
}

async function readTextFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Failed to read file."));
    reader.readAsText(file);
  });
}

export default function App() {
  const [sequenceInput, setSequenceInput] = useState("");
  const [filename, setFilename] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const aminoBars = useMemo(() => {
    if (!result?.analysis?.amino_acid_frequency) {
      return [];
    }
    return normalizeFrequencies(result.analysis.amino_acid_frequency);
  }, [result]);

  const probabilities = useMemo(() => {
    const probs = result?.prediction?.probabilities;
    if (!probs) {
      return [];
    }

    return [
      { label: "Hydrolase", value: Number(probs.hydrolase || 0) },
      { label: "Oxidoreductase", value: Number(probs.oxidoreductase || 0) },
    ];
  }, [result]);

  async function handleFileUpload(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const content = await readTextFile(file);
      setSequenceInput(content);
      setFilename(file.name);
      setError("");
    } catch (uploadError) {
      setError(uploadError.message);
    }
  }

  async function handleAnalyze(event) {
    event.preventDefault();

    if (!sequenceInput.trim()) {
      setError("Please paste a sequence or upload a FASTA file.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ sequence: sequenceInput }),
      });

      const data = await response.json();

      if (!response.ok) {
        const backendError = data?.error || "Prediction failed.";
        throw new Error(backendError);
      }

      setResult(data);
    } catch (predictionError) {
      setResult(null);
      setError(predictionError.message);
    } finally {
      setLoading(false);
    }
  }

  const predictedClass = result?.prediction?.class_name || "-";
  const confidence = result?.prediction?.confidence;
  const sequenceLength = result?.analysis?.length;
  const invalidResidues = result?.analysis?.invalid_residues || [];

  function exportJsonReport() {
    if (!result) {
      return;
    }
    downloadFile(JSON.stringify(result, null, 2), "sequenceview-report.json", "application/json;charset=utf-8");
  }

  function exportCsvReport() {
    if (!result) {
      return;
    }
    downloadFile(buildReportCsv(result), "sequenceview-report.csv", "text/csv;charset=utf-8");
  }

  return (
    <div className="page-shell">
      <div className="ambient-glow ambient-glow-left" />
      <div className="ambient-glow ambient-glow-right" />

      <main className="layout">
        <section className="panel hero-panel">
          <p className="kicker">SequenceView</p>
          <h1>Protein Sequence Insight Dashboard</h1>
          <p className="lead">
            Paste a raw amino-acid sequence or upload a FASTA file to predict enzyme function and inspect
            composition.
          </p>
        </section>

        <section className="panel">
          <form onSubmit={handleAnalyze} className="input-form">
            <label htmlFor="sequenceInput" className="label-title">
              Sequence Input
            </label>
            <textarea
              id="sequenceInput"
              value={sequenceInput}
              onChange={(event) => setSequenceInput(event.target.value)}
              placeholder=">example_sequence\nMKTFFVILV..."
              rows={9}
            />

            <div className="row-actions">
              <label className="upload-button" htmlFor="sequenceFile">
                Upload .fasta
              </label>
              <input
                id="sequenceFile"
                type="file"
                accept=".fasta,.fa,.faa,.txt"
                onChange={handleFileUpload}
                className="hidden-file-input"
              />

              <button type="submit" disabled={loading}>
                {loading ? "Analyzing..." : "Analyze Sequence"}
              </button>
            </div>

            {filename ? <p className="muted">Loaded file: {filename}</p> : null}
            {error ? <p className="error-text">{error}</p> : null}
          </form>
        </section>

        {result ? (
          <section className="dashboard-grid">
            <article className="panel metric-card">
              <p>Predicted Function</p>
              <h2>{predictedClass.replace("_", " ")}</h2>
              {typeof confidence === "number" ? <span>{formatPercent(confidence)} confidence</span> : null}
            </article>

            <article className="panel metric-card">
              <p>Sequence Length</p>
              <h2>{typeof sequenceLength === "number" ? sequenceLength : "-"}</h2>
              <span>residues</span>
            </article>

            <article className="panel wide-card">
              <h3>Report Export</h3>
              <div className="row-actions">
                <button type="button" onClick={exportJsonReport}>
                  Download JSON
                </button>
                <button type="button" onClick={exportCsvReport}>
                  Download CSV
                </button>
              </div>
            </article>

            <article className="panel wide-card">
              <h3>Class Probability</h3>
              <div className="bar-stack">
                {probabilities.map((item) => (
                  <div className="bar-item" key={item.label}>
                    <div className="bar-label-line">
                      <span>{item.label}</span>
                      <span>{formatPercent(item.value)}</span>
                    </div>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${Math.max(3, item.value * 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </article>

            <article className="panel full-card">
              <h3>Amino Acid Distribution</h3>
              <div className="amino-grid">
                {aminoBars.map((item) => (
                  <div key={item.acid} className="amino-row">
                    <span className="acid-token">{item.acid}</span>
                    <div className="bar-track">
                      <div className="bar-fill bar-fill-alt" style={{ width: `${Math.max(1, item.value * 100)}%` }} />
                    </div>
                    <span className="acid-value">{formatPercent(item.value)}</span>
                  </div>
                ))}
              </div>
            </article>

            {invalidResidues.length ? (
              <article className="panel full-card warning-card">
                <h3>Invalid Residues Removed</h3>
                <p>{invalidResidues.join(", ")}</p>
              </article>
            ) : null}
          </section>
        ) : null}
      </main>
    </div>
  );
}
