import { useEffect, useMemo, useRef, useState } from "react";

const AMINO_ACID_ORDER = "ACDEFGHIKLMNPQRSTVWY".split("");

const RESIDUE_GROUPS = {
  nonpolar: "AVLIMFWPG",
  polar: "STCYNQ",
  positive: "KRH",
  negative: "DE",
};

const GROUP_LABELS = {
  nonpolar: "Nonpolar",
  polar: "Polar",
  positive: "Basic (+)",
  negative: "Acidic (-)",
};

const EXAMPLE_SEQUENCE = `>sp|P00698|LYSC_CHICK Lysozyme C
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL`;

function residueGroup(acid) {
  return Object.keys(RESIDUE_GROUPS).find((group) => RESIDUE_GROUPS[group].includes(acid)) || "nonpolar";
}

function chunkSequence(sequence, blockSize = 10, blocksPerLine = 6) {
  const lineSize = blockSize * blocksPerLine;
  const lines = [];
  for (let start = 0; start < sequence.length; start += lineSize) {
    const line = sequence.slice(start, start + lineSize);
    const blocks = [];
    for (let offset = 0; offset < line.length; offset += blockSize) {
      blocks.push(line.slice(offset, offset + blockSize));
    }
    lines.push({ start: start + 1, blocks });
  }
  return lines;
}

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

function formatNumber(value, decimals = 3) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(decimals) : "-";
}

function formatInteger(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat().format(Math.round(value))
    : "-";
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
  const [dragging, setDragging] = useState(false);
  const [copied, setCopied] = useState(false);
  const sequenceInputRef = useRef(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    const textarea = sequenceInputRef.current;
    if (!textarea) {
      return;
    }
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight + 2, 160), 420)}px`;
  }, [sequenceInput]);

  useEffect(() => {
    if (result) {
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [result]);

  const inputStats = useMemo(() => {
    const residues = sequenceInput
      .split("\n")
      .filter((line) => !line.startsWith(">"))
      .join("")
      .replace(/\s/g, "");
    const headers = sequenceInput.split("\n").filter((line) => line.startsWith(">")).length;
    return { residues: residues.length, headers };
  }, [sequenceInput]);

  const aminoBars = useMemo(() => {
    if (!result?.analysis?.amino_acid_frequency) {
      return [];
    }
    return normalizeFrequencies(result.analysis.amino_acid_frequency);
  }, [result]);

  const groupTotals = useMemo(() => {
    const totals = { nonpolar: 0, polar: 0, positive: 0, negative: 0 };
    aminoBars.forEach((item) => {
      totals[residueGroup(item.acid)] += item.value;
    });
    return totals;
  }, [aminoBars]);

  const probabilities = useMemo(() => {
    const probs = result?.prediction?.probabilities;
    if (!probs) {
      return [];
    }
    return [
      { label: "Hydrolase", ec: "EC 3.*", value: Number(probs.hydrolase || 0), tone: "teal" },
      { label: "Oxidoreductase", ec: "EC 1.*", value: Number(probs.oxidoreductase || 0), tone: "amber" },
    ];
  }, [result]);

  const sequenceLines = useMemo(
    () => chunkSequence(result?.analysis?.normalized_sequence || ""),
    [result],
  );

  function loadText(content, name) {
    setSequenceInput(content);
    setFilename(name);
    setError("");
  }

  async function readIntoInput(file) {
    try {
      loadText(await readTextFile(file), file.name);
    } catch (uploadError) {
      setError(uploadError.message);
    }
  }

  async function handleFileUpload(event) {
    const file = event.target.files?.[0];
    if (file) {
      await readIntoInput(file);
    }
    event.target.value = "";
  }

  async function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      await readIntoInput(file);
    }
  }

  function handleClear() {
    setSequenceInput("");
    setFilename("");
    setError("");
    setResult(null);
  }

  async function handleAnalyze(event) {
    event.preventDefault();

    if (!sequenceInput.trim()) {
      setError("Paste a sequence or upload a FASTA file first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence: sequenceInput }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.error || "Prediction failed.");
      }

      setResult(data);
    } catch (predictionError) {
      setResult(null);
      setError(predictionError.message);
    } finally {
      setLoading(false);
    }
  }

  async function copySequence() {
    try {
      await navigator.clipboard.writeText(result?.analysis?.normalized_sequence || "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      setError("Clipboard access was denied.");
    }
  }

  const prediction = result?.prediction;
  const analysis = result?.analysis;
  const confidence = prediction?.confidence;
  const isUncertain = Boolean(prediction?.is_uncertain);
  const verdict = prediction?.class_name || "-";
  const verdictTone = isUncertain ? "uncertain" : prediction?.predicted_class === "oxidoreductase" ? "amber" : "teal";
  const confidencePercent = typeof confidence === "number" ? Math.round(confidence * 100) : 0;
  const residueCount = analysis?.analysis_sequence?.length;
  const invalidResidues = analysis?.invalid_residues || [];
  const averageResidueMass =
    typeof analysis?.molecular_weight === "number" && residueCount > 0
      ? analysis.molecular_weight / residueCount
      : null;
  const instabilityIndex = analysis?.instability_index;
  const isLikelyStable = typeof instabilityIndex === "number" ? instabilityIndex < 40 : null;
  const gravy = analysis?.gravy;

  const stats = analysis
    ? [
        { label: "Molecular weight", value: `${formatNumber(analysis.molecular_weight, 2)} Da` },
        { label: "Isoelectric point", value: formatNumber(analysis.isoelectric_point, 3) },
        { label: "Instability index", value: formatNumber(instabilityIndex, 2), hint: isLikelyStable === null ? null : isLikelyStable ? "Likely stable" : "Possibly unstable", good: isLikelyStable },
        { label: "GRAVY", value: formatNumber(gravy, 3), hint: typeof gravy === "number" ? (gravy > 0 ? "Hydrophobic" : "Hydrophilic") : null },
        { label: "Aromaticity", value: formatNumber(analysis.aromaticity, 3) },
        { label: "Residues analysed", value: formatInteger(residueCount) },
        { label: "Avg. residue mass", value: averageResidueMass ? `${formatNumber(averageResidueMass, 2)} Da` : "-" },
        { label: "Confidence threshold", value: typeof prediction?.confidence_threshold === "number" ? formatPercent(prediction.confidence_threshold) : "-" },
      ]
    : [];

  function exportJsonReport() {
    if (result) {
      downloadFile(JSON.stringify(result, null, 2), "sequenceview-report.json", "application/json;charset=utf-8");
    }
  }

  function exportCsvReport() {
    if (result) {
      downloadFile(buildReportCsv(result), "sequenceview-report.csv", "text/csv;charset=utf-8");
    }
  }

  return (
    <div className="page-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M6 3c0 6 12 6 12 12s-12 6-12 6" />
              <path d="M18 3c0 6-12 6-12 12s12 6 12 6" />
              <path d="M8 7h8M8 17h8" />
            </svg>
          </span>
          <span className="brand-name">SequenceView</span>
        </div>
        <span className="topbar-tag">BiGRU enzyme classifier</span>
      </header>

      <main className="layout">
        <section className="hero">
          <h1>
            Read a protein sequence, <span className="gradient-text">predict its function.</span>
          </h1>
          <p className="lead">
            Classify enzymes as hydrolases or oxidoreductases and explore their biochemical profile in one view.
          </p>
        </section>

        <section className="panel input-panel">
          <form onSubmit={handleAnalyze} className="input-form">
            <div className="input-head">
              <label htmlFor="sequenceInput" className="label-title">
                Sequence
              </label>
              <div className="input-head-actions">
                <button type="button" className="link-button" onClick={() => loadText(EXAMPLE_SEQUENCE, "lysozyme-example.fasta")}>
                  Load example
                </button>
                {sequenceInput ? (
                  <button type="button" className="link-button" onClick={handleClear}>
                    Clear
                  </button>
                ) : null}
              </div>
            </div>

            <div
              className={`dropzone${dragging ? " dropzone-active" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <textarea
                ref={sequenceInputRef}
                id="sequenceInput"
                value={sequenceInput}
                onChange={(event) => {
                  setSequenceInput(event.target.value);
                  setFilename("");
                }}
                placeholder={">example_sequence\nMKTFFVILV..."}
                spellCheck={false}
                rows={6}
              />
              {dragging ? <div className="drop-overlay">Drop a FASTA file to load it</div> : null}
            </div>

            <div className="input-footer">
              <div className="input-meta">
                <span className="chip">{formatInteger(inputStats.residues)} residues</span>
                {inputStats.headers > 1 ? <span className="chip chip-warn">{inputStats.headers} FASTA records (merged)</span> : null}
                {filename ? <span className="chip chip-file">{filename}</span> : null}
              </div>
              <div className="row-actions">
                <label className="btn btn-secondary" htmlFor="sequenceFile">
                  Upload FASTA
                </label>
                <input
                  id="sequenceFile"
                  type="file"
                  accept=".fasta,.fa,.faa,.txt"
                  onChange={handleFileUpload}
                  className="hidden-file-input"
                />
                <button type="submit" className="btn btn-primary" disabled={loading}>
                  {loading ? <span className="spinner" aria-hidden="true" /> : null}
                  {loading ? "Analysing..." : "Analyse sequence"}
                </button>
              </div>
            </div>

            {error ? (
              <p className="error-text" role="alert">
                {error}
              </p>
            ) : null}
          </form>
        </section>

        {loading && !result ? (
          <section className="dashboard-grid" aria-hidden="true">
            <div className="panel skeleton span-5" />
            <div className="panel skeleton span-7" />
          </section>
        ) : null}

        {result ? (
          <section className="dashboard-grid" ref={resultsRef}>
            <article className={`panel verdict-card tone-${verdictTone} span-5`}>
              <p className="eyebrow">Predicted function</p>
              <h2>{verdict}</h2>
              <div className="gauge" style={{ "--value": confidencePercent }}>
                <div className="gauge-inner">
                  <strong>{confidencePercent}%</strong>
                  <span>confidence</span>
                </div>
              </div>
              {isUncertain ? (
                <p className="verdict-note">
                  Below the {formatPercent(prediction.confidence_threshold)} threshold. Closest class: {prediction.predicted_class}.
                </p>
              ) : (
                <p className="verdict-note">Margin between classes: {formatPercent(prediction.margin)}</p>
              )}
            </article>

            <article className="panel span-7">
              <div className="card-head">
                <h3>Class probability</h3>
                <span className="chip">{formatInteger(analysis.length)} aa</span>
              </div>
              <div className="bar-stack">
                {probabilities.map((item) => (
                  <div className="bar-item" key={item.label}>
                    <div className="bar-label-line">
                      <span>
                        {item.label} <em>{item.ec}</em>
                      </span>
                      <strong>{formatPercent(item.value)}</strong>
                    </div>
                    <div className="bar-track bar-track-lg">
                      <div className={`bar-fill fill-${item.tone}`} style={{ width: `${Math.max(2, item.value * 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
              <div className="export-row">
                <button type="button" className="btn btn-secondary btn-sm" onClick={exportJsonReport}>
                  Export JSON
                </button>
                <button type="button" className="btn btn-secondary btn-sm" onClick={exportCsvReport}>
                  Export CSV
                </button>
              </div>
            </article>

            <article className="panel span-12">
              <h3>Sequence properties</h3>
              <div className="stats-grid">
                {stats.map((item) => (
                  <div className="stat-item" key={item.label}>
                    <span>{item.label}</span>
                    <strong>{item.value}</strong>
                    {item.hint ? (
                      <em className={item.good === true ? "hint-good" : item.good === false ? "hint-bad" : ""}>{item.hint}</em>
                    ) : null}
                  </div>
                ))}
              </div>
            </article>

            <article className="panel span-12">
              <div className="card-head">
                <h3>Sequence viewer</h3>
                <button type="button" className="btn btn-secondary btn-sm" onClick={copySequence}>
                  {copied ? "Copied" : "Copy sequence"}
                </button>
              </div>
              <div className="legend">
                {Object.keys(GROUP_LABELS).map((group) => (
                  <span key={group} className="legend-item">
                    <i className={`dot res-${group}`} />
                    {GROUP_LABELS[group]}
                  </span>
                ))}
              </div>
              <div className="seq-viewer" role="region" aria-label="Annotated sequence" tabIndex={0}>
                {sequenceLines.map((line) => (
                  <div className="seq-line" key={line.start}>
                    <span className="seq-index">{line.start}</span>
                    {line.blocks.map((block, blockIndex) => (
                      <span className="seq-block" key={blockIndex}>
                        {block.split("").map((acid, acidIndex) => (
                          <span key={acidIndex} className={`res res-${residueGroup(acid)}`}>
                            {acid}
                          </span>
                        ))}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
            </article>

            <article className="panel span-12">
              <h3>Amino acid composition</h3>
              <div className="group-strip" aria-label="Composition by residue class">
                {Object.keys(GROUP_LABELS).map((group) => (
                  <div
                    key={group}
                    className={`group-segment res-${group}`}
                    style={{ flexGrow: Math.max(groupTotals[group], 0.01) }}
                    title={`${GROUP_LABELS[group]}: ${formatPercent(groupTotals[group])}`}
                  >
                    <span>{GROUP_LABELS[group]}</span>
                    <strong>{formatPercent(groupTotals[group])}</strong>
                  </div>
                ))}
              </div>
              <div className="amino-grid">
                {aminoBars.map((item) => (
                  <div key={item.acid} className="amino-row">
                    <span className={`acid-token res-${residueGroup(item.acid)}`}>{item.acid}</span>
                    <div className="bar-track">
                      <div className={`bar-fill fill-${residueGroup(item.acid)}`} style={{ width: `${Math.max(item.value > 0 ? 1 : 0, item.value * 100)}%` }} />
                    </div>
                    <span className="acid-value">{formatPercent(item.value)}</span>
                  </div>
                ))}
              </div>
            </article>

            {invalidResidues.length ? (
              <article className="panel span-12 warning-card">
                <h3>Non-standard residues removed</h3>
                <p>
                  {invalidResidues.join(", ")} were excluded from the analysis and replaced with an unknown token for prediction.
                </p>
              </article>
            ) : null}
          </section>
        ) : null}
      </main>

      <footer className="footer">SequenceView &middot; trained on reviewed UniProt entries &middot; for research use only</footer>
    </div>
  );
}
