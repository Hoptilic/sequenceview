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
  positive: "Basic",
  negative: "Acidic",
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
      { label: "Hydrolase", ec: "EC 3.*", value: Number(probs.hydrolase || 0), tone: "a" },
      { label: "Oxidoreductase", ec: "EC 1.*", value: Number(probs.oxidoreductase || 0), tone: "b" },
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
  const verdictTone = isUncertain ? "uncertain" : "called";
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
        { label: "Instability index", value: formatNumber(instabilityIndex, 2), hint: isLikelyStable === null ? null : isLikelyStable ? "below 40: likely stable" : "40 or above: possibly unstable" },
        { label: "GRAVY", value: formatNumber(gravy, 3), hint: typeof gravy === "number" ? (gravy > 0 ? "overall hydrophobic" : "overall hydrophilic") : null },
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
    <div className="page">
      <header className="header">
        <h1>SequenceView</h1>
        <p>Hydrolase / oxidoreductase classification and basic physicochemical analysis of protein sequences.</p>
      </header>

      <main>
        <section className="section">
          <form onSubmit={handleAnalyze}>
            <div className="field-head">
              <label htmlFor="sequenceInput">Input sequence (plain or FASTA)</label>
              <span className="field-tools">
                <button type="button" className="link" onClick={() => loadText(EXAMPLE_SEQUENCE, "lysozyme-example.fasta")}>
                  Load example
                </button>
                {sequenceInput ? (
                  <button type="button" className="link" onClick={handleClear}>
                    Clear
                  </button>
                ) : null}
              </span>
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
            </div>

            <div className="form-footer">
              <span className="meta">
                {formatInteger(inputStats.residues)} characters
                {inputStats.headers > 1 ? ` | ${inputStats.headers} FASTA records will be merged` : ""}
                {filename ? ` | ${filename}` : ""}
              </span>
              <span className="actions">
                <label className="button" htmlFor="sequenceFile">
                  Upload file
                </label>
                <input
                  id="sequenceFile"
                  type="file"
                  accept=".fasta,.fa,.faa,.txt"
                  onChange={handleFileUpload}
                  className="hidden-file-input"
                />
                <button type="submit" className="button button-primary" disabled={loading}>
                  {loading ? "Running..." : "Run analysis"}
                </button>
              </span>
            </div>

            {error ? (
              <p className="error" role="alert">
                {error}
              </p>
            ) : null}
          </form>
        </section>

        {result ? (
          <div ref={resultsRef}>
            <section className="section">
              <h2>Prediction</h2>
              <table className="data">
                <tbody>
                  <tr>
                    <th>Call</th>
                    <td className={`call call-${verdictTone}`}>{verdict}</td>
                  </tr>
                  <tr>
                    <th>Top class</th>
                    <td>{prediction.predicted_class}</td>
                  </tr>
                  <tr>
                    <th>Confidence</th>
                    <td>
                      {formatPercent(confidence)} (threshold {formatPercent(prediction.confidence_threshold)})
                    </td>
                  </tr>
                  <tr>
                    <th>Margin</th>
                    <td>{formatPercent(prediction.margin)}</td>
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
            </section>

            <section className="section">
              <h2>Sequence properties</h2>
              <table className="data">
                <tbody>
                  {stats.map((item) => (
                    <tr key={item.label}>
                      <th>{item.label}</th>
                      <td className="num">{item.value}</td>
                      <td className="note">{item.hint || ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section className="section">
              <div className="field-head">
                <h2>Sequence</h2>
                <span className="field-tools">
                  <button type="button" className="link" onClick={copySequence}>
                    {copied ? "Copied" : "Copy"}
                  </button>
                </span>
              </div>
              <div className="legend">
                {Object.keys(GROUP_LABELS).map((group) => (
                  <span key={group} className={`res-${group}`}>
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
                          <span key={acidIndex} className={`res-${residueGroup(acid)}`}>
                            {acid}
                          </span>
                        ))}
                      </span>
                    ))}
                  </div>
                ))}
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

            {invalidResidues.length ? (
              <section className="section notice">
                Non-standard residues ({invalidResidues.join(", ")}) were excluded from the property calculations and
                encoded as unknown for the classifier.
              </section>
            ) : null}

            <section className="section">
              <h2>Export</h2>
              <span className="actions">
                <button type="button" className="button" onClick={exportJsonReport}>
                  JSON
                </button>
                <button type="button" className="button" onClick={exportCsvReport}>
                  CSV
                </button>
              </span>
            </section>
          </div>
        ) : null}
      </main>

      <footer className="footer">Predictions are from a baseline model trained on UniProt annotations and may be wrong.</footer>
    </div>
  );
}
