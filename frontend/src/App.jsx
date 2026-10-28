import { useEffect, useMemo, useRef, useState } from "react";

import BatchTable from "./components/BatchTable.jsx";
import Compare from "./components/Compare.jsx";
import Detail from "./components/Detail.jsx";
import EmbeddingPlot from "./components/EmbeddingPlot.jsx";
import ModelPanel from "./components/ModelPanel.jsx";
import { requestJson } from "./lib/api.js";
import { parseFastaRecords } from "./lib/fasta.js";
import { downloadFile, formatInteger } from "./lib/format.js";
import { buildBatchCsv } from "./lib/report.js";
import { decodeSequenceHash, encodeSequenceHash } from "./lib/share.js";

const EXAMPLE_SEQUENCE = `>sp|P00698|LYSC_CHICK Lysozyme C
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL`;

function readTextFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Failed to read file."));
    reader.readAsText(file);
  });
}

export default function App() {
  const [tab, setTab] = useState("analyse");
  const [sequenceInput, setSequenceInput] = useState("");
  const [filename, setFilename] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [detail, setDetail] = useState(null);
  const [batch, setBatch] = useState(null);
  const [selected, setSelected] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [shareState, setShareState] = useState("");
  const sequenceInputRef = useRef(null);
  const resultsRef = useRef(null);

  const records = useMemo(() => parseFastaRecords(sequenceInput), [sequenceInput]);
  const residueTotal = useMemo(() => records.reduce((total, record) => total + record.sequence.length, 0), [records]);

  useEffect(() => {
    const textarea = sequenceInputRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight + 2, 160), 420)}px`;
    }
  }, [sequenceInput, tab]);

  useEffect(() => {
    const shared = decodeSequenceHash(window.location.hash);
    if (shared) {
      setSequenceInput(shared);
      run(shared);
    }
    // Only runs once on load to restore a shared link.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function scrollToResults() {
    setTimeout(() => resultsRef.current?.scrollIntoView({ block: "start" }), 0);
  }

  async function predictOne(text) {
    return requestJson("/api/predict", { sequence: text });
  }

  async function run(text = sequenceInput) {
    const parsed = parseFastaRecords(text);
    if (!parsed.length) {
      setError("Paste a sequence or upload a FASTA file first.");
      return;
    }

    setLoading(true);
    setError("");
    setShareState("");
    setSelected(null);

    try {
      if (parsed.length === 1) {
        setBatch(null);
        setDetail(await predictOne(text));
      } else {
        const data = await requestJson("/api/predict/batch", { sequence: text });
        setDetail(null);
        setBatch(data);
      }
      scrollToResults();
    } catch (requestError) {
      setDetail(null);
      setBatch(null);
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function openRecord(index) {
    const row = batch.results.find((item) => item.index === index);
    if (!row || row.error) {
      return;
    }

    setLoading(true);
    setError("");
    setShareState("");
    try {
      setDetail(await predictOne(`>${row.id}\n${row.sequence}`));
      setSelected(index);
      scrollToResults();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

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
    setDetail(null);
    setBatch(null);
    setSelected(null);
    window.history.replaceState(null, "", window.location.pathname + window.location.search);
  }

  async function handleShare() {
    const text = detail ? `>${detail.input.id}\n${detail.input.sequence}` : sequenceInput;
    const hash = encodeSequenceHash(text);
    if (!hash) {
      setShareState("too-long");
      return;
    }

    window.history.replaceState(null, "", `#${hash}`);
    try {
      await navigator.clipboard.writeText(window.location.href);
      setShareState("copied");
    } catch {
      setShareState("failed");
    }
  }

  const multiple = records.length > 1;

  return (
    <div className="page">
      <header className="header">
        <h1>SequenceView</h1>
        <p>Hydrolase / oxidoreductase classification and basic physicochemical analysis of protein sequences.</p>
        <nav className="tabs" aria-label="Sections">
          <button type="button" className={tab === "analyse" ? "tab active" : "tab"} onClick={() => setTab("analyse")}>
            Analyse
          </button>
          <button type="button" className={tab === "compare" ? "tab active" : "tab"} onClick={() => setTab("compare")}>
            Compare
          </button>
          <button type="button" className={tab === "model" ? "tab active" : "tab"} onClick={() => setTab("model")}>
            Model
          </button>
        </nav>
      </header>

      <main>
        {tab === "model" ? <ModelPanel /> : null}
        {tab === "compare" ? <Compare /> : null}

        {tab === "analyse" ? (
          <>
            <section className="section">
              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  run();
                }}
              >
                <div className="field-head">
                  <label htmlFor="sequenceInput">Input sequence (plain or FASTA, multiple records allowed)</label>
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
                    {records.length} {records.length === 1 ? "record" : "records"}, {formatInteger(residueTotal)} residues
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
                      {loading ? "Running..." : multiple ? `Run ${records.length} records` : "Run analysis"}
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

            <div ref={resultsRef}>
              {batch ? (
                <section className="section">
                  <h2>Batch results</h2>
                  <BatchTable
                    results={batch.results}
                    selected={selected}
                    onSelect={openRecord}
                    onExport={() => downloadFile(buildBatchCsv(batch.results), "sequenceview-batch.csv", "text/csv;charset=utf-8")}
                  />
                  <p className="note stamp">Model {batch.model.version}. Select a record for the full report.</p>
                </section>
              ) : null}

              {batch?.projection ? (
                <section className="section">
                  <h2>Embedding map</h2>
                  <EmbeddingPlot projection={batch.projection} results={batch.results} selected={selected} onSelect={openRecord} />
                </section>
              ) : null}

              {detail ? (
                <>
                  {batch ? (
                    <section className="section">
                      <button type="button" className="link" onClick={() => { setDetail(null); setSelected(null); }}>
                        Close record
                      </button>
                      <span className="meta"> {detail.input.id}</span>
                    </section>
                  ) : null}
                  <Detail key={`${detail.input.id}:${detail.analysis.length}:${detail.attribution?.scores?.[0]}`} result={detail} onShare={handleShare} shareState={shareState} />
                </>
              ) : null}
            </div>
          </>
        ) : null}
      </main>

      <footer className="footer">Predictions are from a baseline model trained on UniProt annotations and may be wrong.</footer>
    </div>
  );
}
