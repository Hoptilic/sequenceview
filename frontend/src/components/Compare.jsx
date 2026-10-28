import { useState } from "react";

import { requestJson } from "../lib/api.js";
import { formatNumber, formatPercent } from "../lib/format.js";

const LYSOZYME = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL";
const EXAMPLE = {
  a: `>chicken_lysozyme\n${LYSOZYME}`,
  b: `>variant\n${LYSOZYME.slice(0, 51)}ARGDAA${LYSOZYME.slice(57)}`,
};

const LINE = 60;

export function alignmentLines(first, second, width = LINE) {
  const lines = [];
  let positionA = 0;
  let positionB = 0;
  for (let offset = 0; offset < first.length; offset += width) {
    const a = first.slice(offset, offset + width);
    const b = second.slice(offset, offset + width);
    const marks = Array.from(a, (acid, index) => (acid === "-" || b[index] === "-" ? " " : acid === b[index] ? "|" : "."));
    lines.push({ a, b, marks: marks.join(""), startA: positionA + 1, startB: positionB + 1 });
    positionA += a.replaceAll("-", "").length;
    positionB += b.replaceAll("-", "").length;
  }
  return lines;
}

function Side({ title, side }) {
  const prediction = side.prediction;
  return (
    <tr>
      <th>{title}: {side.id}</th>
      <td>{side.sequence.length} aa</td>
      <td className={`call call-${prediction.is_uncertain ? "uncertain" : "called"}`}>{prediction.class_name}</td>
      <td className="num">{formatPercent(prediction.probabilities.oxidoreductase, 2)}</td>
    </tr>
  );
}

export default function Compare() {
  const [inputs, setInputs] = useState({ a: "", b: "" });
  const [state, setState] = useState({ status: "idle", error: "", result: null });

  async function run(event) {
    event.preventDefault();
    setState({ status: "loading", error: "", result: null });
    try {
      setState({ status: "ready", error: "", result: await requestJson("/api/compare", inputs) });
    } catch (error) {
      setState({ status: "error", error: error.message, result: null });
    }
  }

  const { result } = state;
  const alignment = result?.alignment;
  const lines = alignment ? alignmentLines(alignment.aligned_a, alignment.aligned_b) : [];

  return (
    <div>
      <section className="section">
        <form onSubmit={run}>
          <div className="field-head">
            <label htmlFor="compareA">Compare two sequences</label>
            <span className="field-tools">
              <button type="button" className="link" onClick={() => setInputs(EXAMPLE)}>
                Load example
              </button>
            </span>
          </div>
          <div className="compare-inputs">
            {["a", "b"].map((side) => (
              <textarea
                key={side}
                id={side === "a" ? "compareA" : "compareB"}
                value={inputs[side]}
                onChange={(event) => setInputs({ ...inputs, [side]: event.target.value })}
                placeholder={`>${side === "a" ? "reference" : "variant"}\nMKTFFVILV...`}
                spellCheck={false}
                rows={6}
                aria-label={side === "a" ? "First sequence" : "Second sequence"}
              />
            ))}
          </div>
          <div className="form-footer">
            <span className="meta">One FASTA record or plain sequence per box, up to 2000 residues each.</span>
            <button type="submit" className="button button-primary" disabled={state.status === "loading"}>
              {state.status === "loading" ? "Comparing..." : "Compare"}
            </button>
          </div>
          {state.error ? <p className="error" role="alert">{state.error}</p> : null}
        </form>
      </section>

      {result ? (
        <>
          <section className="section">
            <h2>Predictions</h2>
            <table className="data batch">
              <thead>
                <tr><th>Sequence</th><th>Length</th><th>Call</th><th className="num">P(oxidoreductase)</th></tr>
              </thead>
              <tbody>
                <Side title="A" side={result.a} />
                <Side title="B" side={result.b} />
              </tbody>
            </table>
            <p className="note">
              Change in P(oxidoreductase) from A to B: {result.delta_oxidoreductase_probability >= 0 ? "+" : ""}
              {formatNumber(result.delta_oxidoreductase_probability * 100, 2)} percentage points. Model {result.model.version}.
            </p>
          </section>

          <section className="section">
            <h2>Alignment</h2>
            <table className="data">
              <tbody>
                <tr><th>Identity (aligned positions)</th><td className="num">{formatPercent(alignment.identity)}</td></tr>
                <tr><th>Identity (alignment length)</th><td className="num">{formatPercent(alignment.identity_over_alignment)}</td></tr>
                <tr><th>Similarity (BLOSUM62 &gt; 0)</th><td className="num">{formatPercent(alignment.similarity)}</td></tr>
                <tr><th>Gap columns</th><td className="num">{alignment.gaps}</td></tr>
                <tr><th>Score</th><td className="num">{formatNumber(alignment.score, 1)}</td></tr>
              </tbody>
            </table>
            <pre className="alignment" aria-label="Pairwise alignment">
              {lines.map((line) => (
                <span key={line.startA} className="alignment-block">
                  {`A ${String(line.startA).padStart(5)}  ${line.a}\n`}
                  {`${" ".repeat(8)}  ${line.marks}\n`}
                  {`B ${String(line.startB).padStart(5)}  ${line.b}\n\n`}
                </span>
              ))}
            </pre>
            <p className="note">
              Global alignment with BLOSUM62, gap open -10 and extend -0.5. | identical, . substitution, blank gap.
            </p>
          </section>
        </>
      ) : null}
    </div>
  );
}
