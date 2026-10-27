import { useEffect, useState } from "react";

import { requestJson } from "../lib/api.js";
import { formatInteger, formatNumber, formatPercent } from "../lib/format.js";

function ConfusionMatrix({ matrix }) {
  const [[tn, fp], [fn, tp]] = matrix;
  return (
    <table className="data data-compact confusion">
      <thead>
        <tr>
          <th />
          <th className="num">Predicted hydrolase</th>
          <th className="num">Predicted oxidoreductase</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Actual hydrolase</th>
          <td className="num">{formatInteger(tn)}</td>
          <td className="num">{formatInteger(fp)}</td>
        </tr>
        <tr>
          <th>Actual oxidoreductase</th>
          <td className="num">{formatInteger(fn)}</td>
          <td className="num">{formatInteger(tp)}</td>
        </tr>
      </tbody>
    </table>
  );
}

export default function ModelPanel() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    requestJson("/api/model")
      .then(setInfo)
      .catch((requestError) => setError(requestError.message));
  }, []);

  if (error) {
    return <p className="error">{error}</p>;
  }
  if (!info) {
    return <p className="meta">Loading model details...</p>;
  }
  if (!info.loaded) {
    return <p className="error">No model checkpoint is loaded ({info.checkpoint_path}).</p>;
  }

  const metadata = info.metadata || {};
  const metrics = metadata.test_metrics;
  const calibration = metadata.calibration;

  return (
    <div>
      <section className="section">
        <h2>Checkpoint</h2>
        <table className="data">
          <tbody>
            <tr><th>Version</th><td className="num">{info.version}</td></tr>
            <tr><th>SHA-256</th><td className="num wrap">{info.checkpoint_sha256}</td></tr>
            <tr><th>Architecture</th><td>Bidirectional GRU with mean and max pooling</td></tr>
            <tr><th>Classes</th><td>{info.classes.join(", ")}</td></tr>
            <tr><th>Maximum input length</th><td className="num">{info.max_length}</td></tr>
            <tr><th>Training examples per class</th><td className="num">{formatInteger(metadata.hydrolase_count)} hydrolase, {formatInteger(metadata.oxidoreductase_count)} oxidoreductase (before filtering)</td></tr>
            <tr><th>Confidence threshold</th><td className="num">{formatPercent(info.confidence_threshold)}</td></tr>
          </tbody>
        </table>
      </section>

      <section className="section">
        <h2>Held-out test performance</h2>
        {metrics ? (
          <>
            <table className="data data-compact">
              <tbody>
                <tr><th>Accuracy</th><td className="num">{formatPercent(metrics.accuracy, 2)}</td></tr>
                <tr><th>Precision</th><td className="num">{formatPercent(metrics.precision, 2)}</td></tr>
                <tr><th>Recall</th><td className="num">{formatPercent(metrics.recall, 2)}</td></tr>
                <tr><th>Specificity</th><td className="num">{formatPercent(metrics.specificity, 2)}</td></tr>
                <tr><th>F1</th><td className="num">{formatNumber(metrics.f1, 4)}</td></tr>
                <tr><th>Loss</th><td className="num">{formatNumber(metrics.loss, 4)}</td></tr>
                {metrics.examples ? <tr><th>Test examples</th><td className="num">{formatInteger(metrics.examples)}</td></tr> : null}
              </tbody>
            </table>
            <p className="note">Oxidoreductase is the positive class. Precision and recall refer to it.</p>
            {metrics.confusion_matrix ? (
              <ConfusionMatrix matrix={metrics.confusion_matrix} />
            ) : (
              <p className="note">
                Confusion matrix counts are not stored in this checkpoint. Retrain with the current notebook to record them.
              </p>
            )}
          </>
        ) : (
          <p className="note">This checkpoint has no stored test metrics.</p>
        )}
      </section>

      <section className="section">
        <h2>Calibration</h2>
        {calibration ? (
          <table className="data data-compact">
            <tbody>
              <tr><th>Temperature</th><td className="num">{formatNumber(calibration.temperature, 3)}</td></tr>
              <tr><th>Test ECE before</th><td className="num">{formatNumber(calibration.test_ece_before, 4)}</td></tr>
              <tr><th>Test ECE after</th><td className="num">{formatNumber(calibration.test_ece_after, 4)}</td></tr>
            </tbody>
          </table>
        ) : (
          <p className="note">
            This checkpoint is not calibrated (temperature {formatNumber(info.temperature, 2)}), so confidences are the raw
            softmax output and tend to be overconfident. Retrain with the current notebook to fit a temperature on the
            validation split.
          </p>
        )}
      </section>

      <section className="section">
        <h2>Training data and limits</h2>
        <ul className="plain-list">
          <li>Reviewed UniProt entries with evidence at protein level, 50 to 1000 residues.</li>
          <li>Labels are the first EC digit: 1 (oxidoreductase) and 3 (hydrolase). Entries with both were removed.</li>
          <li>Train, validation and test splits are grouped by a k-mer signature to reduce near-duplicate leakage. This is not a full sequence-identity cluster split, so test scores may be optimistic for remote homologs.</li>
          <li>Annotation-derived labels contain noise. The model cannot recognise enzymes outside these two classes and will still assign one of them.</li>
          <li>Per-residue attribution is gradient times input on the predicted-class logit. It indicates sensitivity, not mechanism.</li>
        </ul>
      </section>
    </div>
  );
}
