import { useMemo } from "react";

import { GROUP_LABELS, chunkSequence, residueGroup } from "../lib/residues.js";

function residueClass(acid, score, mode) {
  if (mode === "attribution" && score !== undefined) {
    return score >= 0 ? "attr-pos" : "attr-neg";
  }
  return `res-${residueGroup(acid)}`;
}

export function AttributionStrip({ scores }) {
  if (!scores?.length) {
    return null;
  }
  return (
    <svg className="strip" viewBox={`0 0 ${scores.length} 1`} preserveAspectRatio="none" role="img" aria-label="Attribution along the sequence">
      {scores.map((score, index) => (
        <rect
          key={index}
          x={index}
          width="1.02"
          height="1"
          className={score >= 0 ? "attr-pos" : "attr-neg"}
          style={{ "--w": Math.abs(score) }}
        />
      ))}
    </svg>
  );
}

export default function SequenceViewer({ sequence, scores, mode, motifMask, predictedClass, focus, onSelectPosition }) {
  const lines = useMemo(() => chunkSequence(sequence), [sequence]);
  const hasScores = Boolean(scores?.length);
  const showAttribution = mode === "attribution" && hasScores;

  return (
    <div>
      <div className="legend">
        {showAttribution ? (
          <>
            <span className="legend-swatch attr-pos" style={{ "--w": 0.8 }}>
              supports {predictedClass}
            </span>
            <span className="legend-swatch attr-neg" style={{ "--w": 0.8 }}>
              opposes {predictedClass}
            </span>
          </>
        ) : (
          Object.keys(GROUP_LABELS).map((group) => (
            <span key={group} className={`res-${group}`}>
              {GROUP_LABELS[group]}
            </span>
          ))
        )}
        {motifMask && Array.prototype.some.call(motifMask, Boolean) ? <span className="motif-legend">motif match</span> : null}
      </div>

      <div className="seq-viewer" role="region" aria-label="Annotated sequence" tabIndex={0}>
        {lines.map((line) => (
          <div className="seq-line" key={line.start}>
            <span className="seq-index">{line.start + 1}</span>
            {line.blocks.map((block) => (
              <span className="seq-block" key={block.start}>
                {block.text.split("").map((acid, offset) => {
                  const position = block.start + offset;
                  const score = hasScores ? scores[position] : undefined;
                  const classes = [residueClass(acid, score, showAttribution ? "attribution" : "class")];
                  if (hasScores && position >= scores.length) {
                    classes.push("unseen");
                  }
                  if (motifMask?.[position]) {
                    classes.push("motif");
                  }
                  if (position === focus) {
                    classes.push("focused");
                  }
                  return (
                    <span
                      key={offset}
                      className={classes.join(" ")}
                      style={showAttribution && score !== undefined ? { "--w": Math.abs(score) } : undefined}
                      title={`${acid}${position + 1}${score !== undefined ? ` (${score.toFixed(2)})` : ""}`}
                      onClick={onSelectPosition ? () => onSelectPosition(position) : undefined}
                    >
                      {acid}
                    </span>
                  );
                })}
              </span>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
