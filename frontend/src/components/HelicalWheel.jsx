import { useMemo, useState } from "react";

import { formatNumber } from "../lib/format.js";
import { helicalWheel } from "../lib/secondary.js";
import { residueGroup } from "../lib/residues.js";

const SIZE = 260;
const CENTRE = SIZE / 2;
const RADIUS = 92;

export default function HelicalWheel({ sequence, onFocus }) {
  const [length, setLength] = useState(18);
  const [start, setStart] = useState(0);

  const windowLength = Math.min(length, sequence.length);
  const maxStart = Math.max(0, sequence.length - windowLength);
  const begin = Math.min(start, maxStart);
  const wheel = useMemo(() => helicalWheel(sequence, begin, windowLength), [sequence, begin, windowLength]);

  const place = (angle, radius) => ({
    x: CENTRE + radius * Math.sin(angle),
    y: CENTRE - radius * Math.cos(angle),
  });

  const arrowLength = Math.min(RADIUS - 18, wheel.moment * 55);
  const arrowTip = place(wheel.momentAngle, arrowLength);
  const nodes = wheel.points.map((point, order) => ({
    ...point,
    ...place(point.angle, RADIUS),
    label: order === 0 ? "N" : order === wheel.points.length - 1 ? "C" : "",
  }));

  return (
    <div className="wheel">
      <div className="wheel-controls">
        <label>
          Window start
          <input type="range" min="0" max={maxStart} value={begin} onChange={(event) => setStart(Number(event.target.value))} />
          <span className="num">{begin + 1}</span>
        </label>
        <label>
          Length
          <input type="range" min="8" max={Math.min(36, sequence.length)} value={windowLength} onChange={(event) => setLength(Number(event.target.value))} />
          <span className="num">{windowLength}</span>
        </label>
      </div>

      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="wheel-svg" role="img" aria-label="Helical wheel">
        <circle className="wheel-ring" cx={CENTRE} cy={CENTRE} r={RADIUS} />
        {wheel.moment > 0 ? (
          <g>
            <line className="wheel-moment" x1={CENTRE} y1={CENTRE} x2={arrowTip.x} y2={arrowTip.y} />
            <circle className="wheel-moment-tip" cx={arrowTip.x} cy={arrowTip.y} r="3.5" />
          </g>
        ) : null}
        {nodes.map((node) => (
          <g key={node.position} className="wheel-node" onClick={() => onFocus?.(node.position)}>
            <circle cx={node.x} cy={node.y} r="12" className={`wheel-bubble bubble-${residueGroup(node.acid)}`} />
            <text x={node.x} y={node.y + 4} textAnchor="middle" className="wheel-letter">
              {node.acid}
            </text>
            <text x={node.x} y={node.y - 15} textAnchor="middle" className="wheel-index">
              {node.position + 1}
              {node.label}
            </text>
          </g>
        ))}
      </svg>

      <p className="note">
        Residues {begin + 1} to {begin + windowLength} drawn at 100 degrees per residue, as in an alpha helix. The line from
        the centre is the hydrophobic moment (Eisenberg scale), mu_H = {formatNumber(wheel.moment, 2)}. Values above roughly
        0.35 suggest an amphipathic helix, but only if the window really is helical.
      </p>
    </div>
  );
}
