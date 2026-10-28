import { useMemo } from "react";

import { estimateSecondaryStructure, secondarySegments } from "../lib/secondary.js";

const HEIGHT = 18;

export default function Ribbon({ sequence }) {
  const segments = useMemo(() => secondarySegments(estimateSecondaryStructure(sequence)), [sequence]);

  return (
    <div className="ribbon-block">
      <svg className="ribbon" viewBox={`0 0 ${sequence.length} ${HEIGHT}`} preserveAspectRatio="none" role="img" aria-label="Estimated secondary structure">
        {segments.map((segment) => {
          const width = segment.end - segment.start;
          if (segment.state === "H") {
            return <rect key={segment.start} className="ss-helix" x={segment.start} y="2" width={width} height="14" rx="3" />;
          }
          if (segment.state === "E") {
            const head = Math.min(width, 4);
            const x2 = segment.end;
            const x1 = segment.start;
            return (
              <polygon
                key={segment.start}
                className="ss-strand"
                points={`${x1},5 ${x2 - head},5 ${x2 - head},1 ${x2},9 ${x2 - head},17 ${x2 - head},13 ${x1},13`}
              />
            );
          }
          return <line key={segment.start} className="ss-coil" x1={segment.start} x2={segment.end} y1="9" y2="9" />;
        })}
      </svg>
      <div className="legend">
        <span className="ss-key ss-key-helix">helix</span>
        <span className="ss-key ss-key-strand">strand</span>
        <span>coil</span>
        <span className="note-inline">Chou-Fasman propensity estimate from sequence alone, not a structure prediction.</span>
      </div>
    </div>
  );
}
