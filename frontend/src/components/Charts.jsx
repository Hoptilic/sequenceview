const WIDTH = 420;
const HEIGHT = 190;
const MARGIN = { top: 10, right: 12, bottom: 34, left: 44 };

function niceTicks(low, high, count = 5) {
  const span = high - low || 1;
  const rough = span / count;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 5, 10].map((factor) => factor * magnitude).find((candidate) => candidate >= rough) || rough;
  const ticks = [];
  for (let tick = Math.ceil(low / step) * step; tick <= high + step * 1e-6; tick += step) {
    ticks.push(Number(tick.toFixed(10)));
  }
  return ticks;
}

export function LinePlot({ points, xLabel, yLabel, marker, markerLabel, label }) {
  if (!points.length) {
    return <p className="note">Not enough residues to plot.</p>;
  }

  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yLow = Math.min(...ys, 0);
  const yHigh = Math.max(...ys, 0);
  const yPad = (yHigh - yLow || 1) * 0.08;
  const yMin = yLow - yPad;
  const yMax = yHigh + yPad;

  const innerWidth = WIDTH - MARGIN.left - MARGIN.right;
  const innerHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
  const scaleX = (value) => MARGIN.left + ((value - xMin) / (xMax - xMin || 1)) * innerWidth;
  const scaleY = (value) => MARGIN.top + (1 - (value - yMin) / (yMax - yMin || 1)) * innerHeight;

  const path = points.map(([x, y], index) => `${index ? "L" : "M"}${scaleX(x).toFixed(1)},${scaleY(y).toFixed(1)}`).join("");

  return (
    <svg className="plot" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={label}>
      {niceTicks(yMin, yMax).map((tick) => (
        <g key={`y${tick}`}>
          <line className="plot-grid" x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={scaleY(tick)} y2={scaleY(tick)} />
          <text className="plot-tick" x={MARGIN.left - 6} y={scaleY(tick) + 3} textAnchor="end">
            {tick}
          </text>
        </g>
      ))}
      {niceTicks(xMin, xMax).map((tick) => (
        <text key={`x${tick}`} className="plot-tick" x={scaleX(tick)} y={HEIGHT - MARGIN.bottom + 14} textAnchor="middle">
          {tick}
        </text>
      ))}
      <line className="plot-zero" x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={scaleY(0)} y2={scaleY(0)} />
      {typeof marker === "number" && marker >= xMin && marker <= xMax ? (
        <g>
          <line className="plot-marker" x1={scaleX(marker)} x2={scaleX(marker)} y1={MARGIN.top} y2={HEIGHT - MARGIN.bottom} />
          <text className="plot-tick" x={scaleX(marker) + 4} y={MARGIN.top + 10}>
            {markerLabel}
          </text>
        </g>
      ) : null}
      <path className="plot-line" d={path} />
      <text className="plot-axis" x={MARGIN.left + innerWidth / 2} y={HEIGHT - 4} textAnchor="middle">
        {xLabel}
      </text>
      <text className="plot-axis" transform={`translate(11 ${MARGIN.top + innerHeight / 2}) rotate(-90)`} textAnchor="middle">
        {yLabel}
      </text>
    </svg>
  );
}

export function HydropathyPlot({ profile }) {
  const half = Math.floor((profile?.window || 9) / 2);
  const points = (profile?.values || []).map((value, index) => [index + 1 + half, value]);
  return (
    <LinePlot
      points={points}
      xLabel="Residue (window centre)"
      yLabel="Hydropathy (Kyte-Doolittle)"
      label={`Hydropathy profile, window ${profile?.window}`}
    />
  );
}

export function ChargePlot({ curve, isoelectricPoint }) {
  return (
    <LinePlot
      points={curve || []}
      xLabel="pH"
      yLabel="Net charge"
      marker={isoelectricPoint}
      markerLabel={`pI ${isoelectricPoint?.toFixed(2)}`}
      label="Net charge versus pH"
    />
  );
}
