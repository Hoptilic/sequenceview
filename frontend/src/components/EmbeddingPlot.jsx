const WIDTH = 520;
const HEIGHT = 320;
const MARGIN = { top: 12, right: 14, bottom: 38, left: 44 };

function ticks(low, high) {
  const span = high - low || 1;
  const step = 10 ** Math.floor(Math.log10(span / 4));
  const chosen = [1, 2, 5, 10].map((factor) => factor * step).find((candidate) => span / candidate <= 6) || step;
  const values = [];
  for (let tick = Math.ceil(low / chosen) * chosen; tick <= high; tick += chosen) {
    values.push(Number(tick.toFixed(8)));
  }
  return values;
}

export default function EmbeddingPlot({ projection, results, selected, onSelect }) {
  const byIndex = new Map(results.map((row) => [row.index, row]));
  const points = projection.points.map((point) => ({ ...point, row: byIndex.get(point.index) }));

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const padX = (Math.max(...xs) - Math.min(...xs) || 1) * 0.08;
  const padY = (Math.max(...ys) - Math.min(...ys) || 1) * 0.08;
  const xMin = Math.min(...xs) - padX;
  const xMax = Math.max(...xs) + padX;
  const yMin = Math.min(...ys) - padY;
  const yMax = Math.max(...ys) + padY;

  const innerWidth = WIDTH - MARGIN.left - MARGIN.right;
  const innerHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
  const scaleX = (value) => MARGIN.left + ((value - xMin) / (xMax - xMin)) * innerWidth;
  const scaleY = (value) => MARGIN.top + (1 - (value - yMin) / (yMax - yMin)) * innerHeight;
  const [first, second] = projection.explained_variance;

  return (
    <div>
      <svg className="plot scatter" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Embedding map of the batch">
        {ticks(yMin, yMax).map((tick) => (
          <g key={`y${tick}`}>
            <line className="plot-grid" x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={scaleY(tick)} y2={scaleY(tick)} />
            <text className="plot-tick" x={MARGIN.left - 6} y={scaleY(tick) + 3} textAnchor="end">{tick}</text>
          </g>
        ))}
        {ticks(xMin, xMax).map((tick) => (
          <text key={`x${tick}`} className="plot-tick" x={scaleX(tick)} y={HEIGHT - MARGIN.bottom + 14} textAnchor="middle">{tick}</text>
        ))}
        {points.map((point) => {
          const tone = point.row?.is_uncertain ? "uncertain" : point.row?.predicted_class === "oxidoreductase" ? "b" : "a";
          return (
            <g key={point.index} className="scatter-point" onClick={() => onSelect(point.index)}>
              <circle cx={scaleX(point.x)} cy={scaleY(point.y)} r={selected === point.index ? 7 : 5} className={`dot-${tone}${selected === point.index ? " dot-selected" : ""}`} />
              <title>{`${point.index}. ${point.row?.id ?? ""}: ${point.row?.class_name ?? ""}`}</title>
            </g>
          );
        })}
        <text className="plot-axis" x={MARGIN.left + innerWidth / 2} y={HEIGHT - 4} textAnchor="middle">
          {`PC1 (${(first * 100).toFixed(0)}% of variance)`}
        </text>
        <text className="plot-axis" transform={`translate(11 ${MARGIN.top + innerHeight / 2}) rotate(-90)`} textAnchor="middle">
          {`PC2 (${(second * 100).toFixed(0)}%)`}
        </text>
      </svg>
      <div className="legend">
        <span className="dot-key dot-key-a">hydrolase</span>
        <span className="dot-key dot-key-b">oxidoreductase</span>
        <span className="dot-key dot-key-uncertain">uncertain</span>
      </div>
      <p className="note">
        Each point is one record, placed by the first two principal components of the classifier's pooled features (the vector
        its output layer reads). Nearby records look similar to the network; this is not an evolutionary distance.
      </p>
    </div>
  );
}
