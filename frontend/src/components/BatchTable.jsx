import { useMemo, useState } from "react";

import { formatInteger, formatNumber, formatPercent } from "../lib/format.js";

const COLUMNS = [
  { key: "index", label: "#", numeric: true },
  { key: "id", label: "Record" },
  { key: "length", label: "Length", numeric: true },
  { key: "call", label: "Call" },
  { key: "confidence", label: "Confidence", numeric: true },
  { key: "molecular_weight", label: "MW (Da)", numeric: true },
  { key: "isoelectric_point", label: "pI", numeric: true },
  { key: "warnings", label: "Warnings", numeric: true },
];

function sortValue(row, key) {
  if (key === "call") {
    return row.error ? "zzz" : row.class_name || "";
  }
  if (key === "warnings") {
    return row.warnings?.length ?? 0;
  }
  return row[key] ?? "";
}

function callOf(row) {
  return row.error ? "error" : row.class_name;
}

export default function BatchTable({ results, selected, onSelect, onExport }) {
  const [sortKey, setSortKey] = useState("index");
  const [descending, setDescending] = useState(false);
  const [filter, setFilter] = useState("all");

  const counts = useMemo(() => {
    const tally = { hydrolase: 0, oxidoreductase: 0, uncertain: 0, error: 0 };
    results.forEach((row) => {
      tally[callOf(row)] += 1;
    });
    return tally;
  }, [results]);

  const rows = useMemo(() => {
    const filtered = filter === "all" ? results : results.filter((row) => callOf(row) === filter);
    const sorted = [...filtered].sort((left, right) => {
      const a = sortValue(left, sortKey);
      const b = sortValue(right, sortKey);
      const order = typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b));
      return descending ? -order : order;
    });
    return sorted;
  }, [results, sortKey, descending, filter]);

  function toggleSort(key) {
    if (key === sortKey) {
      setDescending(!descending);
    } else {
      setSortKey(key);
      setDescending(false);
    }
  }

  return (
    <div>
      <div className="field-head">
        <p className="meta">
          {results.length} records: {counts.hydrolase} hydrolase, {counts.oxidoreductase} oxidoreductase,{" "}
          {counts.uncertain} uncertain, {counts.error} failed
        </p>
        <span className="field-tools">
          <label className="inline-label">
            Show{" "}
            <select value={filter} onChange={(event) => setFilter(event.target.value)}>
              <option value="all">all</option>
              <option value="hydrolase">hydrolase</option>
              <option value="oxidoreductase">oxidoreductase</option>
              <option value="uncertain">uncertain</option>
              <option value="error">failed</option>
            </select>
          </label>
          <button type="button" className="link" onClick={onExport}>
            Export CSV
          </button>
        </span>
      </div>

      <div className="table-scroll">
        <table className="data batch">
          <thead>
            <tr>
              {COLUMNS.map((column) => (
                <th key={column.key} className={column.numeric ? "num" : ""} aria-sort={sortKey === column.key ? (descending ? "descending" : "ascending") : "none"}>
                  <button type="button" className="sort" onClick={() => toggleSort(column.key)}>
                    {column.label}
                    {sortKey === column.key ? (descending ? " ▾" : " ▴") : ""}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.index}
                className={`clickable${selected === row.index ? " selected" : ""}`}
                onClick={() => onSelect(row.index)}
              >
                <td className="num">{row.index}</td>
                <td className="clip" title={row.id}>
                  <button type="button" className="link row-link" onClick={(event) => { event.stopPropagation(); onSelect(row.index); }}>
                    {row.id}
                  </button>
                </td>
                {row.error ? (
                  <td colSpan={6} className="error-cell">
                    {row.error}
                  </td>
                ) : (
                  <>
                    <td className="num">{formatInteger(row.length)}</td>
                    <td className={`call call-${row.is_uncertain ? "uncertain" : "called"}`}>{row.class_name}</td>
                    <td className="num">{formatPercent(row.confidence)}</td>
                    <td className="num">{formatNumber(row.molecular_weight, 1)}</td>
                    <td className="num">{formatNumber(row.isoelectric_point, 2)}</td>
                    <td className="num" title={(row.warnings || []).map((warning) => warning.message).join("\n")}>
                      {row.warnings?.length || ""}
                    </td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
