export default function MotifHits({ hits, predictedClass, selectedKey, onSelect }) {
  if (!hits.length) {
    return (
      <p className="note">
        No signature from the bundled library matches. That is common and does not argue against the prediction.
      </p>
    );
  }

  return (
    <div>
      <div className="table-scroll">
        <table className="data batch">
          <thead>
            <tr>
              <th>Signature</th>
              <th>Source</th>
              <th>Typical class</th>
              <th>Positions</th>
              <th>Versus call</th>
            </tr>
          </thead>
          <tbody>
            {hits.map((hit) => {
              const key = `${hit.source}:${hit.name}`;
              const agrees = hit.class === predictedClass;
              return (
                <tr key={key} className={`clickable${selectedKey === key ? " selected" : ""}`} onClick={() => onSelect(selectedKey === key ? null : hit)}>
                  <td className="clip" title={hit.pattern}>{hit.name}</td>
                  <td>{hit.source}</td>
                  <td>{hit.class}</td>
                  <td className="num">
                    {hit.matches.map((match) => `${match.start + 1}-${match.end}`).join(", ")}
                  </td>
                  <td className={agrees ? "" : "differs"}>{agrees ? "agrees" : "differs"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="note">
        Select a row to mark its match on the sequence and structure. Entries labelled heuristic are loose structural motifs
        that occur by chance in many proteins. A matching signature supports a class only loosely; the library is small and
        its absence says nothing.
      </p>
    </div>
  );
}
