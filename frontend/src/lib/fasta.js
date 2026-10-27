export function parseFastaRecords(text) {
  const records = [];
  for (const rawLine of String(text).split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    if (line.startsWith(">")) {
      records.push({ id: line.slice(1).trim() || `record_${records.length + 1}`, parts: [] });
    } else {
      if (!records.length) {
        records.push({ id: "sequence", parts: [] });
      }
      records[records.length - 1].parts.push(line);
    }
  }

  return records
    .map(({ id, parts }) => ({ id, sequence: parts.join("").replace(/\s/g, "").toUpperCase() }))
    .filter((record) => record.sequence);
}
