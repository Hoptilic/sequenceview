export function formatPercent(value, decimals = 1) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(decimals)}%` : "-";
}

export function formatNumber(value, decimals = 3) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(decimals) : "-";
}

export function formatInteger(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat().format(Math.round(value))
    : "-";
}

export function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function csvValue(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function toCsv(rows) {
  return rows.map((row) => row.map(csvValue).join(",")).join("\n");
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
